#include "dressi/dressi_ad.h"

#include <map>

#include <algorithm>
#include <chrono>
#include <optional>

#include <spdlog/spdlog.h>

#include "codegen/glsl_codegen.h"
#include "core/build_backward.h"
#include "core/node.h"
#include "dressi/f.h"
#include "pack/packing.h"
#include "pack/reactive.h"
#include "pack/substage.h"
#include "vk/context.h"
#include "vk/executor.h"
#include "vk/transfer.h"

namespace dressi {

struct DressiAD::Impl {
    VkContextPtr ctx;
    PackingMode packing_mode = PackingMode::RSP;

    InitStatus init_status = BACKWARD;
    Variable loss_var{nullptr};   // last variable of the forward pass
    Optimizer optim_func;

    Variables input_vars;         // requires-grad leaves of the forward pass
    Variables input_grad_vars;    // their gradient variables
    Variables updated_vars;       // optimizer outputs
    Functions all_funcs;          // full computational graph, topo order
    Variables leaf_vars;          // variables without creators (data inputs)
    std::map<Variable, Variable> upd_inp_map;  // updated -> input copy-back
    SubStages substages;
    Stages stages;

    // Reactive cache state
    uint32_t fast_rebuild_count = 2;
    uint32_t full_rebuild_count = 8;
    uint32_t graph_static_cnt = 0;
    std::set<uint64_t> prev_dirty_ids;
    bool pruning_active = false;

    GpuPlan plan;
    bool plan_valid = false;
    std::map<Variable, CpuImage> pending_sends;
    // Image uploads to fuse into the next execute's single submit (the
    // eager binding path; borrowed views valid for the execStep call)
    std::vector<std::pair<Variable, CpuImageView>> batch_uploads;
    // Optional outputs copied after the render plan in the same submit.
    Variables batch_downloads;
    std::optional<CpuImage> batch_download_result;

    void ensureCtx() {
        if (!ctx) {
            // DRESSI_VK_DEBUG=1 turns on the validation layers
            const char* dbg = std::getenv("DRESSI_VK_DEBUG");
            ctx = CreateVkContext(dbg != nullptr && dbg[0] == '1');
        }
    }

    bool grad_outputs_enabled = false;  // keep input_grad_vars as roots
    Variables extra_outputs;  // user-demanded exports (markOutput)
    // Extra end-of-frame copy-backs: {input leaf, updated} pairs for
    // GPU-resident optimizer state (addUpdate)
    std::vector<std::pair<Variable, Variable>> extra_updates;

    Variables rootVars() const {
        Variables roots = updated_vars;
        roots.push_back(loss_var);
        for (const auto& v : extra_outputs) {
            roots.push_back(v);
        }
        for (const auto& [inp, upd] : extra_updates) {
            (void)inp;
            roots.push_back(upd);
        }
        if (grad_outputs_enabled) {
            for (const auto& g : input_grad_vars) {
                roots.push_back(g);
            }
        }
        return roots;
    }

    std::set<uint64_t> dirtyLeafIds() const {
        std::set<uint64_t> ids;
        for (const auto& v : leaf_vars) {
            if (v.IsDirty()) {
                ids.insert(v.id());
            }
        }
        return ids;
    }

    // Seeds of per-iteration change: dirty leaves plus optimizer parameters
    // (their images are overwritten by the copy-back every frame)
    Variables dynamicSeeds() const {
        Variables seeds;
        for (const auto& v : leaf_vars) {
            if (v.IsDirty()) {
                seeds.push_back(v);
            }
        }
        if (optim_func) {
            for (const auto& v : input_vars) {
                seeds.push_back(v);
            }
        }
        for (const auto& [inp, upd] : extra_updates) {
            (void)upd;
            seeds.push_back(inp);
        }
        return seeds;
    }

    PackingLimits packingLimits() const {
        PackingLimits limits;
        const auto& dl = ctx->limits;
        limits.max_input_attachments =
                std::min(16u, dl.maxPerStageDescriptorInputAttachments);
        limits.max_output_attachments = std::min(8u, dl.maxColorAttachments);
        limits.max_sampled_images =
                std::min(32u, dl.maxPerStageDescriptorSampledImages);
        // A render pass has no hard Vulkan cap on total attachments; a big
        // budget lets long same-pixel chains stay in one render pass as
        // chained subpasses (ByRegion deps) instead of separate passes
        limits.max_stage_attachments = 64;
        return limits;
    }
};

DressiAD::DressiAD() : m_impl(std::make_unique<Impl>()) {}

DressiAD::DressiAD(VkContextPtr ctx) : m_impl(std::make_unique<Impl>()) {
    DRESSI_CHECK(ctx, "DressiAD: null VkContext");
    m_impl->ctx = std::move(ctx);
}

DressiAD::~DressiAD() = default;

VkContextPtr DressiAD::createContext(bool debug) {
    const char* dbg = std::getenv("DRESSI_VK_DEBUG");
    return CreateVkContext(debug || (dbg != nullptr && dbg[0] == '1'));
}

void DressiAD::setLossVar(const Variable& loss_var) {
    m_impl->loss_var = loss_var;
    m_impl->init_status = BACKWARD;  // execute the full initialize process
}

void DressiAD::setOptimizer(const Optimizer& optim_func) {
    m_impl->optim_func = optim_func;
    m_impl->init_status = BACKWARD;
}

void DressiAD::setPackingMode(PackingMode mode) {
    m_impl->packing_mode = mode;
    if (m_impl->init_status > SUBSTAGE) {
        m_impl->init_status = SUBSTAGE;
    }
}

void DressiAD::setRebuildCounts(uint32_t fast, uint32_t full) {
    m_impl->fast_rebuild_count = fast;
    m_impl->full_rebuild_count = full;
}

void DressiAD::markOutput(const Variable& var) {
    DRESSI_CHECK(var, "markOutput: null Variable");
    m_impl->extra_outputs.push_back(var);
    if (m_impl->init_status > TRAVERSE) {
        m_impl->init_status = TRAVERSE;  // roots changed; rebuild packing
    }
}

void DressiAD::addUpdate(const Variable& input_leaf, const Variable& updated) {
    DRESSI_CHECK(input_leaf && updated, "addUpdate: null Variable");
    DRESSI_CHECK(!input_leaf.getCreator(),
                 "addUpdate: input must be a leaf Variable");
    DRESSI_CHECK(updated.getCreator(),
                 "addUpdate: updated must be a computed Variable");
    DRESSI_CHECK(input_leaf.getVType() == updated.getVType() &&
                         input_leaf.getImgSize() == updated.getImgSize(),
                 "addUpdate: updated must match the input's type and size");
    m_impl->extra_updates.emplace_back(input_leaf, updated);
    if (m_impl->init_status > OPTIMIZER) {
        m_impl->init_status = OPTIMIZER;  // copy-back set changed
    }
}

void DressiAD::setGradOutputsEnabled(bool enable) {
    if (m_impl->grad_outputs_enabled == enable) {
        return;
    }
    m_impl->grad_outputs_enabled = enable;
    if (m_impl->init_status > TRAVERSE) {
        m_impl->init_status = TRAVERSE;  // roots changed; rebuild packing
    }
}

std::vector<std::pair<Variable, Variable>> DressiAD::inputGrads() const {
    std::vector<std::pair<Variable, Variable>> pairs;
    pairs.reserve(m_impl->input_vars.size());
    for (size_t i = 0; i < m_impl->input_vars.size(); i++) {
        pairs.emplace_back(m_impl->input_vars[i], m_impl->input_grad_vars[i]);
    }
    return pairs;
}

std::string DressiAD::getDeviceName() const {
    if (!m_impl->ctx) {
        return {};
    }
    const auto props = vkw::GetPhysicalProperties(m_impl->ctx->physical_device);
    return std::string(props->deviceName.data());
}

size_t DressiAD::getFuncCount() const {
    return m_impl->all_funcs.size();
}

size_t DressiAD::getStageCount() const {
    return m_impl->stages.size();
}

size_t DressiAD::getSubStageCount() const {
    size_t n = 0;
    for (const auto& stage : m_impl->stages) {
        n += stage.substages.size();
    }
    return n;
}

namespace {

// Phase stopwatch for execStep performance logging (milliseconds)
class PhaseTimer {
public:
    using Clock = std::chrono::steady_clock;
    explicit PhaseTimer(bool enabled) : m_enabled(enabled) {}
    void mark(const char* phase) {
        if (!m_enabled) {
            return;
        }
        const auto now = Clock::now();
        const double ms =
                std::chrono::duration<double, std::milli>(now - m_last)
                        .count();
        m_last = now;
        if (ms >= 0.005) {
            m_entries.emplace_back(phase, ms);
        }
    }
    void report(const char* header) const {
        if (!m_enabled || m_entries.empty()) {
            return;
        }
        std::string line;
        double total = 0.0;
        for (const auto& [phase, ms] : m_entries) {
            line += fmt::format(" {}={:.2f}ms", phase, ms);
            total += ms;
        }
        spdlog::debug("[dressi] {} total={:.2f}ms{}", header, total, line);
    }

private:
    bool m_enabled;
    Clock::time_point m_last = Clock::now();
    std::vector<std::pair<const char*, double>> m_entries;
};

}  // namespace

void DressiAD::execStep() {
    Impl& im = *m_impl;
    DRESSI_CHECK(im.loss_var, "setLossVar() must be called before execStep()");
    im.ensureCtx();
    PhaseTimer timer(spdlog::should_log(spdlog::level::debug));
    const InitStatus entry_status = im.init_status;

    // Reactive cache: watch the dirty pattern of data leaves. A stable
    // pattern lets rebuilds prune clean cached branches; a changed pattern
    // requires re-expanding a pruned plan.
    if (im.init_status == FINISHED) {
        const std::set<uint64_t> cur_dirty = im.dirtyLeafIds();
        if (cur_dirty != im.prev_dirty_ids) {
            im.graph_static_cnt = 0;
            if (im.pruning_active) {
                im.init_status = SUBSTAGE;  // re-include newly dirty branches
            }
        } else {
            im.graph_static_cnt++;
            if (im.fast_rebuild_count != 0 &&
                im.graph_static_cnt == im.fast_rebuild_count) {
                im.init_status = STAGE;  // fast rebuild
            } else if (im.full_rebuild_count != 0 &&
                       im.graph_static_cnt == im.full_rebuild_count) {
                im.init_status = SUBSTAGE;  // full rebuild
            }
        }
        im.prev_dirty_ids = cur_dirty;
    }

    timer.mark("dirty-check");
    if (im.init_status <= BACKWARD) {
        // 1) Traverse the forward graph and generate the backward graph
        std::tie(im.input_vars, im.input_grad_vars) =
                BuildBackward(im.loss_var);
        timer.mark("backward");
    }
    if (im.init_status <= OPTIMIZER) {
        // 2) Build the optimizer to connect forward and backward passes
        im.updated_vars.clear();
        im.upd_inp_map.clear();
        if (im.optim_func) {
            im.updated_vars = im.optim_func(im.input_vars, im.input_grad_vars);
            DRESSI_CHECK(im.updated_vars.size() == im.input_vars.size(),
                         "Optimizer must return one output per input");
            for (size_t i = 0; i < im.updated_vars.size(); i++) {
                const Variable& upd = im.updated_vars[i];
                const Variable& inp = im.input_vars[i];
                DRESSI_CHECK(upd.getVType() == inp.getVType() &&
                                     upd.getImgSize() == inp.getImgSize(),
                             "Optimizer output must match its input's type "
                             "and size");
                im.upd_inp_map[upd] = inp;
            }
        }
        // Extra copy-backs for GPU-resident optimizer state (addUpdate;
        // typically registered from inside the optimizer lambda above)
        for (const auto& [inp, upd] : im.extra_updates) {
            im.upd_inp_map[upd] = inp;
        }
        timer.mark("optimizer");
    }
    if (im.init_status <= TRAVERSE) {
        // 3) Traverse the full computational graph
        im.all_funcs = TraverseFuncs(im.rootVars());
        // Data leaves: function inputs without creators (excluding inlined
        // constants)
        im.leaf_vars.clear();
        std::set<uint64_t> seen_leaves;
        for (const auto& f : im.all_funcs) {
            for (const auto& x : f.getInputVars()) {
                if (!x.getCreator() && !IsInlineConst(x) &&
                    seen_leaves.insert(x.id()).second) {
                    im.leaf_vars.push_back(x);
                }
            }
        }
        timer.mark("traverse");
    }

    const auto has_cached = [&](const Variable& v) {
        return im.plan.imgs.count(v) != 0;
    };
    const std::set<uint64_t> dynamic_ids =
            (im.init_status <= STAGE)
                    ? ComputeDynamicVarIds(im.all_funcs, im.dynamicSeeds())
                    : std::set<uint64_t>{};

    bool func_pruned = im.pruning_active;
    if (im.init_status <= SUBSTAGE) {
        // 4) Pack functions into substages, skipping clean cached branches
        const Functions exec_funcs = FilterExecutableFuncs(
                im.all_funcs, im.rootVars(), dynamic_ids, has_cached);
        size_t packable_count = 0;
        for (const auto& f : im.all_funcs) {
            if (!IsInlineConst(f.getOutputVar())) {
                packable_count++;
            }
        }
        func_pruned = exec_funcs.size() < packable_count;
        if (im.packing_mode == PackingMode::RSP) {
            im.substages = PackFunctionsIntoSubStages(
                    exec_funcs, im.rootVars(), im.packingLimits());
        } else {
            im.substages = TrivialPackSubStages(exec_funcs);
        }
        timer.mark("substage-pack");
        // Oversized 1-D leaf tables are stored 2-D-tiled (see PhysImgSize);
        // codegen folds their linear index reads to match.
        const uint32_t max_dim =
                im.ctx ? im.ctx->limits.maxImageDimension2D : 0u;
        for (auto& ss : im.substages) {
            if (ss.shader_type == RASTER) {
                const RasterShaders shaders = GenerateRasterShaders(ss, max_dim);
                ss.vert_shader_code = shaders.vert;
                ss.shader_code = shaders.frag;
            } else if (ss.shader_type == COMP) {
                ss.shader_code = GenerateCompShader(ss, max_dim);
            } else {
                ss.shader_code = GenerateFragShader(ss, max_dim);
            }
        }
        timer.mark("codegen");
    }
    if (im.init_status <= STAGE) {
        // 5) Pack the substages into stages, skipping clean cached substages
        SubStages filtered = FilterExecutableSubStages(
                im.substages, im.rootVars(), dynamic_ids, has_cached);
        im.pruning_active =
                func_pruned || filtered.size() < im.substages.size();
        if (im.packing_mode == PackingMode::RSP) {
            im.stages = PackSubStagesIntoStages(std::move(filtered),
                                                im.packingLimits());
        } else {
            im.stages = WrapSubStagesIntoStages(std::move(filtered));
        }
        timer.mark("stage-pack");
    }
    if (im.init_status <= VULKAN) {
        // 6) Parse stages into Vulkan objects (GPU-resident data persists)
        im.plan = BuildGpuPlan(*im.ctx, im.stages, im.upd_inp_map,
                               std::move(im.plan));
        im.plan_valid = true;
        timer.mark("vulkan-build");
    }

    // Flush pending CPU->GPU transfers now that images/buffers exist
    for (auto& [var, img] : im.pending_sends) {
        if (auto it = im.plan.vtx_bufs.find(var);
            it != im.plan.vtx_bufs.end()) {
            SendGeometryToBuffer(*im.ctx, it->second, img, var.getVType());
        }
        if (auto it = im.plan.imgs.find(var); it != im.plan.imgs.end()) {
            SendImageToDevice(*im.ctx, it->second, img, var.getVType(),
                              /*image_initialized=*/false);
        }
    }
    im.pending_sends.clear();

    // Fuse the batched image uploads into the execute submit -- but only on
    // the pure fast path (no rebuild this iteration), where every target
    // image already exists in ShaderReadOnlyOptimal. On a rebuild, images
    // may be freshly created (Undefined layout), so upload them normally.
    std::vector<ImageSendItem> fused;
    if (entry_status == FINISHED) {
        for (const auto& [var, view] : im.batch_uploads) {
            if (auto it = im.plan.imgs.find(var); it != im.plan.imgs.end()) {
                fused.push_back({it->second, view, var.getVType(),
                                 /*initialized=*/true});
            }
        }
    } else {
        for (const auto& [var, view] : im.batch_uploads) {
            if (auto it = im.plan.imgs.find(var); it != im.plan.imgs.end()) {
                SendImageToDevice(*im.ctx, it->second, view, var.getVType(),
                                  /*image_initialized=*/false);
            }
        }
    }
    timer.mark("upload");

    // 7) Execute the command buffer (fusing steady-state transfers)
    if (!im.batch_downloads.empty()) {
        std::vector<std::pair<vkw::ImagePackPtr, VType>> downloads;
        downloads.reserve(im.batch_downloads.size());
        for (const auto& var : im.batch_downloads) {
            DRESSI_CHECK(var, "combined recv: null Variable");
            auto it = im.plan.imgs.find(var);
            DRESSI_CHECK(it != im.plan.imgs.end(),
                         "combined recv: the Variable has no GPU image");
            downloads.emplace_back(it->second, var.getVType());
        }
        auto download = PrepareStackedImageDownload(*im.ctx, downloads);
        ExecuteGpuPlanWithTransfers(*im.ctx, im.plan, fused, download);
        im.batch_download_result =
                ResolveStackedImageDownload(*im.ctx, download);
    } else if (!fused.empty()) {
        ExecuteGpuPlanWithUploads(*im.ctx, im.plan, fused);
    } else {
        ExecuteGpuPlan(*im.ctx, im.plan);
    }
    timer.mark("gpu-execute");

    if (entry_status < FINISHED) {
        spdlog::info(
                "[dressi] rebuilt from status {} -> {} stages / {} substages "
                "({} funcs)",
                int(entry_status), im.stages.size(), getSubStageCount(),
                im.all_funcs.size());
        // DRESSI_DUMP_STAGES=1: per-substage packing detail (perf analysis)
        if (const char* dump = std::getenv("DRESSI_DUMP_STAGES");
            dump && dump[0] == '1') {
            for (size_t si = 0; si < im.stages.size(); si++) {
                const Stage& st = im.stages[si];
                spdlog::info("[dressi] stage {} type={} {}x{}", si,
                             int(st.shader_type), st.img_size.w,
                             st.img_size.h);
                for (size_t k = 0; k < st.substages.size(); k++) {
                    const SubStage& ss = st.substages[k];
                    std::string funcs;
                    for (const auto& f : ss.funcs) {
                        funcs += NodeAccess::Node(f)->name + " ";
                    }
                    std::string outs;
                    for (const auto& v : ss.out_vars) {
                        outs += std::to_string(v.id()) + " ";
                    }
                    std::string inps;
                    for (const auto& v : ss.inp_vars) {
                        inps += std::to_string(v.id()) + " ";
                    }
                    spdlog::info(
                            "[dressi]   ss{} fn={} inp=[{}] out=[{}] : {}",
                            k, ss.funcs.size(), inps, outs, funcs);
                }
            }
        }
    }
    timer.report(entry_status < FINISHED ? "execStep(rebuild)"
                                         : "execStep(cached)");

    // Everything computed this iteration is now consistent: mark all data
    // leaves (and their downstream) as clean
    if (im.init_status != FINISHED) {
        im.prev_dirty_ids = im.dirtyLeafIds();
    }
    for (auto& v : im.leaf_vars) {
        v.setDirtyRecursively(false);
    }
    im.init_status = FINISHED;
}

void DressiAD::sendImg(const Variable& var, const CpuImageView& cpu_img) {
    Impl& im = *m_impl;
    DRESSI_CHECK(var, "sendImg: null Variable");
    Variable v = var;
    v.setDirty(true);  // mark as changed
    if (im.plan_valid) {
        bool uploaded = false;
        if (auto it = im.plan.vtx_bufs.find(var);
            it != im.plan.vtx_bufs.end()) {
            im.ensureCtx();
            SendGeometryToBuffer(*im.ctx, it->second, cpu_img,
                                 var.getVType());
            uploaded = true;
        }
        if (auto it = im.plan.imgs.find(var); it != im.plan.imgs.end()) {
            im.ensureCtx();
            SendImageToDevice(*im.ctx, it->second, cpu_img, var.getVType(),
                              /*image_initialized=*/true);
            uploaded = true;
        }
        if (uploaded) {
            return;
        }
    }
    // Images/buffers do not exist yet; copy and flush inside execStep()
    CpuImage owned(cpu_img.width, cpu_img.height, cpu_img.channels);
    std::copy(cpu_img.data, cpu_img.data + cpu_img.numElems(),
              owned.data.begin());
    im.pending_sends[var] = std::move(owned);
}

void DressiAD::sendImg(const Variable& var, const CpuImage& cpu_img) {
    sendImg(var, CpuImageView(cpu_img));
}

void DressiAD::execStepWithSends(
        const std::vector<std::pair<Variable, CpuImageView>>& items) {
    Impl& im = *m_impl;
    im.batch_uploads.clear();
    for (const auto& [var, view] : items) {
        DRESSI_CHECK(var, "execStepWithSends: null Variable");
        Variable v = var;
        v.setDirty(true);
        const bool is_geom =
                im.plan_valid && im.plan.vtx_bufs.count(var) != 0;
        const bool is_img = im.plan_valid && im.plan.imgs.count(var) != 0;
        if (is_geom) {
            im.ensureCtx();
            SendGeometryToBuffer(*im.ctx, im.plan.vtx_bufs.at(var), view,
                                 var.getVType());
        }
        if (is_img) {
            // Deferred: fused into the execute submit (or uploaded normally
            // if a rebuild lands this iteration). The borrowed view stays
            // valid for the duration of this call.
            im.batch_uploads.emplace_back(var, view);
        }
        if (!is_geom && !is_img) {
            CpuImage owned(view.width, view.height, view.channels);
            std::copy(view.data, view.data + view.numElems(),
                      owned.data.begin());
            im.pending_sends[var] = std::move(owned);
        }
    }
    execStep();
    im.batch_uploads.clear();
}

CpuImage DressiAD::execStepWithSendsAndRecvImgsStacked(
        const std::vector<std::pair<Variable, CpuImageView>>& sends,
        const Variables& recvs) {
    Impl& im = *m_impl;
    DRESSI_CHECK(!recvs.empty(),
                 "execStepWithSendsAndRecvImgsStacked: empty recv batch");
    im.batch_downloads = recvs;
    im.batch_download_result.reset();
    execStepWithSends(sends);
    im.batch_downloads.clear();
    DRESSI_CHECK(im.batch_download_result.has_value(),
                 "combined recv did not produce an image");
    CpuImage out = std::move(*im.batch_download_result);
    im.batch_download_result.reset();
    return out;
}

void DressiAD::sendImgs(
        const std::vector<std::pair<Variable, CpuImageView>>& items) {
    Impl& im = *m_impl;
    std::vector<ImageSendItem> batch;
    for (const auto& [var, cpu_img] : items) {
        DRESSI_CHECK(var, "sendImgs: null Variable");
        Variable v = var;
        v.setDirty(true);
        bool uploaded = false;
        if (im.plan_valid) {
            if (auto it = im.plan.vtx_bufs.find(var);
                it != im.plan.vtx_bufs.end()) {
                im.ensureCtx();
                // Host-visible memcpy; no submit involved
                SendGeometryToBuffer(*im.ctx, it->second, cpu_img,
                                     var.getVType());
                uploaded = true;
            }
            if (auto it = im.plan.imgs.find(var); it != im.plan.imgs.end()) {
                batch.push_back({it->second, cpu_img, var.getVType(),
                                 /*initialized=*/true});
                uploaded = true;
            }
        }
        if (!uploaded) {
            CpuImage owned(cpu_img.width, cpu_img.height, cpu_img.channels);
            std::copy(cpu_img.data, cpu_img.data + cpu_img.numElems(),
                      owned.data.begin());
            im.pending_sends[var] = std::move(owned);
        }
    }
    if (!batch.empty()) {
        im.ensureCtx();
        SendImagesToDevice(*im.ctx, batch);
    }
}

CpuImage DressiAD::recvImg(const Variable& var) {
    Impl& im = *m_impl;
    DRESSI_CHECK(var, "recvImg: null Variable");
    if (auto it = im.pending_sends.find(var); it != im.pending_sends.end()) {
        return it->second;  // not yet uploaded; return the pending data
    }
    DRESSI_CHECK(im.plan_valid, "recvImg before the first execStep()");
    auto it = im.plan.imgs.find(var);
    DRESSI_CHECK(it != im.plan.imgs.end(),
                 "recvImg: the Variable has no GPU image");
    return ReceiveImageFromDevice(*im.ctx, it->second, var.getVType());
}

namespace {

std::vector<std::pair<vkw::ImagePackPtr, VType>> CollectRecvItems(
        const GpuPlan& plan, const Variables& vars) {
    std::vector<std::pair<vkw::ImagePackPtr, VType>> items;
    items.reserve(vars.size());
    for (const auto& var : vars) {
        DRESSI_CHECK(var, "recvImgs: null Variable");
        auto it = plan.imgs.find(var);
        DRESSI_CHECK(it != plan.imgs.end(),
                     "recvImgs: the Variable has no GPU image");
        items.emplace_back(it->second, var.getVType());
    }
    return items;
}

}  // namespace

std::vector<CpuImage> DressiAD::recvImgs(const Variables& vars) {
    Impl& im = *m_impl;
    DRESSI_CHECK(im.plan_valid, "recvImgs before the first execStep()");
    return ReceiveImagesFromDevice(*im.ctx,
                                   CollectRecvItems(im.plan, vars));
}

CpuImage DressiAD::recvImgsStacked(const Variables& vars) {
    Impl& im = *m_impl;
    DRESSI_CHECK(im.plan_valid, "recvImgs before the first execStep()");
    return ReceiveImagesStacked(*im.ctx, CollectRecvItems(im.plan, vars));
}

}  // namespace dressi
