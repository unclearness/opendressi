#include "dressi/dressi_ad.h"

#include <map>

#include <algorithm>
#include <chrono>

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

    void ensureCtx() {
        if (!ctx) {
            // DRESSI_VK_DEBUG=1 turns on the validation layers
            const char* dbg = std::getenv("DRESSI_VK_DEBUG");
            ctx = CreateVkContext(dbg != nullptr && dbg[0] == '1');
        }
    }

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
DressiAD::~DressiAD() = default;

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
        for (auto& ss : im.substages) {
            if (ss.shader_type == RASTER) {
                const RasterShaders shaders = GenerateRasterShaders(ss);
                ss.vert_shader_code = shaders.vert;
                ss.shader_code = shaders.frag;
            } else if (ss.shader_type == COMP) {
                ss.shader_code = GenerateCompShader(ss);
            } else {
                ss.shader_code = GenerateFragShader(ss);
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
    timer.mark("upload");

    // 7) Execute the command buffer
    ExecuteGpuPlan(*im.ctx, im.plan);
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

void DressiAD::sendImg(const Variable& var, const CpuImage& cpu_img) {
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
    // Images/buffers do not exist yet; flush inside the next execStep()
    im.pending_sends[var] = cpu_img;
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

}  // namespace dressi
