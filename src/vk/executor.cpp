#include "vk/executor.h"

#include <algorithm>
#include <set>

#include <spdlog/spdlog.h>

#include "codegen/glsl_codegen.h"
#include "core/node.h"
#include "vk/transfer.h"

namespace dressi {

namespace {

vkw::ImagePackPtr CreateVarImage(const VkContext& ctx, const Variable& var) {
    const ImgSize size = var.getImgSize();
    return vkw::CreateImagePack(
            ctx.physical_device, ctx.device, FormatOf(var.getVType()),
            {size.w, size.h}, 1,
            vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eInputAttachment |
                    vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eStorage |
                    vk::ImageUsageFlagBits::eTransferSrc |
                    vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
}

}  // namespace

GpuPlan BuildGpuPlan(const VkContext& ctx, const Stages& stages,
                     const std::map<Variable, Variable>& upd_inp_map,
                     GpuPlan prev_plan) {
    GpuPlan plan;
    // GPU-resident data survives rebuilds: cached images, geometry buffers,
    // and their sampler wrappers
    plan.imgs = std::move(prev_plan.imgs);
    plan.vtx_bufs = std::move(prev_plan.vtx_bufs);
    plan.textures = std::move(prev_plan.textures);
    plan.uif_bufs = std::move(prev_plan.uif_bufs);

    // 1) Create (or reuse) an image per substage I/O variable.
    // Optimizer outputs get their own image; an explicit copy-back into the
    // input's image closes the loop at the end of each iteration (aliasing
    // both to one image would be a render-pass feedback loop).
    const auto ensure_img = [&](const Variable& var) {
        if (!plan.imgs.count(var)) {
            plan.imgs[var] = CreateVarImage(ctx, var);
        }
    };
    // Geometry inputs live in vertex/index buffers. Leaves are host-visible
    // (filled by sendImg); GPU-generated vertex data (a variable with a
    // creator, or a leaf refreshed by the optimizer copy-back) is copied
    // from its image on the GPU each frame -- no CPU round trip. vtx_vars
    // order per Rasterize: {clip_pos, attrib, faces}.
    const auto ensure_vtx_buf = [&](const Variable& var, bool is_index) {
        if (plan.vtx_bufs.count(var)) {
            return;
        }
        const bool gpu_generated = bool(var.getCreator());
        DRESSI_CHECK(!(gpu_generated && is_index),
                     "faces must be a CPU leaf (uint conversion on upload)");
        if (gpu_generated) {
            // The image->buffer copy is texel-exact, so padded types
            // (VEC3 -> RGBA32F) cannot feed a tightly packed buffer
            DRESSI_CHECK(PhysChannels(var.getVType()) ==
                                 NumComponents(var.getVType()),
                         "GPU-generated geometry must be FLOAT/VEC2/VEC4");
        }
        const ImgSize size = var.getImgSize();
        const size_t n_elems =
                size_t(size.w) * size.h * NumComponents(var.getVType());
        plan.vtx_bufs[var] = vkw::CreateBufferPack(
                ctx.physical_device, ctx.device, n_elems * 4,
                (is_index ? vk::BufferUsageFlagBits::eIndexBuffer
                          : vk::BufferUsageFlagBits::eVertexBuffer) |
                        vk::BufferUsageFlagBits::eTransferDst,
                gpu_generated ? vk::MemoryPropertyFlagBits::eDeviceLocal
                              : vkw::HOST_VISIB_COHER_PROPS);
    };
    const auto contains = [](const Variables& vars, const Variable& v) {
        return std::find(vars.begin(), vars.end(), v) != vars.end();
    };

    // Zero-copy optimizer aliasing (the paper's model: input and updated
    // share one VkImage). Legal when the writer substage consumes the input
    // same-pixel only: the attachment is bound as both input and color in
    // eGeneral layout with a subpass self-dependency -- the fullscreen pass
    // has one fragment per pixel, so no fragment reads another's write.
    // Pairs that do not satisfy the conditions keep the end-of-frame
    // copy-back.
    std::map<Variable, Variable> aliased;    // updated -> input
    std::map<Variable, Variable> alias_rev;  // input -> updated
    for (const auto& [upd, inp] : upd_inp_map) {
        int w_stage = -1;
        const SubStage* w_ss = nullptr;
        for (size_t si = 0; si < stages.size() && w_stage < 0; si++) {
            for (const auto& ss : stages[si].substages) {
                if (contains(ss.out_vars, upd)) {
                    w_stage = int(si);
                    w_ss = &ss;
                    break;
                }
            }
        }
        if (w_stage < 0 || stages[size_t(w_stage)].shader_type != FRAG) {
            continue;  // pruned or raster-written: keep the copy-back
        }
        // The writer may read the input as an input attachment (same-pixel)
        // or via its frame-start UBO copy (uif) -- not as a texture
        bool ok = !contains(w_ss->slt_vars, inp) &&
                  !contains(w_ss->tex_vars, inp) &&
                  !contains(w_ss->vtx_vars, inp);
        // No other substage of the writer stage may touch input or updated,
        // and no later stage may read the input (it would observe the new
        // value one frame early)
        for (size_t si = size_t(w_stage); ok && si < stages.size(); si++) {
            for (const auto& ss : stages[si].substages) {
                if (&ss == w_ss) {
                    continue;
                }
                const bool same_stage = si == size_t(w_stage);
                for (const Variables* vars :
                     {&ss.inp_vars, &ss.slt_vars, &ss.tex_vars, &ss.vtx_vars,
                      &ss.uif_vars}) {
                    if (contains(*vars, inp) ||
                        (same_stage && contains(*vars, upd))) {
                        ok = false;
                    }
                }
            }
        }
        if (!ok) {
            continue;
        }
        ensure_img(inp);
        plan.imgs[upd] = plan.imgs.at(inp);
        aliased[upd] = inp;
        alias_rev[inp] = upd;
    }

    // {1,1} uniform inputs: a vec4 UBO per variable (plus the image that
    // serves as the copy source and the sendImg target)
    const auto ensure_uif_buf = [&](const Variable& var) {
        if (!plan.uif_bufs.count(var)) {
            plan.uif_bufs[var] = vkw::CreateBufferPack(
                    ctx.physical_device, ctx.device, 16,
                    vk::BufferUsageFlagBits::eUniformBuffer |
                            vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    };
    for (const auto& stage : stages) {
        for (const auto& ss : stage.substages) {
            for (const auto& v : ss.slt_vars) {
                ensure_img(v);
            }
            for (const auto& v : ss.tex_vars) {
                ensure_img(v);
            }
            for (const auto& v : ss.inp_vars) {
                ensure_img(v);
            }
            for (const auto& v : ss.out_vars) {
                ensure_img(v);
            }
            for (const auto& v : ss.uif_vars) {
                ensure_img(v);
                ensure_uif_buf(v);
            }
            if (ss.shader_type == RASTER) {
                DRESSI_CHECK(ss.vtx_vars.size() == 3,
                             "RASTER substage expects {pos, attrib, faces}");
                ensure_vtx_buf(ss.vtx_vars[0], false);
                ensure_vtx_buf(ss.vtx_vars[1], false);
                ensure_vtx_buf(ss.vtx_vars[2], true);
            }
        }
    }

    // Nearest-sampler textures for texelFetch reads
    const auto ensure_texture = [&](const Variable& var) {
        if (plan.textures.count(var)) {
            return;
        }
        plan.textures[var] = vkw::CreateTexturePack(
                plan.imgs.at(var), ctx.device, vk::Filter::eNearest,
                vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
                vk::SamplerAddressMode::eClampToEdge,
                vk::SamplerAddressMode::eClampToEdge,
                vk::SamplerAddressMode::eClampToEdge);
    };

    // 2) Shaders, render passes, pipelines, descriptor sets
    auto vert_module = ctx.glsl_compiler->compileFromString(
            ctx.device, FullscreenVertShader(),
            vk::ShaderStageFlagBits::eVertex);

    plan.cmd_pack = vkw::CreateCommandBuffersPack(ctx.device,
                                                  ctx.queue_family_idx, 1);
    const auto& cmd = plan.cmd_pack->cmd_bufs[0];
    vkw::BeginCommand(cmd, /*one_time_submit=*/false);

    // Per-stage GPU timestamps (debug logging only). Interval k =
    // ts[k+1]-ts[k] and is described by ts_labels[k].
    const bool ts_enabled = spdlog::should_log(spdlog::level::debug) &&
                            ctx.limits.timestampComputeAndGraphics;
    uint32_t ts_next = 0;
    const auto ts_mark = [&](std::string label) {
        if (!ts_enabled) {
            return;
        }
        cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                            *plan.ts_pool, ts_next++);
        if (!label.empty()) {
            plan.ts_labels.push_back(std::move(label));
        }
    };
    if (ts_enabled) {
        uint32_t n_query = uint32_t(stages.size()) + 3;
        plan.ts_pool = ctx.device->createQueryPoolUnique(
                {{}, vk::QueryType::eTimestamp, n_query});
        plan.ts_period_ns = ctx.limits.timestampPeriod;
        cmd->resetQueryPool(*plan.ts_pool, 0, n_query);
        ts_mark("");  // origin
    }

    // Copies a variable's image into its vertex buffer on the GPU and makes
    // the write visible to vertex fetch (GPU-resident geometry updates)
    const auto copy_img_to_vtx_buf = [&](const Variable& v) {
        vkw::CopyImageToBuffer(cmd, plan.imgs.at(v), plan.vtx_bufs.at(v),
                               vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
        const vk::BufferMemoryBarrier barrier(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eVertexAttributeRead,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                plan.vtx_bufs.at(v)->buf.get(), 0, VK_WHOLE_SIZE);
        cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                             vk::PipelineStageFlagBits::eVertexInput, {},
                             nullptr, barrier, nullptr);
    };

    // Optimizer-updated geometry leaves: refresh the vertex buffer from the
    // leaf's image at frame start (the copy-back wrote it last frame; on
    // the first frame the image holds the same data sendImg uploaded)
    for (const auto& [upd, inp] : upd_inp_map) {
        (void)upd;
        if (plan.vtx_bufs.count(inp) && plan.imgs.count(inp)) {
            copy_img_to_vtx_buf(inp);
        }
    }

    // {1,1} uniform refresh: copy each uif image into its UBO right after
    // the stage that produces it (leaves and pruned cached values at frame
    // start), then make the write visible to uniform reads
    const auto copy_imgs_to_uif_bufs = [&](const Variables& vars) {
        if (vars.empty()) {
            return;
        }
        for (const auto& v : vars) {
            vkw::CopyImageToBuffer(cmd, plan.imgs.at(v), plan.uif_bufs.at(v),
                                   vk::ImageLayout::eShaderReadOnlyOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal);
        }
        // One barrier covers every copied UBO
        const vk::MemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite,
                                        vk::AccessFlagBits::eUniformRead);
        cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                             vk::PipelineStageFlagBits::eFragmentShader, {},
                             barrier, nullptr, nullptr);
    };
    // uif vars grouped by the stage that produces them (leaves and pruned
    // cached values are ready now and copied at frame start)
    std::vector<Variables> uif_after_stage(stages.size());
    {
        std::map<Variable, size_t> producer;
        std::set<Variable> uif_all;
        for (size_t si = 0; si < stages.size(); si++) {
            for (const auto& ss : stages[si].substages) {
                for (const auto& v : ss.out_vars) {
                    producer.emplace(v, si);
                }
                for (const auto& v : ss.uif_vars) {
                    uif_all.insert(v);
                }
            }
        }
        Variables uif_front;
        for (const auto& v : uif_all) {
            if (auto it = producer.find(v); it != producer.end()) {
                uif_after_stage[it->second].push_back(v);
            } else {
                uif_front.push_back(v);
            }
        }
        copy_imgs_to_uif_bufs(uif_front);
    }
    ts_mark("front-copies");

    // Short description of a stage for the timing report
    const auto stage_label = [](const Stage& stage) {
        std::string funcs;
        size_t listed = 0;
        for (const auto& ss : stage.substages) {
            for (const auto& f : ss.funcs) {
                if (listed == 4) {
                    funcs += ",..";
                    break;
                }
                funcs += (listed ? "," : "") + NodeAccess::Node(f)->name;
                listed++;
            }
            if (listed > 4) {
                break;
            }
        }
        size_t n_inp = 0, n_slt = 0, n_out = 0, n_fn = 0;
        for (const auto& ss : stage.substages) {
            n_inp += ss.inp_vars.size();
            n_slt += ss.slt_vars.size();
            n_out += ss.out_vars.size();
            n_fn += ss.funcs.size();
        }
        return fmt::format("{} {}x{} ss={} fn={} i={} s={} o={} [{}]",
                           stage.shader_type == RASTER ? "R"
                           : stage.shader_type == COMP ? "C"
                                                       : "F",
                           stage.img_size.w, stage.img_size.h,
                           stage.substages.size(), n_fn, n_inp, n_slt, n_out,
                           funcs);
    };

    // Records one RASTER stage: indexed depth-tested draw into a cleared
    // attribute attachment (deferred-shading G-buffer channel)
    const auto record_raster_stage = [&](const Stage& stage) {
        const SubStage& ss = stage.substages[0];
        const Variable& pos_var = ss.vtx_vars[0];
        const Variable& attr_var = ss.vtx_vars[1];
        const Variable& faces_var = ss.vtx_vars[2];
        const uint32_t n_outs = uint32_t(ss.out_vars.size());
        DRESSI_CHECK(n_outs >= 1, "Raster substage must have outputs");

        auto rp = vkw::CreateRenderPassPack();
        std::vector<vkw::AttachmentRefInfo> col_refs;
        for (uint32_t i = 0; i < n_outs; i++) {
            vkw::AddAttachientDesc(
                    rp, FormatOf(ss.out_vars[i].getVType()),
                    vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::AttachmentLoadOp::eClear,
                    vk::AttachmentStoreOp::eStore);
            col_refs.emplace_back(i, vk::ImageLayout::eColorAttachmentOptimal);
        }
        const auto depth_format = vk::Format::eD32Sfloat;
        vkw::AddAttachientDesc(
                rp, depth_format, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                vk::AttachmentLoadOp::eClear,
                vk::AttachmentStoreOp::eDontCare);
        vkw::AddSubpassDesc(
                rp, {}, col_refs,
                {n_outs, vk::ImageLayout::eDepthStencilAttachmentOptimal});
        vkw::UpdateRenderPass(ctx.device, rp);
        plan.render_passes.push_back(rp);

        auto depth_img = vkw::CreateImagePack(
                ctx.physical_device, ctx.device, depth_format,
                {stage.img_size.w, stage.img_size.h}, 1,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, true,
                vk::ImageAspectFlagBits::eDepth);
        plan.depth_imgs.push_back(depth_img);

        std::vector<vkw::ImagePackPtr> fb_imgs;
        for (const auto& v : ss.out_vars) {
            fb_imgs.push_back(plan.imgs.at(v));
        }
        fb_imgs.push_back(depth_img);
        auto fb = vkw::CreateFrameBuffer(ctx.device, rp, fb_imgs);
        plan.frame_buffers.push_back(fb);

        const RasterShaders shaders = GenerateRasterShaders(ss);
        auto vert_module = ctx.glsl_compiler->compileFromString(
                ctx.device, shaders.vert, vk::ShaderStageFlagBits::eVertex);
        auto frag_module = ctx.glsl_compiler->compileFromString(
                ctx.device, shaders.frag, vk::ShaderStageFlagBits::eFragment);

        // texelFetch/uniform inputs of the raster fragment shader (binding
        // order shared with the codegen: slt, then uif)
        vkw::DescSetPackPtr desc_set;
        if (!ss.slt_vars.empty() || !ss.uif_vars.empty()) {
            std::vector<vkw::DescSetInfo> binding_infos(
                    ss.slt_vars.size(),
                    {vk::DescriptorType::eCombinedImageSampler, 1,
                     vk::ShaderStageFlagBits::eFragment});
            for (size_t i = 0; i < ss.uif_vars.size(); i++) {
                binding_infos.emplace_back(
                        vk::DescriptorType::eUniformBuffer, 1,
                        vk::ShaderStageFlagBits::eFragment);
            }
            desc_set = vkw::CreateDescriptorSetPack(ctx.device, binding_infos);
            auto write_pack = vkw::CreateWriteDescSetPack();
            for (size_t i = 0; i < ss.slt_vars.size(); i++) {
                ensure_texture(ss.slt_vars[i]);
                vkw::AddWriteDescSet(
                        write_pack, desc_set, uint32_t(i),
                        {plan.textures.at(ss.slt_vars[i])},
                        {vk::ImageLayout::eShaderReadOnlyOptimal});
            }
            uint32_t binding = uint32_t(ss.slt_vars.size());
            for (const auto& v : ss.uif_vars) {
                vkw::AddWriteDescSet(write_pack, desc_set, binding++,
                                     {plan.uif_bufs.at(v)});
            }
            vkw::UpdateDescriptorSets(ctx.device, write_pack);
            plan.desc_sets.push_back(desc_set);
        }

        const uint32_t attr_comps = NumComponents(attr_var.getVType());
        const vk::Format attr_fmt =
                attr_comps == 1 ? vk::Format::eR32Sfloat
                : attr_comps == 2 ? vk::Format::eR32G32Sfloat
                : attr_comps == 3 ? vk::Format::eR32G32B32Sfloat
                                  : vk::Format::eR32G32B32A32Sfloat;
        vkw::PipelineInfo pipeline_info;
        pipeline_info.face_culling = vk::CullModeFlagBits::eNone;
        pipeline_info.depth_test_enable = true;
        pipeline_info.depth_write_enable = true;
        pipeline_info.depth_comp_op = vk::CompareOp::eLessOrEqual;
        pipeline_info.color_blend_infos.resize(n_outs);
        std::vector<vkw::DescSetPackPtr> desc_packs;
        if (desc_set) {
            desc_packs.push_back(desc_set);
        }
        auto pipeline = vkw::CreateGraphicsPipeline(
                ctx.device, {vert_module, frag_module},
                {{0, 16, vk::VertexInputRate::eVertex},
                 {1, attr_comps * 4, vk::VertexInputRate::eVertex}},
                {{0, 0, vk::Format::eR32G32B32A32Sfloat, 0},
                 {1, 1, attr_fmt, 0}},
                pipeline_info, desc_packs, rp, 0);
        plan.pipelines.push_back(pipeline);

        // GPU-generated vertex data: refresh the buffers from the producer
        // stages' images (recorded earlier in this command buffer)
        if (pos_var.getCreator()) {
            copy_img_to_vtx_buf(pos_var);
        }
        if (attr_var.getCreator()) {
            copy_img_to_vtx_buf(attr_var);
        }

        std::vector<vk::ClearValue> clear_vals(
                n_outs, vk::ClearValue(vk::ClearColorValue(
                                std::array<float, 4>{0.f, 0.f, 0.f, 0.f})));
        clear_vals.emplace_back(vk::ClearDepthStencilValue(1.f, 0));
        vkw::CmdBeginRenderPass(cmd, rp, fb, clear_vals);
        vkw::CmdBindPipeline(cmd, pipeline);
        if (desc_set) {
            vkw::CmdBindDescSets(cmd, pipeline, {desc_set});
        }
        const vk::Extent2D extent = {stage.img_size.w, stage.img_size.h};
        vkw::CmdSetViewport(cmd, extent);
        vkw::CmdSetScissor(cmd, extent);
        vkw::CmdBindVertexBuffers(cmd, 0,
                                  {plan.vtx_bufs.at(pos_var),
                                   plan.vtx_bufs.at(attr_var)});
        vkw::CmdBindIndexBuffer(cmd, plan.vtx_bufs.at(faces_var), 0,
                                vk::IndexType::eUint32);
        const uint32_t n_indices =
                faces_var.getImgSize().w * faces_var.getImgSize().h * 3;
        vkw::CmdDrawIndexed(cmd, n_indices);
        vkw::CmdEndRenderPass(cmd);

        for (const auto& v : ss.out_vars) {
            vkw::BarrierImage(
                    cmd, plan.imgs.at(v),
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
                    vk::AccessFlagBits::eColorAttachmentWrite,
                    vk::PipelineStageFlagBits::eFragmentShader,
                    vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eInputAttachmentRead);
        }
    };

    // Copies for uif values produced by stage `si` (recorded after it)
    const auto record_uif_copies = [&](size_t si) {
        copy_imgs_to_uif_bufs(uif_after_stage[si]);
    };

    // Records one COMP stage: a single dispatch (one invocation per pixel,
    // 64-wide groups along the row). Inputs are samplers (inp = same-pixel
    // texelFetch, slt = arbitrary texelFetch), outputs storage images in
    // eGeneral during the dispatch, transitioned back for sampled reads.
    const auto record_comp_stage = [&](const Stage& stage) {
        DRESSI_CHECK(stage.substages.size() == 1,
                     "COMP stages hold one substage");
        const SubStage& ss = stage.substages[0];
        auto comp_module = ctx.glsl_compiler->compileFromString(
                ctx.device, ss.shader_code, vk::ShaderStageFlagBits::eCompute);

        // Binding convention shared with the codegen: inp, slt, uif, out
        std::vector<vkw::DescSetInfo> binding_infos;
        const size_t n_samplers = ss.inp_vars.size() + ss.slt_vars.size();
        for (size_t i = 0; i < n_samplers; i++) {
            binding_infos.emplace_back(
                    vk::DescriptorType::eCombinedImageSampler, 1,
                    vk::ShaderStageFlagBits::eCompute);
        }
        for (size_t i = 0; i < ss.uif_vars.size(); i++) {
            binding_infos.emplace_back(vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eCompute);
        }
        for (size_t i = 0; i < ss.out_vars.size(); i++) {
            binding_infos.emplace_back(vk::DescriptorType::eStorageImage, 1,
                                       vk::ShaderStageFlagBits::eCompute);
        }
        auto desc_set = vkw::CreateDescriptorSetPack(ctx.device,
                                                     binding_infos);
        auto write_pack = vkw::CreateWriteDescSetPack();
        uint32_t binding = 0;
        for (const Variables* sampled : {&ss.inp_vars, &ss.slt_vars}) {
            for (const auto& v : *sampled) {
                ensure_texture(v);
                vkw::AddWriteDescSet(
                        write_pack, desc_set, binding++,
                        {plan.textures.at(v)},
                        {vk::ImageLayout::eShaderReadOnlyOptimal});
            }
        }
        for (const auto& v : ss.uif_vars) {
            vkw::AddWriteDescSet(write_pack, desc_set, binding++,
                                 {plan.uif_bufs.at(v)});
        }
        for (const auto& v : ss.out_vars) {
            vkw::AddWriteDescSet(write_pack, desc_set, binding++,
                                 {plan.imgs.at(v)},
                                 {vk::ImageLayout::eGeneral});
        }
        vkw::UpdateDescriptorSets(ctx.device, write_pack);
        plan.desc_sets.push_back(desc_set);

        auto pipeline = vkw::CreateComputePipeline(ctx.device, comp_module,
                                                   {desc_set});
        plan.pipelines.push_back(pipeline);

        // Outputs -> eGeneral for storage writes (one batched barrier).
        // eUndefined source keeps the recorded transition valid on the
        // first execution too; the dispatch overwrites every pixel.
        const auto out_barriers = [&](vk::ImageLayout from, vk::ImageLayout to,
                                      vk::PipelineStageFlags src_stage,
                                      vk::AccessFlags src_access,
                                      vk::PipelineStageFlags dst_stage,
                                      vk::AccessFlags dst_access) {
            std::vector<vk::ImageMemoryBarrier> barriers;
            for (const auto& v : ss.out_vars) {
                const auto& img = plan.imgs.at(v);
                const vk::ImageSubresourceRange range(
                        img->view_aspects, img->view_miplevel_base,
                        img->view_miplevel_cnt, 0, 1);
                barriers.emplace_back(src_access, dst_access, from, to,
                                      VK_QUEUE_FAMILY_IGNORED,
                                      VK_QUEUE_FAMILY_IGNORED,
                                      img->img_res_pack->img.get(), range);
            }
            cmd->pipelineBarrier(src_stage, dst_stage, {}, nullptr, nullptr,
                                 barriers);
        };
        out_barriers(vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTopOfPipe, {},
                     vk::PipelineStageFlagBits::eComputeShader,
                     vk::AccessFlagBits::eShaderWrite);
        vkw::CmdBindPipeline(cmd, pipeline, vk::PipelineBindPoint::eCompute);
        vkw::CmdBindDescSets(cmd, pipeline, {desc_set}, {},
                             vk::PipelineBindPoint::eCompute);
        vkw::CmdDispatch(cmd, (stage.img_size.w + 63) / 64,
                         stage.img_size.h);
        out_barriers(vk::ImageLayout::eGeneral,
                     vk::ImageLayout::eShaderReadOnlyOptimal,
                     vk::PipelineStageFlagBits::eComputeShader,
                     vk::AccessFlagBits::eShaderWrite,
                     vk::PipelineStageFlagBits::eFragmentShader |
                             vk::PipelineStageFlagBits::eComputeShader |
                             vk::PipelineStageFlagBits::eTransfer,
                     vk::AccessFlagBits::eShaderRead |
                             vk::AccessFlagBits::eTransferRead);
    };

    for (size_t stage_idx = 0; stage_idx < stages.size(); stage_idx++) {
        const Stage& stage = stages[stage_idx];
        if (stage.shader_type == RASTER) {
            record_raster_stage(stage);
            record_uif_copies(stage_idx);
            ts_mark(stage_label(stage));
            continue;
        }
        if (stage.shader_type == COMP) {
            record_comp_stage(stage);
            record_uif_copies(stage_idx);
            ts_mark(stage_label(stage));
            continue;
        }
        DRESSI_CHECK(stage.shader_type == FRAG,
                     "unknown stage shader type");

        // Aliased updates written by this stage (input+color, one image)
        std::map<Variable, Variable> stage_alias;  // updated -> input
        for (const auto& ss : stage.substages) {
            for (const auto& v : ss.out_vars) {
                if (auto it = aliased.find(v); it != aliased.end()) {
                    stage_alias.emplace(v, it->second);
                }
            }
        }
        if (!stage_alias.empty()) {
            // WAR: every earlier read of the shared image (sampled reads,
            // vertex-buffer refresh copies) must finish before this pass
            // writes it / transitions it to eGeneral
            const vk::MemoryBarrier war(
                    vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eTransferRead,
                    vk::AccessFlagBits::eColorAttachmentWrite);
            cmd->pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader |
                                         vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::
                                         eColorAttachmentOutput,
                                 {}, war, nullptr, nullptr);
        }

        auto rp = vkw::CreateRenderPassPack();

        // Attachments: all substage outputs (written) plus external inputs
        // consumed as input attachments (loaded), stage-locally indexed
        std::map<Variable, uint32_t> attach_idx;
        std::map<Variable, size_t> writer_subpass;
        std::vector<vkw::ImagePackPtr> attach_imgs;
        for (size_t ss_idx = 0; ss_idx < stage.substages.size(); ss_idx++) {
            for (const auto& v : stage.substages[ss_idx].out_vars) {
                writer_subpass[v] = ss_idx;
                if (attach_idx.count(v)) {
                    continue;
                }
                attach_idx[v] = uint32_t(attach_imgs.size());
                attach_imgs.push_back(plan.imgs.at(v));
                if (auto al = stage_alias.find(v); al != stage_alias.end()) {
                    // The input reads this same attachment (subpassLoad),
                    // so the old content must be loaded, not discarded
                    attach_idx[al->second] = attach_idx[v];
                    vkw::AddAttachientDesc(
                            rp, FormatOf(v.getVType()),
                            vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::AttachmentLoadOp::eLoad,
                            vk::AttachmentStoreOp::eStore);
                } else {
                    vkw::AddAttachientDesc(
                            rp, FormatOf(v.getVType()),
                            vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::AttachmentLoadOp::eDontCare,
                            vk::AttachmentStoreOp::eStore);
                }
            }
        }
        for (const auto& ss : stage.substages) {
            for (const auto& v : ss.inp_vars) {
                if (attach_idx.count(v)) {
                    continue;  // written in this stage
                }
                attach_idx[v] = uint32_t(attach_imgs.size());
                attach_imgs.push_back(plan.imgs.at(v));
                vkw::AddAttachientDesc(
                        rp, FormatOf(v.getVType()),
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::AttachmentLoadOp::eLoad,
                        vk::AttachmentStoreOp::eStore);
            }
        }

        for (size_t ss_idx = 0; ss_idx < stage.substages.size(); ss_idx++) {
            const SubStage& ss = stage.substages[ss_idx];
            // Inputs whose updated value this subpass writes into the same
            // attachment must be referenced in eGeneral (input+color)
            const auto aliased_here = [&](const Variable& v) {
                const auto it = alias_rev.find(v);
                return it != alias_rev.end() &&
                       contains(ss.out_vars, it->second);
            };
            std::vector<vkw::AttachmentRefInfo> inp_refs;
            for (const auto& v : ss.inp_vars) {
                inp_refs.emplace_back(
                        attach_idx.at(v),
                        aliased_here(v)
                                ? vk::ImageLayout::eGeneral
                                : vk::ImageLayout::eShaderReadOnlyOptimal);
                // Chain producer subpass -> this subpass
                if (auto it = writer_subpass.find(v);
                    it != writer_subpass.end() && it->second != ss_idx) {
                    vkw::AddSubpassDepend(
                            rp,
                            {uint32_t(it->second),
                             vk::PipelineStageFlagBits::eColorAttachmentOutput,
                             vk::AccessFlagBits::eColorAttachmentWrite},
                            {uint32_t(ss_idx),
                             vk::PipelineStageFlagBits::eFragmentShader,
                             vk::AccessFlagBits::eInputAttachmentRead},
                            vk::DependencyFlagBits::eByRegion);
                }
            }
            std::vector<vkw::AttachmentRefInfo> col_refs;
            bool has_alias_out = false;
            for (const auto& v : ss.out_vars) {
                const bool al = stage_alias.count(v) != 0;
                has_alias_out |= al;
                col_refs.emplace_back(
                        attach_idx.at(v),
                        al ? vk::ImageLayout::eGeneral
                           : vk::ImageLayout::eColorAttachmentOptimal);
            }
            if (has_alias_out) {
                // Self-dependency for the input+color attachment (each
                // fragment only touches its own pixel)
                vkw::AddSubpassDepend(
                        rp,
                        {uint32_t(ss_idx),
                         vk::PipelineStageFlagBits::eColorAttachmentOutput,
                         vk::AccessFlagBits::eColorAttachmentWrite},
                        {uint32_t(ss_idx),
                         vk::PipelineStageFlagBits::eFragmentShader,
                         vk::AccessFlagBits::eInputAttachmentRead},
                        vk::DependencyFlagBits::eByRegion);
            }
            vkw::AddSubpassDesc(rp, inp_refs, col_refs);
        }
        vkw::UpdateRenderPass(ctx.device, rp);
        plan.render_passes.push_back(rp);

        auto fb = vkw::CreateFrameBuffer(ctx.device, rp, attach_imgs);
        plan.frame_buffers.push_back(fb);

        const std::vector<vk::ClearValue> clear_vals(
                attach_imgs.size(),
                vk::ClearValue(vk::ClearColorValue(
                        std::array<float, 4>{0.f, 0.f, 0.f, 0.f})));
        vkw::CmdBeginRenderPass(cmd, rp, fb, clear_vals);

        for (size_t ss_idx = 0; ss_idx < stage.substages.size(); ss_idx++) {
            const SubStage& ss = stage.substages[ss_idx];
            if (ss_idx != 0) {
                vkw::CmdNextSubPass(cmd);
            }

            auto frag_module = ctx.glsl_compiler->compileFromString(
                    ctx.device, ss.shader_code,
                    vk::ShaderStageFlagBits::eFragment);

            // Binding convention shared with the codegen: inp, tex, slt,
            // then uif (one vec4 UBO per uniform input)
            vkw::DescSetPackPtr desc_set;
            if (!ss.inp_vars.empty() || !ss.tex_vars.empty() ||
                !ss.slt_vars.empty() || !ss.uif_vars.empty()) {
                std::vector<vkw::DescSetInfo> binding_infos;
                for (size_t i = 0; i < ss.inp_vars.size(); i++) {
                    binding_infos.emplace_back(
                            vk::DescriptorType::eInputAttachment, 1,
                            vk::ShaderStageFlagBits::eFragment);
                }
                const size_t n_samplers =
                        ss.tex_vars.size() + ss.slt_vars.size();
                for (size_t i = 0; i < n_samplers; i++) {
                    binding_infos.emplace_back(
                            vk::DescriptorType::eCombinedImageSampler, 1,
                            vk::ShaderStageFlagBits::eFragment);
                }
                for (size_t i = 0; i < ss.uif_vars.size(); i++) {
                    binding_infos.emplace_back(
                            vk::DescriptorType::eUniformBuffer, 1,
                            vk::ShaderStageFlagBits::eFragment);
                }
                desc_set = vkw::CreateDescriptorSetPack(ctx.device,
                                                        binding_infos);
                auto write_pack = vkw::CreateWriteDescSetPack();
                for (size_t i = 0; i < ss.inp_vars.size(); i++) {
                    const Variable& v = ss.inp_vars[i];
                    // Aliased input+color attachments are in eGeneral
                    const auto al = alias_rev.find(v);
                    const bool general =
                            al != alias_rev.end() &&
                            contains(ss.out_vars, al->second);
                    vkw::AddWriteDescSet(
                            write_pack, desc_set, uint32_t(i),
                            {plan.imgs.at(v)},
                            {general ? vk::ImageLayout::eGeneral
                                     : vk::ImageLayout::
                                               eShaderReadOnlyOptimal});
                }
                uint32_t binding = uint32_t(ss.inp_vars.size());
                for (const Variables* sampled :
                     {&ss.tex_vars, &ss.slt_vars}) {
                    for (const auto& v : *sampled) {
                        ensure_texture(v);
                        vkw::AddWriteDescSet(
                                write_pack, desc_set, binding++,
                                {plan.textures.at(v)},
                                {vk::ImageLayout::eShaderReadOnlyOptimal});
                    }
                }
                for (const auto& v : ss.uif_vars) {
                    vkw::AddWriteDescSet(write_pack, desc_set, binding++,
                                         {plan.uif_bufs.at(v)});
                }
                vkw::UpdateDescriptorSets(ctx.device, write_pack);
                plan.desc_sets.push_back(desc_set);
            }

            vkw::PipelineInfo pipeline_info;
            pipeline_info.face_culling = vk::CullModeFlagBits::eNone;
            pipeline_info.depth_test_enable = false;
            pipeline_info.depth_write_enable = false;
            pipeline_info.color_blend_infos.resize(ss.out_vars.size());

            std::vector<vkw::DescSetPackPtr> desc_packs;
            if (desc_set) {
                desc_packs.push_back(desc_set);
            }
            auto pipeline = vkw::CreateGraphicsPipeline(
                    ctx.device, {vert_module, frag_module}, {}, {},
                    pipeline_info, desc_packs, rp, uint32_t(ss_idx));
            plan.pipelines.push_back(pipeline);

            vkw::CmdBindPipeline(cmd, pipeline);
            if (desc_set) {
                vkw::CmdBindDescSets(cmd, pipeline, {desc_set});
            }
            const vk::Extent2D extent = {stage.img_size.w, stage.img_size.h};
            vkw::CmdSetViewport(cmd, extent);
            vkw::CmdSetScissor(cmd, extent);
            vkw::CmdDraw(cmd, 3);
        }
        vkw::CmdEndRenderPass(cmd);

        // Make outputs visible to later sampled and input-attachment reads
        // (aliased updates also feed next frame's transfer reads: the
        // vertex-buffer and UBO refresh copies)
        for (const auto& [v, ss_idx] : writer_subpass) {
            (void)ss_idx;
            const bool al = stage_alias.count(v) != 0;
            vkw::BarrierImage(
                    cmd, plan.imgs.at(v),
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
                    vk::AccessFlagBits::eColorAttachmentWrite,
                    vk::PipelineStageFlagBits::eFragmentShader |
                            (al ? vk::PipelineStageFlagBits::eTransfer
                                : vk::PipelineStageFlagBits(0)),
                    vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eInputAttachmentRead |
                            (al ? vk::AccessFlagBits::eTransferRead
                                : vk::AccessFlagBits(0)));
        }
        record_uif_copies(stage_idx);
        ts_mark(stage_label(stage));
    }

    // Copy optimizer outputs back into their input images so the next
    // iteration reads the updated values (skipped for zero-copy aliased
    // pairs, which already wrote the input's image directly)
    for (const auto& [upd, inp] : upd_inp_map) {
        if (aliased.count(upd)) {
            continue;
        }
        const auto upd_it = plan.imgs.find(upd);
        const auto inp_it = plan.imgs.find(inp);
        if (upd_it == plan.imgs.end() || inp_it == plan.imgs.end()) {
            continue;
        }
        ensure_img(inp);
        vkw::CopyImage(cmd, upd_it->second, inp_it->second,
                       vk::ImageLayout::eShaderReadOnlyOptimal,
                       vk::ImageLayout::eShaderReadOnlyOptimal,
                       vk::ImageLayout::eShaderReadOnlyOptimal,
                       vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    ts_mark("copy-backs");
    if (ts_enabled) {
        plan.ts_accum_us.assign(plan.ts_labels.size(), 0.0);
    }
    vkw::EndCommand(cmd);

    plan.fence = vkw::CreateFence(ctx.device);
    return plan;
}

void ExecuteGpuPlan(const VkContext& ctx, GpuPlan& plan) {
    DRESSI_CHECK(plan.cmd_pack, "GPU plan has no recorded commands");
    vkw::ResetFence(ctx.device, plan.fence);
    vkw::QueueSubmit(ctx.queue, plan.cmd_pack->cmd_bufs[0], plan.fence);
    const auto res = vkw::WaitForFence(ctx.device, plan.fence);
    DRESSI_CHECK(res == vk::Result::eSuccess, "GPU execution failed");

    if (plan.ts_pool && !plan.ts_labels.empty()) {
        const uint32_t n = uint32_t(plan.ts_labels.size()) + 1;
        std::vector<uint64_t> ticks(n);
        const auto qres = ctx.device->getQueryPoolResults(
                *plan.ts_pool, 0, n, n * sizeof(uint64_t), ticks.data(),
                sizeof(uint64_t),
                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (qres == vk::Result::eSuccess) {
            for (size_t i = 0; i + 1 < ticks.size(); i++) {
                plan.ts_accum_us[i] += double(ticks[i + 1] - ticks[i]) *
                                       plan.ts_period_ns * 1e-3;
            }
            plan.ts_frames++;
            constexpr uint32_t kReportEvery = 100;
            if (plan.ts_frames == kReportEvery) {
                double total = 0.0;
                for (const double us : plan.ts_accum_us) {
                    total += us;
                }
                spdlog::debug("[dressi] GPU stage timing avg over {} frames "
                              "(total {:.1f} us):",
                              plan.ts_frames, total / plan.ts_frames);
                for (size_t i = 0; i < plan.ts_labels.size(); i++) {
                    spdlog::debug("[dressi]   {:7.1f} us  {}",
                                  plan.ts_accum_us[i] / plan.ts_frames,
                                  plan.ts_labels[i]);
                }
                plan.ts_accum_us.assign(plan.ts_labels.size(), 0.0);
                plan.ts_frames = 0;
            }
        }
    }
}

}  // namespace dressi
