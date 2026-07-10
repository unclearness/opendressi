#include "vk/executor.h"

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
                    vk::ImageUsageFlagBits::eTransferSrc |
                    vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
}

}  // namespace

GpuPlan BuildGpuPlan(const VkContext& ctx, const Stages& stages,
                     const std::map<Variable, Variable>& upd_inp_map,
                     std::map<Variable, vkw::ImagePackPtr> prev_imgs) {
    GpuPlan plan;
    plan.imgs = std::move(prev_imgs);

    // 1) Create (or reuse) an image per substage I/O variable.
    // Optimizer outputs get their own image; an explicit copy-back into the
    // input's image closes the loop at the end of each iteration (aliasing
    // both to one image would be a render-pass feedback loop).
    const auto ensure_img = [&](const Variable& var) {
        if (!plan.imgs.count(var)) {
            plan.imgs[var] = CreateVarImage(ctx, var);
        }
    };
    // Geometry inputs live in vertex/index buffers (host-visible; filled by
    // sendImg). vtx_vars order per Rasterize: {clip_pos, attrib, faces}.
    const auto ensure_vtx_buf = [&](const Variable& var, bool is_index) {
        if (plan.vtx_bufs.count(var)) {
            return;
        }
        DRESSI_CHECK(!var.getCreator(),
                     "GPU-generated vertex attributes are not supported yet");
        const ImgSize size = var.getImgSize();
        const size_t n_elems =
                size_t(size.w) * size.h * NumComponents(var.getVType());
        plan.vtx_bufs[var] = vkw::CreateBufferPack(
                ctx.physical_device, ctx.device, n_elems * 4,
                is_index ? vk::BufferUsageFlagBits::eIndexBuffer
                         : vk::BufferUsageFlagBits::eVertexBuffer,
                vkw::HOST_VISIB_COHER_PROPS);
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

    // Records one RASTER stage: indexed depth-tested draw into a cleared
    // attribute attachment (deferred-shading G-buffer channel)
    const auto record_raster_stage = [&](const Stage& stage) {
        const SubStage& ss = stage.substages[0];
        const Variable& out = ss.out_vars[0];
        const Variable& pos_var = ss.vtx_vars[0];
        const Variable& attr_var = ss.vtx_vars[1];
        const Variable& faces_var = ss.vtx_vars[2];

        auto rp = vkw::CreateRenderPassPack();
        vkw::AddAttachientDesc(rp, FormatOf(out.getVType()),
                               vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::AttachmentLoadOp::eClear,
                               vk::AttachmentStoreOp::eStore);
        const auto depth_format = vk::Format::eD32Sfloat;
        vkw::AddAttachientDesc(
                rp, depth_format, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                vk::AttachmentLoadOp::eClear,
                vk::AttachmentStoreOp::eDontCare);
        vkw::AddSubpassDesc(
                rp, {}, {{0, vk::ImageLayout::eColorAttachmentOptimal}},
                {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
        vkw::UpdateRenderPass(ctx.device, rp);
        plan.render_passes.push_back(rp);

        auto depth_img = vkw::CreateImagePack(
                ctx.physical_device, ctx.device, depth_format,
                {stage.img_size.w, stage.img_size.h}, 1,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, true,
                vk::ImageAspectFlagBits::eDepth);
        plan.depth_imgs.push_back(depth_img);

        auto fb = vkw::CreateFrameBuffer(ctx.device, rp,
                                         {plan.imgs.at(out), depth_img});
        plan.frame_buffers.push_back(fb);

        const RasterShaders shaders = GenerateRasterShaders(ss);
        auto vert_module = ctx.glsl_compiler->compileFromString(
                ctx.device, shaders.vert, vk::ShaderStageFlagBits::eVertex);
        auto frag_module = ctx.glsl_compiler->compileFromString(
                ctx.device, shaders.frag, vk::ShaderStageFlagBits::eFragment);

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
        pipeline_info.color_blend_infos.resize(1);
        auto pipeline = vkw::CreateGraphicsPipeline(
                ctx.device, {vert_module, frag_module},
                {{0, 16, vk::VertexInputRate::eVertex},
                 {1, attr_comps * 4, vk::VertexInputRate::eVertex}},
                {{0, 0, vk::Format::eR32G32B32A32Sfloat, 0},
                 {1, 1, attr_fmt, 0}},
                pipeline_info, {}, rp, 0);
        plan.pipelines.push_back(pipeline);

        const std::vector<vk::ClearValue> clear_vals = {
                vk::ClearValue(vk::ClearColorValue(
                        std::array<float, 4>{0.f, 0.f, 0.f, 0.f})),
                vk::ClearValue(vk::ClearDepthStencilValue(1.f, 0))};
        vkw::CmdBeginRenderPass(cmd, rp, fb, clear_vals);
        vkw::CmdBindPipeline(cmd, pipeline);
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

        vkw::BarrierImage(cmd, plan.imgs.at(out),
                          vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::PipelineStageFlagBits::eColorAttachmentOutput,
                          vk::AccessFlagBits::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits::eFragmentShader,
                          vk::AccessFlagBits::eShaderRead |
                                  vk::AccessFlagBits::eInputAttachmentRead);
    };

    for (const auto& stage : stages) {
        if (stage.shader_type == RASTER) {
            record_raster_stage(stage);
            continue;
        }
        DRESSI_CHECK(stage.shader_type == FRAG,
                     "COMP stages are not executable yet");
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
                vkw::AddAttachientDesc(
                        rp, FormatOf(v.getVType()),
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::AttachmentLoadOp::eDontCare,
                        vk::AttachmentStoreOp::eStore);
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
            std::vector<vkw::AttachmentRefInfo> inp_refs;
            for (const auto& v : ss.inp_vars) {
                inp_refs.emplace_back(attach_idx.at(v),
                                      vk::ImageLayout::eShaderReadOnlyOptimal);
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
            for (const auto& v : ss.out_vars) {
                col_refs.emplace_back(attach_idx.at(v),
                                      vk::ImageLayout::eColorAttachmentOptimal);
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

            // Binding convention shared with the codegen: inp, tex, then slt
            vkw::DescSetPackPtr desc_set;
            if (!ss.inp_vars.empty() || !ss.tex_vars.empty() ||
                !ss.slt_vars.empty()) {
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
                desc_set = vkw::CreateDescriptorSetPack(ctx.device,
                                                        binding_infos);
                auto write_pack = vkw::CreateWriteDescSetPack();
                for (size_t i = 0; i < ss.inp_vars.size(); i++) {
                    vkw::AddWriteDescSet(
                            write_pack, desc_set, uint32_t(i),
                            {plan.imgs.at(ss.inp_vars[i])},
                            {vk::ImageLayout::eShaderReadOnlyOptimal});
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
        for (const auto& [v, ss_idx] : writer_subpass) {
            (void)ss_idx;
            vkw::BarrierImage(cmd, plan.imgs.at(v),
                              vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eColorAttachmentWrite,
                              vk::PipelineStageFlagBits::eFragmentShader,
                              vk::AccessFlagBits::eShaderRead |
                                      vk::AccessFlagBits::eInputAttachmentRead);
        }
    }

    // Copy optimizer outputs back into their input images so the next
    // iteration reads the updated values
    for (const auto& [upd, inp] : upd_inp_map) {
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
}

}  // namespace dressi
