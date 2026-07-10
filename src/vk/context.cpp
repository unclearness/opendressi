#include "vk/context.h"

namespace dressi {

VkContextPtr CreateVkContext(bool debug_enable) {
    auto ctx = std::make_shared<VkContext>();
    ctx->instance = vkw::CreateInstance("dressi", 1, "dressi", 1, debug_enable,
                                        /*surface_enable=*/false);
    ctx->physical_device = vkw::GetFirstPhysicalDevice(ctx->instance);
    ctx->queue_family_idx = vkw::GetQueueFamilyIdx(ctx->physical_device);
    // Enable the physical device's feature set (gl_PrimitiveID in fragment
    // shaders needs the Geometry capability from the geometryShader feature)
    ctx->device = vkw::CreateDevice(ctx->queue_family_idx,
                                    ctx->physical_device, 1,
                                    /*swapchain_support=*/false,
                                    vkw::GetPhysicalFeatures2(
                                            ctx->physical_device));
    ctx->queue = vkw::GetQueue(ctx->device, ctx->queue_family_idx);
    ctx->limits = vkw::GetPhysicalProperties(ctx->physical_device)->limits;
    ctx->glsl_compiler = std::make_unique<vkw::GLSLCompiler>();
    return ctx;
}

void RunOneShot(
        const VkContext& ctx,
        const std::function<void(const vk::UniqueCommandBuffer&)>& record) {
    auto cmd_pack = vkw::CreateCommandBuffersPack(ctx.device,
                                                  ctx.queue_family_idx, 1);
    const auto& cmd = cmd_pack->cmd_bufs[0];
    vkw::BeginCommand(cmd, /*one_time_submit=*/true);
    record(cmd);
    vkw::EndCommand(cmd);
    auto fence = vkw::CreateFence(ctx.device);
    vkw::QueueSubmit(ctx.queue, cmd, fence);
    vkw::WaitForFence(ctx.device, fence);
}

}  // namespace dressi
