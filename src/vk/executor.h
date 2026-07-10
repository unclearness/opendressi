#ifndef DRESSI_VK_EXECUTOR_H
#define DRESSI_VK_EXECUTOR_H

#include <vkw/vkw.h>

#include <map>

#include "dressi/variable.h"
#include "pack/substage.h"
#include "vk/context.h"

namespace dressi {

// Built Vulkan objects for a packed stage list ("ParseStagesAsVulkanObjects"
// of the Dressi paper). Images persist across rebuilds via the map passed in.
struct GpuPlan {
    std::map<Variable, vkw::ImagePackPtr> imgs;
    std::map<Variable, vkw::TexturePackPtr> textures;  // nearest samplers
    // Vertex/index buffers for RASTER inputs (leaf geometry data)
    std::map<Variable, vkw::BufferPackPtr> vtx_bufs;
    std::vector<vkw::ImagePackPtr> depth_imgs;  // one per RASTER stage
    std::vector<vkw::RenderPassPackPtr> render_passes;
    std::vector<vkw::FrameBufferPackPtr> frame_buffers;
    std::vector<vkw::PipelinePackPtr> pipelines;
    std::vector<vkw::DescSetPackPtr> desc_sets;
    vkw::CommandBuffersPackPtr cmd_pack;
    vkw::FencePtr fence;
};

// Creates images (reusing `prev_imgs` entries; aliasing updated->input pairs
// through `upd_inp_map`), builds render passes/pipelines/descriptor sets from
// the stages, and records the full command buffer.
GpuPlan BuildGpuPlan(const VkContext& ctx, const Stages& stages,
                     const std::map<Variable, Variable>& upd_inp_map,
                     std::map<Variable, vkw::ImagePackPtr> prev_imgs);

// Submits the recorded command buffer and waits for completion.
void ExecuteGpuPlan(const VkContext& ctx, GpuPlan& plan);

}  // namespace dressi

#endif  // DRESSI_VK_EXECUTOR_H
