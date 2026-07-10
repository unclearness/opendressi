#ifndef DRESSI_VK_TRANSFER_H
#define DRESSI_VK_TRANSFER_H

#include <vkw/vkw.h>

#include "dressi/types.h"
#include "vk/context.h"

namespace dressi {

// VkFormat for a VType image (float family only in M1; VEC3 pads to RGBA)
vk::Format FormatOf(VType vtype);

// CPU -> GPU: re-strides logical channels to the physical format (VEC3 pad)
// and copies via a staging buffer. Leaves the image in ShaderReadOnlyOptimal.
void SendImageToDevice(const VkContext& ctx, const vkw::ImagePackPtr& img,
                       const CpuImageView& cpu_img, VType vtype,
                       bool image_initialized);
inline void SendImageToDevice(const VkContext& ctx,
                              const vkw::ImagePackPtr& img,
                              const CpuImage& cpu_img, VType vtype,
                              bool image_initialized) {
    SendImageToDevice(ctx, img, CpuImageView(cpu_img), vtype,
                      image_initialized);
}

// Batched CPU -> GPU upload: every image rides one staging buffer and one
// command-buffer submit (a per-image submit costs a fence wait each)
struct ImageSendItem {
    vkw::ImagePackPtr img;
    CpuImageView cpu;
    VType vtype = FLOAT;
    bool initialized = true;
};
void SendImagesToDevice(const VkContext& ctx,
                        const std::vector<ImageSendItem>& items);

// GPU -> CPU counterpart (expects ShaderReadOnlyOptimal layout)
CpuImage ReceiveImageFromDevice(const VkContext& ctx,
                                const vkw::ImagePackPtr& img, VType vtype);

// Batched GPU -> CPU download: one staging buffer, one submit, one map
std::vector<CpuImage> ReceiveImagesFromDevice(
        const VkContext& ctx,
        const std::vector<std::pair<vkw::ImagePackPtr, VType>>& items);

// Same-shape variant: strips every image into one contiguous {w, h*n}
// buffer (no per-item allocations; callers view it as a batch)
CpuImage ReceiveImagesStacked(
        const VkContext& ctx,
        const std::vector<std::pair<vkw::ImagePackPtr, VType>>& items);

// Fills a host-visible vertex/index buffer from CPU data. Int VTypes
// (face indices) are converted from the CpuImage's float storage to uint32.
void SendGeometryToBuffer(const VkContext& ctx,
                          const vkw::BufferPackPtr& buf,
                          const CpuImageView& cpu_img, VType vtype);
inline void SendGeometryToBuffer(const VkContext& ctx,
                                 const vkw::BufferPackPtr& buf,
                                 const CpuImage& cpu_img, VType vtype) {
    SendGeometryToBuffer(ctx, buf, CpuImageView(cpu_img), vtype);
}

}  // namespace dressi

#endif  // DRESSI_VK_TRANSFER_H
