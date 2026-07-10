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
                       const CpuImage& cpu_img, VType vtype,
                       bool image_initialized);

// GPU -> CPU counterpart (expects ShaderReadOnlyOptimal layout)
CpuImage ReceiveImageFromDevice(const VkContext& ctx,
                                const vkw::ImagePackPtr& img, VType vtype);

// Fills a host-visible vertex/index buffer from CPU data. Int VTypes
// (face indices) are converted from the CpuImage's float storage to uint32.
void SendGeometryToBuffer(const VkContext& ctx,
                          const vkw::BufferPackPtr& buf,
                          const CpuImage& cpu_img, VType vtype);

}  // namespace dressi

#endif  // DRESSI_VK_TRANSFER_H
