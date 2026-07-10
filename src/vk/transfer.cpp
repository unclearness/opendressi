#include "vk/transfer.h"

#include <cstring>

#include "codegen/glsl_codegen.h"

namespace dressi {

namespace {

// Grow-on-demand persistent staging. `cached` requests HostCached memory
// (falling back to plain coherent where unavailable) -- required for
// readback, where the CPU-side memcpy reads the mapped memory.
vkw::BufferPackPtr& EnsureStaging(const VkContext& ctx,
                                  vkw::BufferPackPtr& slot, size_t bytes,
                                  vk::BufferUsageFlags usage, bool cached) {
    if (slot && slot->size >= bytes) {
        return slot;
    }
    // Mutating the shared_ptr-held context's scratch: transfers are
    // synchronous and externally serialized (one queue)
    if (cached) {
        constexpr auto kCachedProps = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent |
                                      vk::MemoryPropertyFlagBits::eHostCached;
        try {
            slot = vkw::CreateBufferPack(ctx.physical_device, ctx.device,
                                         bytes, usage, kCachedProps);
            return slot;
        } catch (const std::runtime_error&) {
            // No cached+coherent memory type on this device; fall through
        }
    }
    slot = vkw::CreateBufferPack(ctx.physical_device, ctx.device, bytes,
                                 usage, vkw::HOST_VISIB_COHER_PROPS);
    return slot;
}

}  // namespace

vk::Format FormatOf(VType vtype) {
    switch (PhysChannels(vtype)) {
        case 1: return vk::Format::eR32Sfloat;
        case 2: return vk::Format::eR32G32Sfloat;
        case 4: return vk::Format::eR32G32B32A32Sfloat;
    }
    DRESSI_CHECK(false, "Unsupported VType for GPU image");
}

void SendImageToDevice(const VkContext& ctx, const vkw::ImagePackPtr& img,
                       const CpuImageView& cpu_img, VType vtype,
                       bool image_initialized) {
    const uint32_t n_logical = NumComponents(vtype);
    const uint32_t n_phys = PhysChannels(vtype);
    DRESSI_CHECK(cpu_img.channels == n_logical,
                 "sendImg: channel count does not match the Variable's VType");
    DRESSI_CHECK(cpu_img.width == img->view_size.width &&
                         cpu_img.height == img->view_size.height,
                 "sendImg: image size mismatch");

    const size_t n_pixels = size_t(cpu_img.width) * cpu_img.height;
    const size_t n_bytes = n_pixels * n_phys * sizeof(float);
    const auto& staging =
            EnsureStaging(ctx, ctx.send_staging, n_bytes,
                          vk::BufferUsageFlagBits::eTransferSrc,
                          /*cached=*/false);
    if (n_logical == n_phys) {
        vkw::SendToDevice(ctx.device, staging, cpu_img.data, n_bytes);
    } else {
        std::vector<float> phys(n_pixels * n_phys, 0.f);
        for (size_t p = 0; p < n_pixels; p++) {
            for (uint32_t c = 0; c < n_logical; c++) {
                phys[p * n_phys + c] = cpu_img.data[p * n_logical + c];
            }
        }
        vkw::SendToDevice(ctx.device, staging, phys.data(), n_bytes);
    }

    const auto init_layout = image_initialized
                                     ? vk::ImageLayout::eShaderReadOnlyOptimal
                                     : vk::ImageLayout::eUndefined;
    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        vkw::CopyBufferToImage(cmd, staging, img, init_layout,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    });
}

void SendImagesToDevice(const VkContext& ctx,
                        const std::vector<ImageSendItem>& items) {
    if (items.empty()) {
        return;
    }
    if (items.size() == 1) {
        SendImageToDevice(ctx, items[0].img, items[0].cpu, items[0].vtype,
                          items[0].initialized);
        return;
    }

    // Layout: each image's physical pixels at a 16-byte-aligned offset
    std::vector<size_t> offsets(items.size());
    size_t total = 0;
    for (size_t i = 0; i < items.size(); i++) {
        const auto& it = items[i];
        const uint32_t n_phys = PhysChannels(it.vtype);
        DRESSI_CHECK(it.cpu.channels == NumComponents(it.vtype),
                     "sendImgs: channel count mismatch");
        DRESSI_CHECK(it.cpu.width == it.img->view_size.width &&
                             it.cpu.height == it.img->view_size.height,
                     "sendImgs: image size mismatch");
        offsets[i] = total;
        total += size_t(it.cpu.width) * it.cpu.height * n_phys *
                 sizeof(float);
        total = (total + 15) & ~size_t(15);
    }
    const auto& staging =
            EnsureStaging(ctx, ctx.send_staging, total,
                          vk::BufferUsageFlagBits::eTransferSrc,
                          /*cached=*/false);

    // Fill every region through one map
    auto* base = static_cast<uint8_t*>(ctx.device->mapMemory(
            staging->dev_mem_pack->dev_mem.get(), 0,
            staging->dev_mem_pack->dev_mem_size));
    for (size_t i = 0; i < items.size(); i++) {
        const auto& it = items[i];
        const uint32_t n_logical = NumComponents(it.vtype);
        const uint32_t n_phys = PhysChannels(it.vtype);
        const size_t n_pixels = size_t(it.cpu.width) * it.cpu.height;
        float* dst = reinterpret_cast<float*>(base + offsets[i]);
        if (n_logical == n_phys) {
            std::memcpy(dst, it.cpu.data,
                        n_pixels * n_phys * sizeof(float));
        } else {
            std::memset(dst, 0, n_pixels * n_phys * sizeof(float));
            for (size_t p = 0; p < n_pixels; p++) {
                for (uint32_t c = 0; c < n_logical; c++) {
                    dst[p * n_phys + c] = it.cpu.data[p * n_logical + c];
                }
            }
        }
    }
    ctx.device->unmapMemory(staging->dev_mem_pack->dev_mem.get());

    // One submit: transition, copy at offset, transition back -- per image
    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        constexpr auto kDst = vk::ImageLayout::eTransferDstOptimal;
        constexpr auto kRead = vk::ImageLayout::eShaderReadOnlyOptimal;
        for (size_t i = 0; i < items.size(); i++) {
            const auto& it = items[i];
            const auto init_layout = it.initialized
                                             ? kRead
                                             : vk::ImageLayout::eUndefined;
            vkw::SetImageLayout(cmd, it.img, init_layout, kDst);
            const vk::BufferImageCopy region(
                    offsets[i], it.cpu.width, it.cpu.height,
                    vk::ImageSubresourceLayers(
                            vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                    vk::Offset3D(0, 0, 0),
                    vk::Extent3D(it.cpu.width, it.cpu.height, 1));
            cmd->copyBufferToImage(staging->buf.get(),
                                   it.img->img_res_pack->img.get(), kDst,
                                   region);
            vkw::SetImageLayout(cmd, it.img, kDst, kRead);
        }
    });
}

void SendGeometryToBuffer(const VkContext& ctx,
                          const vkw::BufferPackPtr& buf,
                          const CpuImageView& cpu_img, VType vtype) {
    DRESSI_CHECK(cpu_img.channels == NumComponents(vtype),
                 "sendImg: channel count does not match the Variable's VType");
    const size_t n_elems = cpu_img.numElems();
    DRESSI_CHECK(buf->size >= n_elems * 4, "Geometry buffer size mismatch");
    if (IsIntVType(vtype)) {
        std::vector<uint32_t> ints(n_elems);
        for (size_t i = 0; i < n_elems; i++) {
            ints[i] = uint32_t(int64_t(cpu_img.data[i]));
        }
        vkw::SendToDevice(ctx.device, buf, ints.data(), n_elems * 4);
    } else {
        vkw::SendToDevice(ctx.device, buf, cpu_img.data, n_elems * 4);
    }
}

CpuImage ReceiveImageFromDevice(const VkContext& ctx,
                                const vkw::ImagePackPtr& img, VType vtype) {
    const uint32_t n_logical = NumComponents(vtype);
    const uint32_t n_phys = PhysChannels(vtype);
    const uint32_t w = img->view_size.width;
    const uint32_t h = img->view_size.height;
    const size_t n_pixels = size_t(w) * h;

    const size_t n_bytes = n_pixels * n_phys * sizeof(float);
    const auto& staging =
            EnsureStaging(ctx, ctx.recv_staging, n_bytes,
                          vk::BufferUsageFlagBits::eTransferDst,
                          /*cached=*/true);
    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        vkw::CopyImageToBuffer(cmd, img, staging,
                               vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    CpuImage out(w, h, n_logical);
    if (n_logical == n_phys) {
        vkw::RecvFromDevice(ctx.device, staging, out.data.data(), n_bytes);
    } else {
        std::vector<float> phys(n_pixels * n_phys);
        vkw::RecvFromDevice(ctx.device, staging, phys.data(), n_bytes);
        for (size_t p = 0; p < n_pixels; p++) {
            for (uint32_t c = 0; c < n_logical; c++) {
                out.data[p * n_logical + c] = phys[p * n_phys + c];
            }
        }
    }
    return out;
}

std::vector<CpuImage> ReceiveImagesFromDevice(
        const VkContext& ctx,
        const std::vector<std::pair<vkw::ImagePackPtr, VType>>& items) {
    std::vector<CpuImage> out;
    out.reserve(items.size());
    if (items.empty()) {
        return out;
    }
    if (items.size() == 1) {
        out.push_back(ReceiveImageFromDevice(ctx, items[0].first,
                                             items[0].second));
        return out;
    }

    std::vector<size_t> offsets(items.size());
    size_t total = 0;
    for (size_t i = 0; i < items.size(); i++) {
        const auto& [img, vtype] = items[i];
        offsets[i] = total;
        total += size_t(img->view_size.width) * img->view_size.height *
                 PhysChannels(vtype) * sizeof(float);
        total = (total + 15) & ~size_t(15);
    }
    const auto& staging =
            EnsureStaging(ctx, ctx.recv_staging, total,
                          vk::BufferUsageFlagBits::eTransferDst,
                          /*cached=*/true);

    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        constexpr auto kSrc = vk::ImageLayout::eTransferSrcOptimal;
        constexpr auto kRead = vk::ImageLayout::eShaderReadOnlyOptimal;
        for (size_t i = 0; i < items.size(); i++) {
            const auto& img = items[i].first;
            vkw::SetImageLayout(cmd, img, kRead, kSrc);
            const vk::BufferImageCopy region(
                    offsets[i], img->view_size.width, img->view_size.height,
                    vk::ImageSubresourceLayers(
                            vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                    vk::Offset3D(0, 0, 0),
                    vk::Extent3D(img->view_size.width,
                                 img->view_size.height, 1));
            cmd->copyImageToBuffer(img->img_res_pack->img.get(), kSrc,
                                   staging->buf.get(), region);
            vkw::SetImageLayout(cmd, img, kSrc, kRead);
        }
    });

    const auto* base = static_cast<const uint8_t*>(ctx.device->mapMemory(
            staging->dev_mem_pack->dev_mem.get(), 0,
            staging->dev_mem_pack->dev_mem_size));
    for (size_t i = 0; i < items.size(); i++) {
        const auto& [img, vtype] = items[i];
        const uint32_t n_logical = NumComponents(vtype);
        const uint32_t n_phys = PhysChannels(vtype);
        const uint32_t w = img->view_size.width;
        const uint32_t h = img->view_size.height;
        const size_t n_pixels = size_t(w) * h;
        const float* src = reinterpret_cast<const float*>(base + offsets[i]);
        CpuImage cpu(w, h, n_logical);
        if (n_logical == n_phys) {
            std::memcpy(cpu.data.data(), src,
                        n_pixels * n_phys * sizeof(float));
        } else {
            for (size_t p = 0; p < n_pixels; p++) {
                for (uint32_t c = 0; c < n_logical; c++) {
                    cpu.data[p * n_logical + c] = src[p * n_phys + c];
                }
            }
        }
        out.push_back(std::move(cpu));
    }
    ctx.device->unmapMemory(staging->dev_mem_pack->dev_mem.get());
    return out;
}

CpuImage ReceiveImagesStacked(
        const VkContext& ctx,
        const std::vector<std::pair<vkw::ImagePackPtr, VType>>& items) {
    DRESSI_CHECK(!items.empty(), "recvImgsStacked: empty batch");
    const auto& [img0, vtype] = items[0];
    const uint32_t w = img0->view_size.width;
    const uint32_t h = img0->view_size.height;
    const uint32_t n_logical = NumComponents(vtype);
    const uint32_t n_phys = PhysChannels(vtype);
    const size_t n_pixels = size_t(w) * h;
    const size_t item_bytes = n_pixels * n_phys * sizeof(float);
    const size_t stride = (item_bytes + 15) & ~size_t(15);
    for (const auto& [img, vt] : items) {
        DRESSI_CHECK(vt == vtype && img->view_size.width == w &&
                             img->view_size.height == h,
                     "recvImgsStacked: images must share one shape");
    }

    const auto& staging =
            EnsureStaging(ctx, ctx.recv_staging, stride * items.size(),
                          vk::BufferUsageFlagBits::eTransferDst,
                          /*cached=*/true);
    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        constexpr auto kSrc = vk::ImageLayout::eTransferSrcOptimal;
        constexpr auto kRead = vk::ImageLayout::eShaderReadOnlyOptimal;
        for (size_t i = 0; i < items.size(); i++) {
            const auto& img = items[i].first;
            vkw::SetImageLayout(cmd, img, kRead, kSrc);
            const vk::BufferImageCopy region(
                    stride * i, w, h,
                    vk::ImageSubresourceLayers(
                            vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                    vk::Offset3D(0, 0, 0), vk::Extent3D(w, h, 1));
            cmd->copyImageToBuffer(img->img_res_pack->img.get(), kSrc,
                                   staging->buf.get(), region);
            vkw::SetImageLayout(cmd, img, kSrc, kRead);
        }
    });

    const auto* base = static_cast<const uint8_t*>(ctx.device->mapMemory(
            staging->dev_mem_pack->dev_mem.get(), 0,
            staging->dev_mem_pack->dev_mem_size));
    CpuImage out(w, h * uint32_t(items.size()), n_logical);
    for (size_t i = 0; i < items.size(); i++) {
        const float* src = reinterpret_cast<const float*>(base + stride * i);
        float* dst = &out.data[i * n_pixels * n_logical];
        if (n_logical == n_phys) {
            std::memcpy(dst, src, item_bytes);
        } else {
            for (size_t p = 0; p < n_pixels; p++) {
                for (uint32_t c = 0; c < n_logical; c++) {
                    dst[p * n_logical + c] = src[p * n_phys + c];
                }
            }
        }
    }
    ctx.device->unmapMemory(staging->dev_mem_pack->dev_mem.get());
    return out;
}

}  // namespace dressi
