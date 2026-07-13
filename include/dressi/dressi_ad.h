#ifndef DRESSI_DRESSI_AD_H
#define DRESSI_DRESSI_AD_H

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dressi/types.h"
#include "dressi/variable.h"

namespace dressi {

struct VkContext;
using VkContextPtr = std::shared_ptr<VkContext>;

// DressiAD: DR-specialized AD library class (Appendix A of the Dressi paper).
// Owns the Vulkan context and the staged build pipeline of execStep():
// backward construction, optimizer wiring, graph traversal, substage/stage
// packing, Vulkan object creation, and command execution.
class DressiAD {
public:
    // Optimizer function
    // (takes inputs and their gradients, and returns optimized outputs)
    using Optimizer = std::function<Variables(Variables xs, Variables gxs)>;

    // Building status
    enum InitStatus {
        BACKWARD,
        OPTIMIZER,
        TRAVERSE,
        SUBSTAGE,
        STAGE,
        VULKAN,
        FINISHED,
    };

    enum class PackingMode {
        Naive,  // 1 function = 1 pass (correctness baseline)
        RSP,    // reactive shader packing
    };

    DressiAD();
    // Shares an existing Vulkan context (instance/device/queue) instead of
    // creating one per instance. Lets many DressiAD graphs coexist on one
    // device (e.g. a shape-keyed engine cache in language bindings).
    explicit DressiAD(VkContextPtr ctx);
    ~DressiAD();
    DressiAD(const DressiAD&) = delete;
    DressiAD& operator=(const DressiAD&) = delete;

    // Creates a headless Vulkan context suitable for the shared-context
    // constructor. `debug` (or DRESSI_VK_DEBUG=1) enables validation layers.
    static VkContextPtr createContext(bool debug = false);

    // Setters
    void setLossVar(const Variable& loss_var);
    void setOptimizer(const Optimizer& optim_func);
    void setPackingMode(PackingMode mode);
    // Reactive cache thresholds: after the dirty pattern has been stable for
    // `fast` iterations a fast rebuild (stage packing only) prunes cached
    // branches; after `full` iterations a full rebuild (substage packing)
    // does. Pass 0 to disable either.
    void setRebuildCounts(uint32_t fast, uint32_t full);

    // Forces the variable to stay a substage output so recvImg() can read
    // it even when packing would otherwise fuse it into a shader. Call
    // before the first execStep() (or a rebuild follows).
    void markOutput(const Variable& var);

    // Registers an extra end-of-frame copy-back: `updated` (a computed
    // variable) overwrites `input_leaf`'s image each iteration, exactly
    // like optimizer outputs. Lets optimizer STATE (momenta, iteration
    // counters, ...) live entirely on the GPU with no per-frame CPU
    // traffic. Safe to call from inside the optimizer lambda (once).
    void addUpdate(const Variable& input_leaf, const Variable& updated);

    // Keeps every requires-grad leaf's gradient variable alive as a graph
    // root and substage output so recvImg() can read raw gradients without
    // wiring an optimizer. Call before the first execStep() (or a rebuild
    // follows).
    void setGradOutputsEnabled(bool enable);
    // (requires-grad leaf, gradient variable) pairs of the current backward
    // graph. Valid after the first execStep(); pair with
    // setGradOutputsEnabled(true) to make the gradients recvImg-readable.
    std::vector<std::pair<Variable, Variable>> inputGrads() const;

    // Introspection (current build). getFuncCount() is the unpacked backward-
    // graph op count (the trivial-packer 1-func-per-pass baseline); comparing
    // it to getSubStageCount()/getStageCount() shows how far substage/stage
    // packing collapsed it. The packed counts are device-dependent — greedy
    // fusion is bounded by the physical device's Vulkan limits.
    size_t getFuncCount() const;
    size_t getStageCount() const;
    size_t getSubStageCount() const;
    // Vulkan physical-device name (e.g. "Adreno (TM) 740"); empty until a
    // context exists (shared-context construction or the first execStep).
    std::string getDeviceName() const;

    // Execute one step of rendering and optimization
    void execStep();
    // Upload the given image leaves and execute in ONE submit: on the
    // steady-state fast path the uploads are recorded into the same
    // command-buffer batch as the render plan, removing the separate
    // upload fence wait (the eager binding hot path). Views are borrowed
    // for the duration of the call.
    void execStepWithSends(
            const std::vector<std::pair<Variable, CpuImageView>>& items);
    // Upload leaves, execute, and read same-shape outputs with one
    // vkQueueSubmit/fence wait on the steady-state path.
    CpuImage execStepWithSendsAndRecvImgsStacked(
            const std::vector<std::pair<Variable, CpuImageView>>& sends,
            const Variables& recvs);

    // Transfer image data between CPU and GPU. Sends before the first
    // execStep() are kept pending and flushed once images exist.
    void sendImg(const Variable& var, const CpuImage& cpu_img);
    // Zero-copy variant for borrowed buffers (language bindings): the view
    // is only read during the call (copied internally if still pending).
    void sendImg(const Variable& var, const CpuImageView& cpu_img);
    // Batched upload: all image transfers share one staging buffer and one
    // submit (each individual sendImg costs a synchronous fence wait).
    void sendImgs(const std::vector<std::pair<Variable, CpuImageView>>& items);
    CpuImage recvImg(const Variable& var);
    // Batched download: all images share one staging buffer and one submit
    std::vector<CpuImage> recvImgs(const Variables& vars);
    // Same, but all images must share one size/vtype; returns them stacked
    // as a single {w, h * n} image (callers reshape to a batch) so the
    // batch needs no per-item copies downstream
    CpuImage recvImgsStacked(const Variables& vars);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace dressi

#endif  // DRESSI_DRESSI_AD_H
