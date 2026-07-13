#ifndef DRESSI_EXAMPLES_BENCH_H
#define DRESSI_EXAMPLES_BENCH_H

#include <cstdint>
#include <string>
#include <vector>

namespace dressi_examples {

// Median of per-iteration wall-time samples (0 when empty). Steady-state
// median with the warmup excluded is THE cross-device metric — see
// CLAUDE.md "Benchmarking".
double MedianMs(std::vector<double> samples);

// One-time build/warmup overhead in ms: the extra wall time the excluded
// warmup iterations cost beyond the steady-state rate. Captures the full
// pack + GLSL-compile + Vulkan-build ladder AND the reactive-cache
// rebuilds at iter ~2 and ~8 (all within the warmup window).
// `warmup_samples` are the per-iter ms of the excluded warmup iters;
// `steady_median` is the post-warmup MedianMs. Rough by design (wall
// clock, not the per-phase breakdown) — see CLAUDE.md "Benchmarking".
double WarmupMs(const std::vector<double>& warmup_samples,
                double steady_median);

// One flat-JSON benchmark record for scripts/bench_summary.py. The
// constructor fills the common keys (example, device, platform); add the
// example's parameters and metrics, then save() into the run's out dir.
class BenchRecord {
public:
    BenchRecord(const std::string& example, const std::string& device);

    void add(const std::string& key, const std::string& value);
    void add(const std::string& key, double value, int precision = 4);
    void add(const std::string& key, int64_t value);
    void add(const std::string& key, bool value);

    // Record the packing reduction as three keys (funcs, substages, stages):
    // the unpacked backward-graph op count collapsed by substage packing then
    // by stage packing into Vulkan render passes. Pass the steady-state build's
    // DressiAD::getFuncCount()/getSubStageCount()/getStageCount(). The packed
    // counts are DEVICE-DEPENDENT — greedy fusion is bounded by the physical
    // device's Vulkan limits, so the same graph packs differently per GPU.
    void addPacking(int64_t funcs, int64_t substages, int64_t stages);

    void save(const std::string& path) const;

private:
    std::string m_json;  // accumulated "key":value pairs
};

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_BENCH_H
