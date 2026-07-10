#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "core/node.h"
#include "pack/packing.h"

namespace dressi {

namespace {

bool Contains(const Variables& vars, const Variable& v) {
    return std::find(vars.begin(), vars.end(), v) != vars.end();
}

void PushUnique(Variables& vars, const Variable& v) {
    if (!Contains(vars, v)) {
        vars.push_back(v);
    }
}

void Erase(Variables& vars, const Variable& v) {
    vars.erase(std::remove(vars.begin(), vars.end(), v), vars.end());
}

InputAccess AccessOf(const Function& func, size_t input_idx) {
    const auto& access = NodeAccess::Node(func)->input_access;
    return access.empty() ? InputAccess::SamePixel : access[input_idx];
}

// Tries to add `func` in front of the substage (producers join after their
// consumers). Returns false if the placement is structurally impossible.
bool PushFrontFuncIntoSubStage(const Function& func, SubStage& ss,
                               const std::set<Variable>& used_vars) {
    const Variable y = func.getOutputVar();
    const ImgSize y_size = y.getImgSize();

    if (ss.funcs.empty()) {
        ss.img_size = y_size;
        ss.shader_type = func.getShaderType();
    } else {
        // All functions of a substage evaluate on the same pixel grid
        if (y_size != ss.img_size ||
            func.getShaderType() != ss.shader_type) {
            return false;
        }
        // One draw per RASTER substage
        if (func.getShaderType() == RASTER) {
            return false;
        }
    }

    // A texelFetch consumer in this substage reads y from another pass;
    // generating y here as well would require sampling a mid-pass attachment
    if (Contains(ss.slt_vars, y) || Contains(ss.tex_vars, y) ||
        Contains(ss.vtx_vars, y)) {
        return false;
    }

    // Fuse: y is now generated inside instead of read as input attachment
    Erase(ss.inp_vars, y);
    ss.gen_vars.push_back(y);
    if (used_vars.count(y)) {
        PushUnique(ss.out_vars, y);
    }

    const Variables xs = func.getInputVars();
    for (size_t i = 0; i < xs.size(); i++) {
        const Variable& x = xs[i];
        if (IsInlineConst(x)) {
            continue;
        }
        if (func.getShaderType() == RASTER) {
            ss.vtx_vars.push_back(x);  // ordered: pos, attrib, faces
        } else if (AccessOf(func, i) == InputAccess::TexelFetch) {
            PushUnique(ss.slt_vars, x);
        } else if (AccessOf(func, i) == InputAccess::Sampled) {
            PushUnique(ss.tex_vars, x);
        } else if (x.getImgSize() == ss.img_size) {
            PushUnique(ss.inp_vars, x);  // same-pixel read
        } else if (x.getImgSize().isUniform()) {
            PushUnique(ss.slt_vars, x);  // uniform read via texelFetch(0,0)
        } else {
            return false;  // size-mismatched same-pixel read is impossible
        }
    }

    // texelFetch/sampled inputs must come from another substage
    for (const auto& gv : ss.gen_vars) {
        if (Contains(ss.slt_vars, gv) || Contains(ss.tex_vars, gv)) {
            return false;
        }
    }

    ss.funcs.push_back(func);
    return true;
}

bool IsSubStageVkLimitsSatisfied(const SubStage& ss,
                                 const PackingLimits& limits) {
    if (ss.inp_vars.size() > limits.max_input_attachments) {
        return false;
    }
    if (ss.out_vars.size() > limits.max_output_attachments) {
        return false;
    }
    if (ss.slt_vars.size() + ss.tex_vars.size() >
        limits.max_sampled_images) {
        return false;
    }
    for (const auto& v : ss.out_vars) {
        if (v.getImgSize() != ss.img_size) {
            return false;
        }
    }
    return true;
}

// Edges between `func` and the active substage: consumers of its output
// inside the substage plus shared input variables.
uint32_t CountEdges(const Function& func, const SubStage& ss) {
    uint32_t edges = 0;
    const Variable y = func.getOutputVar();
    for (const auto& packed : ss.funcs) {
        for (const auto& x : packed.getInputVars()) {
            if (x == y) {
                edges++;
            }
        }
    }
    for (const auto& x : func.getInputVars()) {
        if (Contains(ss.inp_vars, x) || Contains(ss.slt_vars, x)) {
            edges++;
        }
    }
    return edges;
}

}  // namespace

SubStages PackFunctionsIntoSubStages(const Functions& dirty_funcs,
                                     const Variables& root_vars,
                                     const PackingLimits& limits) {
    // Constants are inlined into consumer shaders, never packed
    Functions packable;
    for (const auto& f : dirty_funcs) {
        if (!IsInlineConst(f.getOutputVar())) {
            packable.push_back(f);
        }
    }
    // Remaining functions and, per variable, how many remaining functions
    // consume it (for CollectLatestFuncs)
    std::set<Function> remaining(packable.begin(), packable.end());
    std::unordered_map<uint64_t, uint32_t> pending_consumers;
    for (const auto& f : remaining) {
        for (const auto& x : f.getInputVars()) {
            pending_consumers[x.id()]++;
        }
    }

    std::set<Variable> used_vars(root_vars.begin(), root_vars.end());
    SubStages substages;  // collected in reverse execution order
    SubStage active;

    const auto close_active = [&]() {
        if (active.funcs.empty()) {
            return;
        }
        // Inputs of the closed substage are demanded from earlier substages
        for (const auto& v : active.inp_vars) {
            used_vars.insert(v);
        }
        for (const auto& v : active.slt_vars) {
            used_vars.insert(v);
        }
        for (const auto& v : active.tex_vars) {
            used_vars.insert(v);
        }
        substages.push_back(std::move(active));
        active = SubStage{};
    };

    while (!remaining.empty()) {
        // Functions whose outputs are not consumed by any remaining function
        Functions candidates;
        for (const auto& f : remaining) {
            const auto it = pending_consumers.find(f.getOutputVar().id());
            if (it == pending_consumers.end() || it->second == 0) {
                candidates.push_back(f);
            }
        }
        DRESSI_CHECK(!candidates.empty(),
                     "Packing found no latest function (graph cycle?)");

        // Prefer candidates with more edges into the active substage
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](const Function& a, const Function& b) {
                             const uint32_t ea = CountEdges(a, active);
                             const uint32_t eb = CountEdges(b, active);
                             if (ea != eb) {
                                 return ea > eb;
                             }
                             return a.id() > b.id();
                         });

        bool found = false;
        for (const auto& f : candidates) {
            SubStage trial = active;
            if (PushFrontFuncIntoSubStage(f, trial, used_vars) &&
                IsSubStageVkLimitsSatisfied(trial, limits)) {
                active = std::move(trial);
                remaining.erase(f);
                for (const auto& x : f.getInputVars()) {
                    pending_consumers[x.id()]--;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            DRESSI_CHECK(!active.funcs.empty(),
                         "No function is packable into an empty substage");
            close_active();
        }
    }
    close_active();

    std::reverse(substages.begin(), substages.end());
    return substages;
}

Stages PackSubStagesIntoStages(SubStages substages,
                               const PackingLimits& limits) {
    Stages stages;
    Stage current;

    const auto close_current = [&]() {
        if (current.substages.empty()) {
            return;
        }
        // Aggregate stage-level I/O (inputs produced within the stage stay
        // subpass-internal but remain attachments either way)
        for (const auto& ss : current.substages) {
            for (const auto& v : ss.inp_vars) {
                PushUnique(current.inp_vars, v);
            }
            for (const auto& v : ss.slt_vars) {
                PushUnique(current.slt_vars, v);
            }
            for (const auto& v : ss.out_vars) {
                PushUnique(current.out_vars, v);
            }
        }
        stages.push_back(std::move(current));
        current = Stage{};
    };

    for (auto& ss : substages) {
        bool compatible = !current.substages.empty();
        if (compatible &&
            (ss.shader_type != current.shader_type ||
             ss.img_size != current.img_size)) {
            compatible = false;
        }
        // RASTER substages keep their own render pass (dedicated depth
        // attachment and clear semantics)
        if (compatible && (ss.shader_type == RASTER ||
                           current.shader_type == RASTER)) {
            compatible = false;
        }
        if (compatible) {
            // texelFetch cannot read an attachment written by this render
            // pass; such substages must start a new stage
            for (const auto& slt : ss.slt_vars) {
                for (const auto& prev : current.substages) {
                    if (Contains(prev.out_vars, slt)) {
                        compatible = false;
                    }
                }
            }
        }
        if (compatible) {
            // Attachment budget: writes + external input attachments
            std::set<Variable> attachments;
            for (const auto& prev : current.substages) {
                for (const auto& v : prev.out_vars) {
                    attachments.insert(v);
                }
                for (const auto& v : prev.inp_vars) {
                    attachments.insert(v);
                }
            }
            for (const auto& v : ss.out_vars) {
                attachments.insert(v);
            }
            for (const auto& v : ss.inp_vars) {
                attachments.insert(v);
            }
            if (attachments.size() > limits.max_stage_attachments) {
                compatible = false;
            }
        }

        if (!compatible) {
            close_current();
            current.shader_type = ss.shader_type;
            current.img_size = ss.img_size;
        }
        current.substages.push_back(std::move(ss));
    }
    close_current();
    return stages;
}

Stages GreedyPack(const Functions& funcs, const Variables& root_vars,
                  const PackingLimits& limits) {
    Variables roots;
    for (const auto& r : root_vars) {
        if (r && !IsInlineConst(r)) {
            roots.push_back(r);
        }
    }
    SubStages substages = PackFunctionsIntoSubStages(funcs, roots, limits);
    return PackSubStagesIntoStages(std::move(substages), limits);
}

}  // namespace dressi
