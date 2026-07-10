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
        if (y_size != ss.img_size) {
            return false;
        }
        if (func.getShaderType() != ss.shader_type) {
            if (func.getShaderType() == RASTER && ss.shader_type == FRAG) {
                // Raster-headed fusion (the paper's "same shader types
                // except for top rasterization"): a RASTER function may
                // join a FRAG substage as its top; packing is back-to-
                // front, so nothing can be pushed before it (the type
                // mismatch rejects it). The executor's raster pass binds
                // slt textures only, so the fused body must not need
                // input attachments or samplers.
                // The raster output must be consumed same-pixel here (it
                // arrives as the interpolated v_attr, not an attachment)
                if (!Contains(ss.inp_vars, y) || ss.inp_vars.size() != 1 ||
                    !ss.tex_vars.empty()) {
                    return false;
                }
                ss.shader_type = RASTER;
            } else if (func.getShaderType() == FRAG &&
                       ss.shader_type == COMP) {
                // FRAG snippets are shader-agnostic; they may fuse into a
                // COMP substage (same-pixel reads become texelFetch at the
                // invocation coordinate). COMP never joins FRAG: the
                // substage type is set by its latest function.
            } else {
                return false;
            }
        } else if (func.getShaderType() == RASTER) {
            return false;  // one draw per RASTER substage
        }
    }

    // A texelFetch/uniform consumer in this substage reads y from another
    // pass; generating y here as well would require reading a mid-pass value
    if (Contains(ss.slt_vars, y) || Contains(ss.tex_vars, y) ||
        Contains(ss.vtx_vars, y) || Contains(ss.uif_vars, y)) {
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
            if (AccessOf(func, i) == InputAccess::TexelFetch) {
                PushUnique(ss.slt_vars, x);  // raster FS texelFetch input
            } else {
                ss.vtx_vars.push_back(x);  // ordered: pos, attrib, faces
            }
        } else if (AccessOf(func, i) == InputAccess::TexelFetch) {
            PushUnique(ss.slt_vars, x);
        } else if (AccessOf(func, i) == InputAccess::Sampled) {
            if (ss.shader_type == COMP) {
                return false;  // implicit-LOD texture() is FRAG-only
            }
            PushUnique(ss.tex_vars, x);
        } else if (x.getImgSize() == ss.img_size) {
            // Same-pixel read. For COMP substages inp_vars are bound as
            // samplers and read by texelFetch at the invocation coordinate
            // (no input attachments in compute); the classification stays
            // `inp` so producers can still fuse in front.
            PushUnique(ss.inp_vars, x);
        } else if (x.getImgSize().isUniform()) {
            // {1,1} broadcast value. Leaves become real uniforms (vec4 UBO);
            // GPU-generated scalars stay texelFetch(0,0) reads -- refreshing
            // a UBO mid-frame costs more (copy + layout transitions) than
            // the broadcast fetch it would save.
            if (!x.getCreator()) {
                PushUnique(ss.uif_vars, x);
            } else {
                PushUnique(ss.slt_vars, x);
            }
        } else {
            return false;  // size-mismatched same-pixel read is impossible
        }
    }

    // texelFetch/sampled/uniform inputs must come from another substage
    for (const auto& gv : ss.gen_vars) {
        if (Contains(ss.slt_vars, gv) || Contains(ss.tex_vars, gv) ||
            Contains(ss.uif_vars, gv)) {
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
        if (Contains(ss.inp_vars, x) || Contains(ss.slt_vars, x) ||
            Contains(ss.uif_vars, x)) {
            edges++;
        }
    }
    return edges;
}

}  // namespace

// ---------------------------- Rematerialization ------------------------------
// A consumer substage recomputes cheap same-pixel producer chains instead of
// reading them through materialized attachments; afterwards, outputs nobody
// reads anymore are dropped (and fully dead substages removed). This trades
// ALU for image traffic -- the win is in wide backward chains where forward
// intermediates would otherwise cross subpass/stage boundaries as fp32
// attachments.

// Clone-safe = deterministic per-pixel value from bound inputs only.
// Coordinate-pure markers are whitelisted explicitly; raster/compute ops and
// everything touching fragment state stay put.
bool IsCloneSafe(const Function& f) {
    const auto& node = NodeAccess::Node(f);
    if (node->shader_type != FRAG) {
        return false;
    }
    const std::string& c = node->fwd_code;
    if (c.rfind("__", 0) == 0) {
        return c == "__face_fetch__" || c == "__screen_coord__";
    }
    return true;
}

void RematerializeSubStages(SubStages& substages, const Variables& root_vars,
                            const PackingLimits& limits) {
    for (auto& ss : substages) {
        if (ss.shader_type == RASTER) {
            continue;  // raster passes keep their exact shape
        }
        std::set<uint64_t> gen_ids, inp_ids, slt_ids, tex_ids, uif_ids;
        for (const auto& v : ss.gen_vars) {
            gen_ids.insert(v.id());
        }
        for (const auto& v : ss.inp_vars) {
            inp_ids.insert(v.id());
        }
        for (const auto& v : ss.slt_vars) {
            slt_ids.insert(v.id());
        }
        for (const auto& v : ss.tex_vars) {
            tex_ids.insert(v.id());
        }
        for (const auto& v : ss.uif_vars) {
            uif_ids.insert(v.id());
        }

        size_t budget = std::max<size_t>(32, ss.funcs.size());
        const Variables inps = ss.inp_vars;  // snapshot; mutated below
        for (const auto& x : inps) {
            Functions plan;
            Variables new_slt, new_uif;
            std::set<uint64_t> planned_fn, planned_out;
            const std::function<bool(const Variable&)> visit =
                    [&](const Variable& v) -> bool {
                const Function f = v.getCreator();
                if (!f || !IsCloneSafe(f) ||
                    f.getOutputVar().getImgSize() != ss.img_size) {
                    return false;
                }
                // A cloned value must not shadow an existing texelFetch
                // input (the generic slt load would collide with the clone)
                if (slt_ids.count(v.id()) || tex_ids.count(v.id()) ||
                    Contains(new_slt, v)) {
                    return false;
                }
                if (planned_fn.count(f.id())) {
                    return true;
                }
                if (plan.size() + 1 > budget) {
                    return false;
                }
                planned_fn.insert(f.id());
                const Variables xs = f.getInputVars();
                for (size_t i = 0; i < xs.size(); i++) {
                    const Variable& xi = xs[i];
                    if (IsInlineConst(xi)) {
                        continue;
                    }
                    const InputAccess acc = AccessOf(f, i);
                    if (acc == InputAccess::Sampled) {
                        return false;
                    }
                    if (acc == InputAccess::TexelFetch) {
                        // Arbitrary-coord reads need the materialized image;
                        // a value generated (or being cloned) in this shader
                        // cannot also be fetched -- reject the plan
                        if (gen_ids.count(xi.id()) ||
                            planned_out.count(xi.id())) {
                            return false;
                        }
                        if (!slt_ids.count(xi.id())) {
                            PushUnique(new_slt, xi);
                        }
                        continue;
                    }
                    // SamePixel: already available, or recurse
                    if (gen_ids.count(xi.id()) || inp_ids.count(xi.id()) ||
                        planned_out.count(xi.id())) {
                        continue;
                    }
                    if (xi.getImgSize() == ss.img_size) {
                        if (slt_ids.count(xi.id())) {
                            continue;  // same-size slt reads at own coord
                        }
                        if (!visit(xi)) {
                            return false;
                        }
                        continue;
                    }
                    if (xi.getImgSize().isUniform()) {
                        if (uif_ids.count(xi.id()) ||
                            slt_ids.count(xi.id())) {
                            continue;
                        }
                        if (!xi.getCreator()) {
                            PushUnique(new_uif, xi);
                        } else {
                            PushUnique(new_slt, xi);
                        }
                        continue;
                    }
                    return false;
                }
                plan.push_back(f);
                planned_out.insert(v.id());
                return true;
            };
            if (!visit(x)) {
                continue;
            }
            // Sampler budget must hold with the clone's extra fetch inputs
            size_t n_new_slt = 0;
            for (const auto& v : new_slt) {
                if (!slt_ids.count(v.id())) {
                    n_new_slt++;
                }
            }
            if (ss.slt_vars.size() + ss.tex_vars.size() + n_new_slt >
                limits.max_sampled_images) {
                continue;
            }
            // Apply the clone plan
            for (const auto& f : plan) {
                ss.funcs.push_back(f);
                const Variable y = f.getOutputVar();
                ss.gen_vars.push_back(y);
                gen_ids.insert(y.id());
            }
            for (const auto& v : new_slt) {
                PushUnique(ss.slt_vars, v);
                slt_ids.insert(v.id());
            }
            for (const auto& v : new_uif) {
                PushUnique(ss.uif_vars, v);
                uif_ids.insert(v.id());
            }
            Erase(ss.inp_vars, x);
            inp_ids.erase(x.id());
            budget -= std::min(budget, plan.size());
        }
    }

    // Dead-output elimination: an output survives only if a later substage
    // still reads it or it is externally demanded; substages left without
    // outputs are dropped entirely.
    std::set<uint64_t> demand;
    for (const auto& r : root_vars) {
        if (r) {
            demand.insert(r.id());
        }
    }
    std::vector<bool> alive(substages.size(), true);
    for (size_t i = substages.size(); i-- > 0;) {
        SubStage& ss = substages[i];
        Variables kept;
        for (const auto& v : ss.out_vars) {
            if (demand.count(v.id())) {
                kept.push_back(v);
            }
        }
        ss.out_vars = std::move(kept);
        if (ss.out_vars.empty()) {
            alive[i] = false;
            continue;
        }
        for (const Variables* vars :
             {&ss.inp_vars, &ss.slt_vars, &ss.tex_vars, &ss.vtx_vars,
              &ss.uif_vars}) {
            for (const auto& v : *vars) {
                demand.insert(v.id());
            }
        }
    }
    SubStages filtered;
    for (size_t i = 0; i < substages.size(); i++) {
        if (alive[i]) {
            filtered.push_back(std::move(substages[i]));
        }
    }
    substages = std::move(filtered);
}

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
        for (const auto& v : active.uif_vars) {
            used_vars.insert(v);
        }
        // GPU-generated raster geometry: the executor copies the vertex
        // buffer from the producer's image, so it must stay an output
        for (const auto& v : active.vtx_vars) {
            if (v.getCreator()) {
                used_vars.insert(v);
            }
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
            if (active.funcs.empty()) {
                std::string names;
                for (const auto& f : candidates) {
                    names += " " + NodeAccess::Node(f)->name;
                }
                DRESSI_CHECK(false,
                             "No function is packable into an empty "
                             "substage; candidates:" +
                                     names);
            }
            close_active();
        }
    }
    close_active();

    std::reverse(substages.begin(), substages.end());
    RematerializeSubStages(substages, root_vars, limits);
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
            for (const auto& v : ss.uif_vars) {
                PushUnique(current.uif_vars, v);
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
        // attachment and clear semantics); COMP substages are dispatches
        // outside any render pass
        if (compatible && (ss.shader_type != FRAG ||
                           current.shader_type != FRAG)) {
            compatible = false;
        }
        if (compatible) {
            // texelFetch cannot read an attachment written by this render
            // pass, and a uniform's image->buffer copy cannot be recorded
            // mid-pass; such substages must start a new stage
            for (const Variables* deps : {&ss.slt_vars, &ss.uif_vars}) {
                for (const auto& v : *deps) {
                    for (const auto& prev : current.substages) {
                        if (Contains(prev.out_vars, v)) {
                            compatible = false;
                        }
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
