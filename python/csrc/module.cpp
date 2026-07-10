// dressi._C: nanobind bindings of the minimal Dressi surface the
// torch layer needs (DressiAD engine, Variable graph handles, the F:: op
// subset used by the nvdiffrast-compatible ops, and the CPU mesh-topology
// builders). Graph construction only happens on engine-cache misses, so
// the binding stays thin; images cross as (H, W, C) float32 ndarrays.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include <spdlog/spdlog.h>

#include "dressi/dressi.h"
#include "dressi/mesh_utils.h"

namespace nb = nanobind;
using namespace dressi;

namespace {

using NdArrayIn =
        nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

CpuImage ToCpuImage(const NdArrayIn& arr) {
    const uint32_t h = uint32_t(arr.shape(0));
    const uint32_t w = uint32_t(arr.shape(1));
    const uint32_t c = uint32_t(arr.shape(2));
    CpuImage img(w, h, c);
    std::memcpy(img.data.data(), arr.data(), sizeof(float) * img.data.size());
    return img;
}

nb::ndarray<nb::numpy, float> ToNdArray(CpuImage&& img) {
    const size_t h = img.height, w = img.width, c = img.channels;
    auto* vec = new std::vector<float>(std::move(img.data));
    nb::capsule owner(vec, [](void* p) noexcept {
        delete static_cast<std::vector<float>*>(p);
    });
    return nb::ndarray<nb::numpy, float>(vec->data(), {h, w, c}, owner);
}

// Opaque holder: dressi::VkContext is intentionally incomplete in the
// public headers (its definition pulls in Vulkan-Hpp)
struct PyVkContext {
    VkContextPtr ptr;
};

}  // namespace

NB_MODULE(_C, m) {
    m.doc() = "opendressi core bindings (Dressi differentiable renderer)";
    // Engine/graph caches legitimately outlive the interpreter's last GC
    // pass (the torch layer caches DressiAD engines per shape signature)
    nb::set_leak_warnings(false);

    nb::exception<DressiError>(m, "DressiError");

    // "debug" prints per-phase execStep timings and the per-stage GPU
    // timestamp breakdown every 100 frames
    m.def("set_log_level", [](const std::string& level) {
        spdlog::set_level(spdlog::level::from_str(level));
    });

    nb::enum_<VType>(m, "VType")
            .value("FLOAT", VType::FLOAT)
            .value("VEC2", VType::VEC2)
            .value("VEC3", VType::VEC3)
            .value("VEC4", VType::VEC4)
            .value("INT", VType::INT)
            .value("IVEC2", VType::IVEC2)
            .value("IVEC3", VType::IVEC3)
            .value("IVEC4", VType::IVEC4);

    nb::class_<Variable>(m, "Variable")
            .def(nb::init<>())
            .def("__init__",
                 [](Variable* self, VType vtype, uint32_t w, uint32_t h) {
                     new (self) Variable(vtype, ImgSize{w, h});
                 },
                 nb::arg("vtype"), nb::arg("w"), nb::arg("h"))
            .def_prop_ro("vtype", &Variable::getVType)
            .def_prop_ro("width",
                         [](const Variable& v) { return v.getImgSize().w; })
            .def_prop_ro("height",
                         [](const Variable& v) { return v.getImgSize().h; })
            .def_prop_ro("id", &Variable::id)
            .def("set_requires_grad", &Variable::setRequiresGrad)
            .def("set_requires_grad_recursively",
                 &Variable::setRequiresGradRecursively,
                 nb::arg("req_grad") = true)
            .def("__bool__",
                 [](const Variable& v) { return bool(v); })
            .def("__eq__",
                 [](const Variable& a, const Variable& b) { return a == b; })
            .def("__hash__", [](const Variable& v) { return v.id(); })
            // Graph-building operators (Variable x Variable / float)
            .def("__add__", [](const Variable& a, const Variable& b) { return a + b; })
            .def("__sub__", [](const Variable& a, const Variable& b) { return a - b; })
            .def("__mul__", [](const Variable& a, const Variable& b) { return a * b; })
            .def("__truediv__", [](const Variable& a, const Variable& b) { return a / b; })
            .def("__add__", [](const Variable& a, float b) { return a + b; })
            .def("__sub__", [](const Variable& a, float b) { return a - b; })
            .def("__mul__", [](const Variable& a, float b) { return a * b; })
            .def("__truediv__", [](const Variable& a, float b) { return a / b; })
            .def("__radd__", [](const Variable& b, float a) { return a + b; })
            .def("__rsub__", [](const Variable& b, float a) { return a - b; })
            .def("__rmul__", [](const Variable& b, float a) { return a * b; })
            .def("__rtruediv__", [](const Variable& b, float a) { return a / b; })
            .def("__neg__", [](const Variable& x) { return -x; });

    nb::class_<PyVkContext>(m, "VkContext");

    nb::class_<DressiAD> ad(m, "DressiAD");
    nb::enum_<DressiAD::PackingMode>(ad, "PackingMode")
            .value("Naive", DressiAD::PackingMode::Naive)
            .value("RSP", DressiAD::PackingMode::RSP);
    ad.def(nb::init<>())
            .def("__init__",
                 [](DressiAD* self, const PyVkContext& ctx) {
                     new (self) DressiAD(ctx.ptr);
                 },
                 nb::arg("ctx"))
            .def_static("create_context",
                        [](bool debug) {
                            nb::gil_scoped_release rel;
                            return PyVkContext{DressiAD::createContext(debug)};
                        },
                        nb::arg("debug") = false)
            .def("set_loss_var", &DressiAD::setLossVar)
            // The optimizer callback runs on the C++ side during the first
            // exec_step (build time only); nanobind re-acquires the GIL
            .def("set_optimizer", &DressiAD::setOptimizer)
            .def("add_update", &DressiAD::addUpdate)
            .def("set_packing_mode", &DressiAD::setPackingMode)
            .def("set_rebuild_counts", &DressiAD::setRebuildCounts)
            .def("mark_output", &DressiAD::markOutput)
            .def("set_grad_outputs_enabled", &DressiAD::setGradOutputsEnabled)
            .def("input_grads", &DressiAD::inputGrads)
            .def("get_stage_count", &DressiAD::getStageCount)
            .def("get_substage_count", &DressiAD::getSubStageCount)
            .def("exec_step", &DressiAD::execStep,
                 nb::call_guard<nb::gil_scoped_release>())
            .def("send_img",
                 [](DressiAD& self, const Variable& var, const NdArrayIn& arr) {
                     // Borrowed view: the ndarray outlives the call
                     const CpuImageView view(arr.data(),
                                             uint32_t(arr.shape(1)),
                                             uint32_t(arr.shape(0)),
                                             uint32_t(arr.shape(2)));
                     nb::gil_scoped_release rel;
                     self.sendImg(var, view);
                 })
            .def("send_imgs",
                 [](DressiAD& self,
                    const std::vector<std::pair<Variable, NdArrayIn>>& items) {
                     std::vector<std::pair<Variable, CpuImageView>> views;
                     views.reserve(items.size());
                     for (const auto& [var, arr] : items) {
                         views.emplace_back(
                                 var, CpuImageView(arr.data(),
                                                   uint32_t(arr.shape(1)),
                                                   uint32_t(arr.shape(0)),
                                                   uint32_t(arr.shape(2))));
                     }
                     nb::gil_scoped_release rel;
                     self.sendImgs(views);
                 })
            .def("recv_img", [](DressiAD& self, const Variable& var) {
                CpuImage img;
                {
                    nb::gil_scoped_release rel;
                    img = self.recvImg(var);
                }
                return ToNdArray(std::move(img));
            })
            .def("recv_imgs_stacked",
                 [](DressiAD& self, const Variables& vars) {
                     CpuImage img;
                     {
                         nb::gil_scoped_release rel;
                         img = self.recvImgsStacked(vars);
                     }
                     // (n*h, w, c); the torch layer reshapes to the batch
                     return ToNdArray(std::move(img));
                 })
            .def("recv_imgs", [](DressiAD& self, const Variables& vars) {
                std::vector<CpuImage> imgs;
                {
                    nb::gil_scoped_release rel;
                    imgs = self.recvImgs(vars);
                }
                std::vector<nb::ndarray<nb::numpy, float>> arrs;
                arrs.reserve(imgs.size());
                for (auto& img : imgs) {
                    arrs.push_back(ToNdArray(std::move(img)));
                }
                return arrs;
            });

    // ------------------------------ namespace F ------------------------------
    nb::module_ f = m.def_submodule("f", "Dressi op library (namespace F)");
    f.def("constant", &F::Float);
    f.def("vec2", nb::overload_cast<float, float>(&F::Vec2));
    f.def("vec3", nb::overload_cast<float, float, float>(&F::Vec3));
    f.def("vec4", nb::overload_cast<float, float, float, float>(&F::Vec4));
    f.def("vec2v", nb::overload_cast<const Variable&, const Variable&>(&F::Vec2));
    f.def("vec3v", nb::overload_cast<const Variable&, const Variable&,
                                     const Variable&>(&F::Vec3));
    f.def("vec4v",
          nb::overload_cast<const Variable&, const Variable&, const Variable&,
                            const Variable&>(&F::Vec4));
    f.def("vec3_xy_z",
          nb::overload_cast<const Variable&, const Variable&>(&F::Vec3));
    f.def("vec4_xyz_w",
          nb::overload_cast<const Variable&, const Variable&>(&F::Vec4));
    f.def("get_x", &F::GetX);
    f.def("get_y", &F::GetY);
    f.def("get_z", &F::GetZ);
    f.def("get_w", &F::GetW);
    f.def("abs", &F::Abs);
    f.def("sqrt", &F::Sqrt);
    f.def("exp", &F::Exp);
    f.def("pow", &F::Pow);
    f.def("sin", &F::Sin);
    f.def("floor", &F::Floor);
    f.def("fract", &F::Fract);
    f.def("sigmoid", &F::Sigmoid);
    f.def("min", &F::Min);
    f.def("max", &F::Max);
    f.def("clamp", &F::Clamp);
    f.def("step", &F::Step);
    f.def("less", &F::Less);
    f.def("greater", &F::Greater);
    f.def("dot", &F::Dot);
    f.def("comp_sum", &F::CompSum);
    f.def("sum", &F::Sum);
    f.def("mean", &F::Mean);
    f.def("sum_pixel_wise", &F::SumPixelWise);
    f.def("broadcast", [](const Variable& x, uint32_t w, uint32_t h) {
        return F::Broadcast(x, ImgSize{w, h});
    });
    f.def("stop_gradient", &F::StopGradient);
    f.def("screen_coord", [](uint32_t w, uint32_t h) {
        return F::ScreenCoord(ImgSize{w, h});
    });
    f.def("rasterize",
          [](const Variable& clip, const Variable& attr, const Variable& faces,
             uint32_t w, uint32_t h) {
              return F::Rasterize(clip, attr, faces, ImgSize{w, h});
          });
    f.def("rasterize_face_id",
          [](const Variable& clip, const Variable& attr, const Variable& faces,
             uint32_t w, uint32_t h) {
              return F::RasterizeFaceId(clip, attr, faces, ImgSize{w, h});
          });
    f.def("rasterize_soft",
          [](const Variable& clip_soft, const Variable& face_id,
             const Variable& faces_soft, const Variable& clip_hard_tex,
             const Variable& faces_tex, const Variable& vtx_faces_tex,
             uint32_t w, uint32_t h, float radius_px) {
              return F::RasterizeSoft(clip_soft, face_id, faces_soft,
                                      clip_hard_tex, faces_tex, vtx_faces_tex,
                                      ImgSize{w, h}, radius_px);
          });
    f.def("rasterize_soft_peel",
          [](const Variable& clip_soft, const Variable& face_id,
             const Variable& faces_soft, const Variable& clip_hard_tex,
             const Variable& faces_tex, const Variable& vtx_faces_tex,
             const Variable& prev_shifted_depth, uint32_t w, uint32_t h,
             float radius_px) {
              return F::RasterizeSoft(clip_soft, face_id, faces_soft,
                                      clip_hard_tex, faces_tex, vtx_faces_tex,
                                      prev_shifted_depth, ImgSize{w, h},
                                      radius_px);
          });
    f.def("anti_alias", &F::AntiAlias, nb::arg("img"), nb::arg("tri_id"),
          nb::arg("vtx_clip"), nb::arg("faces"), nb::arg("vtx_faces_tex"),
          nb::arg("seed"), nb::arg("n_samples") = 8);
    f.def("lookup_faces", &F::LookupFaces);
    f.def("face_fetch", &F::FaceFetch, nb::arg("tri_attr"), nb::arg("idx_img"),
          nb::arg("tri_prj_0"), nb::arg("tri_prj_1"), nb::arg("tri_prj_2"),
          nb::arg("seed"), nb::arg("radius_px"), nb::arg("n_samples") = 4);
    f.def("texture", &F::Texture);
    f.def("soft_clip",
          [](const Variable& clip_hard_tex, const Variable& faces_tex,
             uint32_t w, uint32_t h, float radius_px) {
              return F::SoftClip(clip_hard_tex, faces_tex, ImgSize{w, h},
                                 radius_px);
          });
    f.def("set_frag_depth", &F::SetFragDepth);
    f.def("peel_depth", &F::PeelDepth);
    f.def("sum_all", &F::SumAll);
    f.def("pixel_fetch", &F::PixelFetch);
    f.def("tile_fetch", &F::TileFetch);
    f.def("vertex_neighbor_mean", &F::VertexNeighborMean);
    f.def("normal_consistency_face_term", &F::NormalConsistencyFaceTerm);
    f.def("normal_consistency_vertex_grad", &F::NormalConsistencyVertexGrad);

    // ------------------------ CPU mesh-topology builders ---------------------
    m.def("vertex_faces_tex", [](const NdArrayIn& faces, uint32_t n_verts) {
        return ToNdArray(VertexFacesTex(ToCpuImage(faces), n_verts));
    });
    m.def("vertex_neighbors_tex",
          [](const NdArrayIn& faces, uint32_t n_verts) {
              return ToNdArray(VertexNeighborsTex(ToCpuImage(faces), n_verts));
          });
    m.def("face_neighbors_tex", [](const NdArrayIn& faces) {
        return ToNdArray(FaceNeighborsTex(ToCpuImage(faces)));
    });
    m.def("build_soft_geometry",
          [](const NdArrayIn& hard_clip, const NdArrayIn& faces, uint32_t w,
             uint32_t h, float radius_px) {
              SoftGeometry soft =
                      BuildSoftGeometry(ToCpuImage(hard_clip),
                                        ToCpuImage(faces), ImgSize{w, h},
                                        radius_px);
              return nb::make_tuple(ToNdArray(std::move(soft.clip)),
                                    ToNdArray(std::move(soft.face_id)),
                                    ToNdArray(std::move(soft.faces)));
          });
    m.def("uv_as_clip", [](const NdArrayIn& uv) {
        return ToNdArray(UvAsClip(ToCpuImage(uv)));
    });
    m.def("screen_attr", [](const NdArrayIn& clip, uint32_t w, uint32_t h) {
        return ToNdArray(ScreenAttr(ToCpuImage(clip), ImgSize{w, h}));
    });
}
