"""nvdiffrast-compatible ops backed by shape-keyed Dressi engines.

Performance model:
- Batching: a call with N batch items builds ONE engine whose graph holds
  N unrolled per-item subgraphs (static topology leaves shared), so a
  minibatch costs one execStep, one batched upload submit, and one
  batched download submit.
- Forward/backward split: forward runs a gradient-free INFER engine;
  backward runs the TRAIN engine (whose exec recomputes the forward
  internally). This keeps expensive backward stages (e.g. the per-vertex
  AntiAlias gather) out of the forward pass entirely.
- Per-item leaves are named "<base>#<i>"; Engine.run() flushes queued
  uploads as one staging submit.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import _C
from .context import RasterizeVulkanContext
from .convert import (
    as_hwc,
    check_resolution,
    clip_gl_to_vk,
    clip_grad_vk_to_gl,
    register_cpu_copy,
    split_batch,
    to_cpu_f32,
    vertex_array,
)
from .engine import Engine, upload_token
from .provenance import InterpInfo, RastInfo, attach, lookup

f = _C.f

# Per-corner barycentric constants (u, v) = (1,0), (0,1), (0,0): rasterizing
# them as unwelded vertex attributes yields nvdiffrast's perspective-correct
# barycentrics per pixel.
_CORNER_UV = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

_VTYPE_BY_CHANNELS = {
    1: _C.VType.FLOAT,
    2: _C.VType.VEC2,
    3: _C.VType.VEC3,
    4: _C.VType.VEC4,
}

_ZERO_SEED = np.zeros((1, 1, 1), np.float32)


def _vtype_of(channels: int, what: str):
    vt = _VTYPE_BY_CHANNELS.get(channels)
    if vt is None:
        raise ValueError(
            f"dressi.torch: {what} must have 1..4 channels, got {channels}")
    return vt


def _needs_grad(*tensors) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)


def _item(name: str, i: int) -> str:
    return f"{name}#{i}"


def _stack_batch(outs: list) -> torch.Tensor:
    """torch.stack copies; a single item just gets a batch-dim view."""
    return outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs, 0)


def _image_batch(t: torch.Tensor, name: str) -> torch.Tensor:
    """Accept [H, W, C] or [N, H, W, C]; return [N, H, W, C]."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 4:
        return t
    raise ValueError(
        f"{name} must be [H, W, C] or [N, H, W, C], got {tuple(t.shape)}")


def _to_device(cpu_out: torch.Tensor, device) -> torch.Tensor:
    out = cpu_out.to(device)
    if out is not cpu_out:
        register_cpu_copy(out, cpu_out)
    return out


# ------------------------------- rasterize ---------------------------------


def _build_rasterize_engine(ctx: RasterizeVulkanContext, md, h: int, w: int,
                            n_items: int) -> Engine:
    # Unwelding happens CPU-side (pos gathered per corner before upload):
    # an in-graph SoftClip(radius=0) unweld was measured to degrade AA
    # silhouette convergence (IoU 0.97 -> 0.94) and was reverted
    eng = Engine(ctx._ctx)
    faces_seq = eng.leaf("faces_seq", _C.VType.IVEC3, md.n_faces, 1)
    ones3f = eng.leaf("ones3f", _C.VType.FLOAT, md.n_faces * 3, 1)
    outs = []
    for i in range(n_items):
        clip3 = eng.leaf(_item("clip3", i), _C.VType.VEC4,
                         md.n_faces * 3, 1)
        attr4 = eng.leaf(_item("attr4", i), _C.VType.VEC4,
                         md.n_faces * 3, 1)
        g = f.rasterize(clip3, attr4, faces_seq, w, h)  # (u, v, z_gl, w)
        tid = f.rasterize_face_id(clip3, ones3f, faces_seq, w, h)
        mask = f.greater(tid, f.constant(0.5))
        zw = f.get_z(g) / (f.get_w(g) + (1.0 - mask))  # guarded z/w (bg 0)
        outs.append(f.vec4v(f.get_x(g), f.get_y(g), zw, tid))
    eng.finalize(outs)
    return eng


def rasterize(ctx: RasterizeVulkanContext, pos: torch.Tensor,
              tri: torch.Tensor, resolution, ranges=None, grad_db: bool = False):
    """Drop-in nvdiffrast rasterize: returns (rast, rast_db).

    rast[..., :] = (u, v, z/w, triangle_id + 1); 0 = background. v1 is not
    differentiable w.r.t. pos (geometry gradients come from antialias /
    rasterize_soft, matching the silhouette workflow); rast_db is zeros.
    """
    if ranges is not None:
        raise NotImplementedError("dressi.torch: range mode is not supported")
    del grad_db
    h, w = check_resolution(resolution)
    in_device = pos.device
    pos_b = split_batch(to_cpu_f32(pos, "pos"))
    md = ctx.mesh(tri)
    if pos_b.shape[2] != 4:
        raise ValueError(f"pos must have 4 components, got {pos_b.shape[2]}")
    n = pos_b.shape[0]

    key = ("rasterize", h, w, md.n_faces, md.n_verts, n)
    eng: Engine = ctx.engine(
        key, lambda: _build_rasterize_engine(ctx, md, h, w, n))

    with ctx.lock:
        eng.upload("faces_seq", md.faces_seq_f32, token=("static", id(md)),
                   ref=md)
        eng.upload("ones3f", np.ones((1, md.n_faces * 3, 1), np.float32),
                   token=("static", id(md)), ref=md)
        # Vectorized prep: one unweld gather / concat for the whole batch
        clip3_all = clip_gl_to_vk(pos_b)[:, md.unweld]  # [N, 3F, 4]
        corner_uv = torch.from_numpy(
            np.tile(_CORNER_UV, (md.n_faces, 1))).expand(n, -1, -1)
        attr4_all = torch.cat(
            [corner_uv, pos_b[:, md.unweld][:, :, 2:4]], dim=2)
        tok = upload_token(pos_b)
        for i in range(n):
            eng.upload(_item("clip3", i), vertex_array(clip3_all[i]),
                       token=("clip3", i, tok), ref=pos_b)
            eng.upload(_item("attr4", i), vertex_array(attr4_all[i]),
                       token=("attr4", i, tok), ref=pos_b)
        eng.run()
        rast_cpu = torch.from_numpy(
            eng.read_outputs_stacked()).reshape(n, h, w, 4)
    rast = _to_device(rast_cpu, in_device)
    attach(rast, RastInfo(pos=pos_b, tri=tri, resolution=(h, w),
                          extra={"ctx": ctx}))
    return rast, torch.zeros_like(rast)


# ------------------------------ interpolate --------------------------------


def _build_interpolate_engine(ctx, md, h: int, w: int, c_attr: int,
                              n_items: int, shared_attr: bool,
                              train: bool) -> Engine:
    eng = Engine(ctx._ctx)
    faces_l = eng.leaf("faces_f", _C.VType.VEC3, md.n_faces, 1)
    vft_l = eng.leaf("vft", _C.VType.FLOAT, md.n_verts, md.max_deg)
    seed_l = eng.leaf("ffseed", _C.VType.FLOAT, 1, 1)
    attr_vt = _vtype_of(c_attr, "attr")
    if shared_attr:
        attr_shared = eng.leaf("attr", attr_vt, md.n_verts, 1)
    outs, grad_leaves = [], []
    for i in range(n_items):
        attr_l = (attr_shared if shared_attr
                  else eng.leaf(_item("attr", i), attr_vt, md.n_verts, 1))
        rast_l = eng.leaf(_item("rast", i), _C.VType.VEC4, w, h)
        clip_l = eng.leaf(_item("clip", i), _C.VType.VEC4, md.n_verts, 1)
        u, v = f.get_x(rast_l), f.get_y(rast_l)
        tid = f.get_w(rast_l)
        prj = [f.lookup_faces(clip_l, faces_l, vft_l, c) for c in range(3)]
        # Exact FaceFetch backward (n_samples=0): the true VJP for interior
        # support, chained with LookupFaces' exact vertex scatter
        pix = [
            f.face_fetch(f.lookup_faces(attr_l, faces_l, vft_l, c), tid,
                         prj[0], prj[1], prj[2], seed_l, 1.0, 0)
            for c in range(3)
        ]
        outs.append(u * pix[0] + v * pix[1] + (1.0 - u - v) * pix[2])
        if train and not shared_attr:
            grad_leaves.append(_item("attr", i))
    if train and shared_attr:
        grad_leaves = ["attr"]  # per-item contributions accumulate in-graph
    eng.finalize(outs, grad_leaves=grad_leaves)
    return eng


def _interpolate_uploads(eng: Engine, md, rinfo: RastInfo, rast_cpu, attr_b,
                         shared_attr: bool):
    eng.upload("faces_f", md.faces_f32, token=("static", id(md)), ref=md)
    eng.upload("vft", md.vtx_faces_tex, token=("vft", id(md)), ref=md)
    eng.upload("ffseed", _ZERO_SEED, token="zero")
    if shared_attr:
        eng.upload("attr", vertex_array(attr_b),
                   token=("attr", upload_token(attr_b)), ref=attr_b)
    clip_vk = clip_gl_to_vk(rinfo.pos)
    for i in range(rast_cpu.shape[0]):
        eng.upload(_item("clip", i),
                   vertex_array(clip_vk[i if clip_vk.shape[0] > 1 else 0]),
                   token=("clip", upload_token(rinfo.pos), i), ref=rinfo.pos)
        eng.upload(_item("rast", i), as_hwc(rast_cpu[i]),
                   token=("rast", upload_token(rast_cpu), i), ref=rast_cpu)
        if not shared_attr:
            eng.upload(_item("attr", i), vertex_array(attr_b[i]),
                       token=("attr", upload_token(attr_b), i), ref=attr_b)


class _InterpolateFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, attr, rast, vctx, md, rinfo, train):
        h, w = rinfo.resolution
        attr_b = to_cpu_f32(attr, "attr")
        rast_cpu = to_cpu_f32(rast, "rast")
        c_attr = attr_b.shape[-1]
        n = rast_cpu.shape[0]
        shared_attr = attr_b.dim() == 2

        # Forward runs the gradient-free engine; the train engine only
        # ever executes inside backward()
        key = ("interpolate", h, w, md.n_faces, md.n_verts, c_attr, n,
               shared_attr, False)
        eng: Engine = vctx.engine(
            key, lambda: _build_interpolate_engine(vctx, md, h, w, c_attr, n,
                                                   shared_attr, False))
        with vctx.lock:
            _interpolate_uploads(eng, md, rinfo, rast_cpu, attr_b,
                                 shared_attr)
            eng.run()
            out = torch.from_numpy(
                eng.read_outputs_stacked()).reshape(n, h, w, -1)

        fctx.dressi = (vctx, md, rinfo, n, shared_attr, c_attr, (h, w))
        fctx.save_for_backward(attr, rast)
        return out

    @staticmethod
    def backward(fctx, grad_out):
        vctx, md, rinfo, n, shared_attr, c_attr, (h, w) = fctx.dressi
        attr, rast = fctx.saved_tensors
        attr_b = to_cpu_f32(attr, "attr")
        rast_cpu = to_cpu_f32(rast, "rast")
        g = to_cpu_f32(grad_out, "grad_out")

        key = ("interpolate", h, w, md.n_faces, md.n_verts, c_attr, n,
               shared_attr, True)
        eng: Engine = vctx.engine(
            key, lambda: _build_interpolate_engine(vctx, md, h, w, c_attr, n,
                                                   shared_attr, True))
        with vctx.lock:
            _interpolate_uploads(eng, md, rinfo, rast_cpu, attr_b,
                                 shared_attr)
            for i in range(n):
                eng.upload(Engine.seed_name(i), as_hwc(g[i]))
            eng.run()
            if shared_attr:
                g_attr = torch.from_numpy(eng.read_grad("attr"))[0]
            else:
                g_attr = torch.from_numpy(eng.read_grads_stacked(
                    [_item("attr", i) for i in range(n)])).reshape(
                        n, md.n_verts, -1)
        return g_attr.to(attr.device), None, None, None, None, None


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Drop-in nvdiffrast interpolate: (pixel_attr, None).

    Differentiable w.r.t. `attr` (exact VJP). Gradients w.r.t. `rast`
    (i.e. barycentric derivatives) are not produced in v1; `diff_attrs`
    (attribute pixel derivatives) is unsupported.
    """
    if rast_db is not None or diff_attrs is not None:
        raise NotImplementedError(
            "dressi.torch.interpolate: rast_db/diff_attrs are not supported")
    rinfo: RastInfo = lookup(rast, "rasterize")
    vctx = rinfo.extra["ctx"]  # the context travels with the provenance
    md = vctx.mesh(tri)
    train = _needs_grad(attr)
    out = _InterpolateFn.apply(attr, rast, vctx, md, rinfo, train)
    out_dev = out.to(attr.device)
    if out_dev is not out:
        register_cpu_copy(out_dev, out.detach())
    attach(out_dev, InterpInfo(attr=attr, rast_info=rinfo))
    return out_dev, None


# -------------------------------- texture ----------------------------------


def _build_texture_engine(ctx, h, w, th, tw, c_tex, mesh_dims, n_items,
                          train) -> Engine:
    eng = Engine(ctx._ctx)
    tex_l = eng.leaf("tex", _vtype_of(c_tex, "tex"), tw, th)
    outs = []
    for i in range(n_items):
        uv_l = eng.leaf(_item("uv", i), _C.VType.VEC2, w, h)
        if train:
            n_verts, n_faces = mesh_dims
            uvclip_l = eng.leaf(_item("uv_clip", i), _C.VType.VEC4,
                                n_verts, 1)
            sattr_l = eng.leaf(_item("screen_attr", i), _C.VType.VEC4,
                               n_verts, 1)
            faces_l = (eng.leaves.get("faces_i")
                       or eng.leaf("faces_i", _C.VType.IVEC3, n_faces, 1))
            # Inverse-UV table: the mesh rasterized in UV space carrying
            # its screen positions (F::Texture's gradient gather geometry)
            inv_uv = f.rasterize(uvclip_l, sattr_l, faces_l, tw, th)
        else:
            # Forward never reads inv_uv (it only feeds the backward
            # gather); an unreferenced dummy keeps the graph pure sampling
            inv_uv = _C.Variable(_C.VType.VEC4, tw, th)
        outs.append(f.texture(tex_l, uv_l, inv_uv))
    eng.finalize(outs, grad_leaves=["tex"] if train else [])
    return eng


def _texture_uploads(eng, tex_b, uv_cpu, md, rinfo, iinfo, train, res):
    h, w = res
    eng.upload("tex", as_hwc(tex_b), token=("tex", upload_token(tex_b)),
               ref=tex_b)
    for i in range(uv_cpu.shape[0]):
        eng.upload(_item("uv", i), as_hwc(uv_cpu[i]),
                   token=("uv", upload_token(uv_cpu), i), ref=uv_cpu)
        if train:
            vtx_uv = to_cpu_f32(iinfo.attr, "uv attr")
            if vtx_uv.dim() == 3:
                vtx_uv = vtx_uv[i if vtx_uv.shape[0] > 1 else 0]
            eng.upload(_item("uv_clip", i),
                       _C.uv_as_clip(vertex_array(vtx_uv)),
                       token=("uv_clip", upload_token(iinfo.attr), i),
                       ref=iinfo.attr)
            pos_n = rinfo.pos[i if rinfo.pos.shape[0] > 1 else 0]
            clip_vk = clip_gl_to_vk(pos_n)
            eng.upload(_item("screen_attr", i),
                       _C.screen_attr(vertex_array(clip_vk), w, h),
                       token=("sattr", upload_token(rinfo.pos), i),
                       ref=rinfo.pos)
            eng.upload("faces_i", md.faces_f32, token=("static", id(md)),
                       ref=md)


class _TextureFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, tex, uv, vctx, iinfo, train):
        tex_b = to_cpu_f32(tex, "tex")
        uv_cpu = to_cpu_f32(uv, "uv")
        if tex_b.dim() == 4:
            if tex_b.shape[0] != 1:
                raise NotImplementedError(
                    "dressi.torch.texture: per-item texture batches are not "
                    "supported in v1 (pass [H, W, C] or [1, H, W, C])")
            tex_b = tex_b[0]
        th, tw, c_tex = tex_b.shape
        n, h, w = uv_cpu.shape[0], uv_cpu.shape[1], uv_cpu.shape[2]

        md = rinfo = None
        mesh_dims = ()
        if train:
            rinfo = iinfo.rast_info
            md = vctx.mesh(rinfo.tri)
            mesh_dims = (md.n_verts, md.n_faces)
        key = ("texture", h, w, th, tw, c_tex, mesh_dims, n, False)
        eng: Engine = vctx.engine(
            key, lambda: _build_texture_engine(vctx, h, w, th, tw, c_tex,
                                               mesh_dims, n, False))
        with vctx.lock:
            _texture_uploads(eng, tex_b, uv_cpu, md, rinfo, iinfo, False,
                             (h, w))
            eng.run()
            out = torch.from_numpy(
                eng.read_outputs_stacked()).reshape(n, h, w, -1)
        fctx.dressi = (vctx, md, rinfo, iinfo, n, mesh_dims,
                       (h, w, th, tw, c_tex))
        fctx.save_for_backward(tex, uv)
        return out

    @staticmethod
    def backward(fctx, grad_out):
        vctx, md, rinfo, iinfo, n, mesh_dims, dims = fctx.dressi
        h, w, th, tw, c_tex = dims
        tex, uv = fctx.saved_tensors
        tex_b = to_cpu_f32(tex, "tex")
        if tex_b.dim() == 4:
            tex_b = tex_b[0]
        uv_cpu = to_cpu_f32(uv, "uv")
        g = to_cpu_f32(grad_out, "grad_out")

        key = ("texture", h, w, th, tw, c_tex, mesh_dims, n, True)
        eng: Engine = vctx.engine(
            key, lambda: _build_texture_engine(vctx, h, w, th, tw, c_tex,
                                               mesh_dims, n, True))
        with vctx.lock:
            _texture_uploads(eng, tex_b, uv_cpu, md, rinfo, iinfo, True,
                             (h, w))
            for i in range(n):
                eng.upload(Engine.seed_name(i), as_hwc(g[i]))
            eng.run()
            g_tex = torch.from_numpy(eng.read_grad("tex"))
        if tex.dim() == 4:
            g_tex = g_tex.unsqueeze(0)
        return g_tex.to(tex.device), None, None, None, None


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode="auto", boundary_mode="wrap", max_mip_level=None):
    """Drop-in nvdiffrast texture (nearest filtering only in v1).

    Differentiable w.r.t. `tex` (through the inverse-UV table built from
    `uv`'s provenance); not differentiable w.r.t. `uv`. Mipmaps and linear
    filtering are unsupported.
    """
    if uv_da is not None or mip_level_bias is not None or mip is not None:
        raise NotImplementedError("dressi.torch.texture: mipmaps unsupported")
    if filter_mode not in ("auto", "nearest"):
        raise NotImplementedError(
            f"dressi.torch.texture: filter_mode='{filter_mode}' unsupported "
            "(v1 is nearest-only)")
    del boundary_mode, max_mip_level
    train = _needs_grad(tex)
    try:
        iinfo = lookup(uv, "interpolate")
        vctx = iinfo.rast_info.extra["ctx"]
    except RuntimeError:
        raise RuntimeError(
            "dressi.torch.texture: `uv` must come from "
            "dressi.torch.interpolate in v1 (context discovery)")
    if uv.dim() == 3:
        uv = uv.unsqueeze(0)
    out = _TextureFn.apply(tex, uv, vctx, iinfo, train)
    return out.to(tex.device)


# ------------------------------- antialias ---------------------------------


def _build_antialias_engine(ctx, md, h, w, c_img, n_items, grad_mask,
                            n_samples) -> Engine:
    eng = Engine(ctx._ctx)
    faces_l = eng.leaf("faces_f", _C.VType.VEC3, md.n_faces, 1)
    vft_l = eng.leaf("vft", _C.VType.FLOAT, md.n_verts, md.max_deg)
    outs, grads = [], []
    for i in range(n_items):
        img_l = eng.leaf(_item("img", i), _vtype_of(c_img, "color"), w, h)
        tid_l = eng.leaf(_item("tid", i), _C.VType.FLOAT, w, h)
        clip_l = eng.leaf(_item("clip", i), _C.VType.VEC4, md.n_verts, 1)
        # Per-item jitter seed: a shared seed would give every batch item
        # the same stochastic-backward sample pattern (correlated noise
        # measurably hurts convergence)
        seed_l = eng.leaf(_item("aaseed", i), _C.VType.FLOAT, 1, 1)
        outs.append(f.anti_alias(img_l, tid_l, clip_l, faces_l, vft_l,
                                 seed_l, n_samples))
        if grad_mask[0]:
            grads.append(_item("img", i))
        if grad_mask[1]:
            grads.append(_item("clip", i))
    eng.finalize(outs, grad_leaves=grads)
    return eng


def _antialias_uploads(eng, md, color_cpu, rast_cpu, pos_b, seed_value):
    n = color_cpu.shape[0]
    eng.upload("faces_f", md.faces_f32, token=("static", id(md)), ref=md)
    eng.upload("vft", md.vtx_faces_tex, token=("vft", id(md)), ref=md)
    clip_vk = clip_gl_to_vk(pos_b)
    for i in range(n):
        eng.upload(_item("aaseed", i),
                   np.full((1, 1, 1), float(seed_value * n + i), np.float32))
        eng.upload(_item("img", i), as_hwc(color_cpu[i]),
                   token=("img", upload_token(color_cpu), i), ref=color_cpu)
        eng.upload(_item("tid", i), as_hwc(rast_cpu[i, :, :, 3:4]),
                   token=("tid", upload_token(rast_cpu), i), ref=rast_cpu)
        eng.upload(_item("clip", i),
                   vertex_array(clip_vk[i if clip_vk.shape[0] > 1 else 0]),
                   token=("clip", upload_token(pos_b), i), ref=pos_b)


class _AntiAliasFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, color, rast, pos, vctx, md, grad_mask, n_samples,
                pos_gradient_boost):
        color_cpu = _image_batch(to_cpu_f32(color, "color"), "color")
        rast_cpu = to_cpu_f32(rast, "rast")
        pos_b = split_batch(to_cpu_f32(pos, "pos"))
        n, h, w = color_cpu.shape[0], color_cpu.shape[1], color_cpu.shape[2]
        c_img = color_cpu.shape[3]

        key = ("antialias", h, w, md.n_faces, md.n_verts, c_img, n,
               (False, False), n_samples)
        eng: Engine = vctx.engine(
            key, lambda: _build_antialias_engine(vctx, md, h, w, c_img, n,
                                                 (False, False), n_samples))
        with vctx.lock:
            _antialias_uploads(eng, md, color_cpu, rast_cpu, pos_b, 0)
            eng.run()
            out = torch.from_numpy(
                eng.read_outputs_stacked()).reshape(n, h, w, -1)
        fctx.dressi = (vctx, md, grad_mask, n, pos_gradient_boost,
                       n_samples, (h, w, c_img))
        fctx.save_for_backward(color, rast, pos)
        return out

    @staticmethod
    def backward(fctx, grad_out):
        vctx, md, grad_mask, n, boost, n_samples, dims = fctx.dressi
        h, w, c_img = dims
        color, rast, pos = fctx.saved_tensors
        color_cpu = _image_batch(to_cpu_f32(color, "color"), "color")
        rast_cpu = to_cpu_f32(rast, "rast")
        pos_b = split_batch(to_cpu_f32(pos, "pos"))
        g = _image_batch(to_cpu_f32(grad_out, "grad_out"), "grad_out")

        key = ("antialias", h, w, md.n_faces, md.n_verts, c_img, n,
               grad_mask, n_samples)
        eng: Engine = vctx.engine(
            key, lambda: _build_antialias_engine(vctx, md, h, w, c_img, n,
                                                 grad_mask, n_samples))
        with vctx.lock:
            # Advance the stochastic-backward jitter each backward pass
            eng.iter_seed = getattr(eng, "iter_seed", 0) + 1
            _antialias_uploads(eng, md, color_cpu, rast_cpu, pos_b,
                               eng.iter_seed)
            for i in range(n):
                eng.upload(Engine.seed_name(i), as_hwc(g[i]))
            eng.run()
            g_color = g_pos = None
            if grad_mask[0]:
                g_color = torch.from_numpy(eng.read_grads_stacked(
                    [_item("img", i) for i in range(n)])).reshape(
                        n, h, w, -1)
            if grad_mask[1]:
                g_clip = torch.from_numpy(eng.read_grads_stacked(
                    [_item("clip", i) for i in range(n)])).reshape(n, -1, 4)
                g_pos = clip_grad_vk_to_gl(g_clip) * boost

        out_gc = None
        if g_color is not None:
            out_gc = g_color[0] if color.dim() == 3 else g_color
            out_gc = out_gc.to(color.device)
        out_gp = None
        if g_pos is not None:
            out_gp = g_pos.sum(dim=0) if pos.dim() == 2 else g_pos
            out_gp = out_gp.to(pos.device)
        return (out_gc, None, out_gp, None, None, None, None, None)


def antialias(color, rast, pos, tri, topology_hash=None,
              pos_gradient_boost=1.0, n_samples=8):
    """Drop-in nvdiffrast antialias (Dr.Hair screen-space AA, Eq.3-5).

    Differentiable w.r.t. `color` and `pos`. `n_samples` selects the
    vertex-gradient backward: 0 = exact bbox scan, > 0 = stochastic
    (n jittered edge samples per face per backward, the fast default).
    `topology_hash` is accepted and ignored.
    """
    del topology_hash
    rinfo: RastInfo = lookup(rast, "rasterize")
    vctx = rinfo.extra["ctx"]
    md = vctx.mesh(tri)
    grad_mask = (_needs_grad(color), _needs_grad(pos))
    out = _AntiAliasFn.apply(color, rast, pos, vctx, md, grad_mask,
                             n_samples, pos_gradient_boost)
    return out.to(color.device)


# ------------------------------ rasterize_soft -----------------------------


def _build_rasterize_soft_engine(ctx, md, h, w, radius_px, peels, n_items,
                                 train) -> Engine:
    eng = Engine(ctx._ctx)
    face_id_l = eng.leaf("face_id", _C.VType.FLOAT, md.n_faces * 3, 1)
    faces_soft_l = eng.leaf("faces_soft", _C.VType.IVEC3, md.n_faces, 1)
    faces_l = eng.leaf("faces_f", _C.VType.VEC3, md.n_faces, 1)
    vft_l = eng.leaf("vft", _C.VType.FLOAT, md.n_verts, md.max_deg)
    zero, one = f.constant(0.0), f.constant(1.0)
    outs, grads = [], []
    for i in range(n_items):
        clip_l = eng.leaf(_item("clip", i), _C.VType.VEC4, md.n_verts, 1)
        soft_clip = f.soft_clip(clip_l, faces_l, w, h, radius_px)
        prev_shift = None
        for peel in range(peels):
            if peel == 0:
                rs = f.rasterize_soft(soft_clip, face_id_l, faces_soft_l,
                                      clip_l, faces_l, vft_l, w, h,
                                      radius_px)
            else:
                rs = f.rasterize_soft_peel(soft_clip, face_id_l,
                                           faces_soft_l, clip_l, faces_l,
                                           vft_l, prev_shift, w, h,
                                           radius_px)
            outs.append(rs)
            if peel + 1 < peels:
                # Next peel's threshold: this layer's Eq.3 shifted depth
                # rebuilt from the output channels (0 at background)
                dist_c, cov = f.get_x(rs), f.get_w(rs)
                ge = f.step(zero, dist_c)
                prev_shift = cov * (
                    ge * (f.clamp(f.get_z(rs), zero, one) * 0.5)
                    + (1.0 - ge) * (f.clamp(dist_c * (-1.0 / radius_px),
                                            zero, one) * 0.5 + 0.5))
        if train:
            grads.append(_item("clip", i))
    eng.finalize(outs, grad_leaves=grads)
    return eng


def _rasterize_soft_uploads(eng, md, pos_b):
    eng.upload("face_id", md.corner_face_id, token=("static", id(md)), ref=md)
    eng.upload("faces_soft", md.faces_seq_f32, token=("seq", id(md)), ref=md)
    eng.upload("faces_f", md.faces_f32, token=("faces", id(md)), ref=md)
    eng.upload("vft", md.vtx_faces_tex, token=("vft", id(md)), ref=md)
    clip_vk = clip_gl_to_vk(pos_b)
    for i in range(pos_b.shape[0]):
        eng.upload(_item("clip", i), vertex_array(clip_vk[i]),
                   token=("clip", upload_token(pos_b), i), ref=pos_b)


class _RasterizeSoftFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, pos, vctx, md, resolution, radius_px, peels, train):
        h, w = resolution
        pos_b = split_batch(to_cpu_f32(pos, "pos"))
        n = pos_b.shape[0]
        key = ("rasterize_soft", h, w, md.n_faces, md.n_verts, radius_px,
               peels, n, False)
        eng: Engine = vctx.engine(
            key, lambda: _build_rasterize_soft_engine(vctx, md, h, w,
                                                      radius_px, peels, n,
                                                      False))
        with vctx.lock:
            _rasterize_soft_uploads(eng, md, pos_b)
            eng.run()
            out = torch.from_numpy(
                eng.read_outputs_stacked()).reshape(n, peels, h, w, 4)
        fctx.dressi = (vctx, md, n, peels, radius_px, (h, w))
        fctx.save_for_backward(pos)
        return out  # [N, K, H, W, 4]

    @staticmethod
    def backward(fctx, grad_out):
        vctx, md, n, peels, radius_px, (h, w) = fctx.dressi
        (pos,) = fctx.saved_tensors
        pos_b = split_batch(to_cpu_f32(pos, "pos"))
        g = to_cpu_f32(grad_out, "grad_out")  # [N, K, H, W, 4]

        key = ("rasterize_soft", h, w, md.n_faces, md.n_verts, radius_px,
               peels, n, True)
        eng: Engine = vctx.engine(
            key, lambda: _build_rasterize_soft_engine(vctx, md, h, w,
                                                      radius_px, peels, n,
                                                      True))
        with vctx.lock:
            _rasterize_soft_uploads(eng, md, pos_b)
            for i in range(n):
                for k in range(peels):
                    eng.upload(Engine.seed_name(i * peels + k),
                               as_hwc(g[i, k]))
            eng.run()
            g_clip = torch.from_numpy(eng.read_grads_stacked(
                [_item("clip", i) for i in range(n)])).reshape(n, -1, 4)
            g_pos = clip_grad_vk_to_gl(g_clip)
        if pos.dim() == 2:
            g_pos = g_pos.sum(dim=0)
        return g_pos.to(pos.device), None, None, None, None, None, None


def rasterize_soft(ctx, pos, tri, resolution, radius_px=3.0, peels=1):
    """HardSoftRas soft rasterization (extension; no nvdiffrast analogue).

    Returns [N, K, H, W, 4] with channels (signed_dist_px, face_id,
    hard_depth, coverage) per depth-peel layer K. Differentiable w.r.t.
    `pos` through the signed-distance channel (exact adjacency-bounded
    gather backward). Compose sigmoid/Eq.6 silhouettes in torch, e.g.
    `cov * torch.sigmoid(dist / sigma)`.
    """
    h, w = check_resolution(resolution)
    if peels < 1:
        raise ValueError("peels must be >= 1")
    md = ctx.mesh(tri)
    train = _needs_grad(pos)
    out = _RasterizeSoftFn.apply(pos, ctx, md, (h, w), float(radius_px),
                                 int(peels), train)
    return out.to(pos.device)
