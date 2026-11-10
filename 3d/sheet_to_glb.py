"""Reconstruct a full 3D GLB from a multi-view character sheet (green screen).

Uses the four orthographic body views on the top row of the sheet
(front, right side, back, left side):
  1. Chroma-key each view -> silhouette, align them on feet + head top.
  2. Visual hull: a voxel is solid if it falls inside the front/back
     silhouette (X-Y plane) and inside the side silhouettes (Z-Y plane).
  3. Marching cubes + Taubin smoothing -> watertight surface.
  4. Colour every vertex by projecting it into the views that face it,
     with a per-view depth test so hidden surfaces don't pick up colour.

Usage: python sheet_to_glb.py archer_sheet.png egyptian_archer_full.glb
"""
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure

HEIGHT_M = 1.80
RES = 480                     # voxels along the height
TOP_ROW = (0, 640)            # y-range of the full-body row
# x-ranges of the panels (found from empty columns between figures)
PANELS = {"front": (0, 354), "right": (354, 596), "back": (596, 869),
          "left": (869, 1081), "front34": (1081, 1308), "back34": (1308, 1536)}
# camera yaw of each view (degrees). world: +x = character's left, +y = up,
# +z = facing direction; camera at (sin a, 0, cos a), image-right axis
# (cos a, 0, -sin a)
YAW = {"front": 0, "right": -90, "back": 180, "left": 90,
       "front34": -45, "back34": -135}


def key(img):
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    green = (g > 110) & (g > r * 1.25) & (g > b * 1.25)
    m = (~green).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 80:
            out[lab == i] = 1
    # remove thin separator lines touching the panel border vertically
    return out.astype(bool)


def despill(img, mask):
    f = img.astype(np.float32)
    f[..., 1] = np.minimum(f[..., 1], np.maximum(f[..., 0], f[..., 2]))
    f = np.clip(f, 0, 255).astype(np.uint8)
    return cv2.inpaint(f, (~mask).astype(np.uint8), 4, cv2.INPAINT_TELEA)


def sample(img, u, v):
    """Bilinear lookup of img at float pixel coords (u, v)."""
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001)
    u0 = u.astype(int); v0 = v.astype(int); fu = (u - u0)[:, None]; fv = (v - v0)[:, None]
    im = img.astype(np.float64)
    return ((im[v0, u0] * (1 - fu) + im[v0, u0 + 1] * fu) * (1 - fv) +
            (im[v0 + 1, u0] * (1 - fu) + im[v0 + 1, u0 + 1] * fu) * fv)


def write_glb(path, pos, nrm, faces, rgb):
    """Minimal glTF 2.0 binary writer: one mesh, vertex colours, matte material."""
    import json
    import struct
    srgb = rgb.astype(np.float32) / 255
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    arrays = [faces.astype(np.uint32).ravel(), pos.astype(np.float32),
              nrm.astype(np.float32), lin.astype(np.float32)]
    blob, views, off = b"", [], 0
    for a in arrays:
        data = a.tobytes()
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
        blob += data + b"\0" * (-len(data) % 4)
        off = len(blob)
    p = pos.astype(np.float32)
    acc = [
        {"bufferView": 0, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
        {"bufferView": 1, "componentType": 5126, "count": len(p), "type": "VEC3",
         "min": p.min(0).tolist(), "max": p.max(0).tolist()},
        {"bufferView": 2, "componentType": 5126, "count": len(p), "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": len(p), "type": "VEC3"},
    ]
    gltf = {
        "asset": {"version": "2.0", "generator": "sheet_to_glb.py"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "EgyptianArcher"}],
        "meshes": [{"name": "EgyptianArcher", "primitives": [{
            "attributes": {"POSITION": 1, "NORMAL": 2, "COLOR_0": 3},
            "indices": 0, "material": 0}]}],
        "materials": [{"name": "skin_cloth", "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
            "roughnessFactor": 0.75}}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views, "accessors": acc,
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


class View:
    def __init__(self, name, sheet):
        x0, x1 = PANELS[name]
        y0, y1 = TOP_ROW
        img = sheet[y0:y1, x0:x1]
        self.name = name
        self.mask = key(img)
        self.mask[:, :3] = False
        self.mask[:, -3:] = False
        self.tex = despill(img, self.mask)
        ys, xs = np.nonzero(self.mask)
        self.bottom = ys.max()
        h = self.mask.shape[0]
        legs = (ys > self.bottom - 0.28 * h) & (ys < self.bottom - 0.08 * h)
        lc = np.median(xs[legs])
        band = self.mask[:, int(lc - 25):int(lc + 25)]
        self.top = np.nonzero(band.any(1))[0].min()
        # horizontal anchor = centre of the head (consistent in every view);
        # take the widest-run row segment around the band 6-12% below head top
        hc = []
        for r in range(int(self.top + 0.05 * (self.bottom - self.top)),
                       int(self.top + 0.11 * (self.bottom - self.top))):
            row = self.mask[r]
            runs = np.split(np.arange(len(row))[row],
                            np.nonzero(np.diff(np.arange(len(row))[row]) > 1)[0] + 1)
            best = min(runs, key=lambda q: abs(q.mean() - lc))
            hc.append(best.mean())
        self.cx = float(np.median(hc)) if "34" not in name else float(lc)
        self.s = (self.bottom - self.top) / HEIGHT_M          # px per metre
        a = np.radians(YAW[name])
        self.ax = np.array([np.cos(a), -np.sin(a)])
        self.dir = np.array([np.sin(a), 0, np.cos(a)])

    def project(self, x, y, z):
        u = self.cx + (x * self.ax[0] + z * self.ax[1]) * self.s
        v = self.bottom - y * self.s
        return u, v

    def inside(self, x, y, z, dil):
        m = self.mask_d[dil]
        u, v = self.project(x, y, z)
        u = np.round(u).astype(int); v = np.round(v).astype(int)
        ok = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
        res = np.zeros(u.shape, bool)
        res[ok] = m[v[ok], u[ok]]
        return res


def coverage(view, pts, rows_low):
    """Fraction of the view's (lower) silhouette covered by projected points."""
    m = view.mask_d[1].copy()
    cut = int(view.bottom - rows_low * view.s)
    if rows_low < HEIGHT_M:
        m[: max(cut, 0)] = False
    else:
        m[int(view.bottom - 0.62 * view.s):] = False
    u, v = view.project(pts[:, 0], pts[:, 1], pts[:, 2])
    u = np.round(u).astype(int); v = np.round(v).astype(int)
    ok = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
    proj = np.zeros(m.shape, np.uint8)
    proj[v[ok], u[ok]] = 1
    proj = cv2.dilate(proj, np.ones((5, 5), np.uint8)).astype(bool)
    return (proj & m).sum() / max(m.sum(), 1)


def fit_view(view, others, vol, X, Y, Z, low, kilt):
    """Find yaw / offset of a 3/4 view so that carving with it keeps the hull
    consistent with the main views (no real leg removed, ghosts removed)."""
    sel = vol & low
    pts = np.c_[X[sel], Y[sel], Z[sel]][::5]
    base_cx, best = view.cx, (-1, None)
    for yaw in range(YAW[view.name] - 24, YAW[view.name] + 25, 3):
        a = np.radians(yaw)
        view.ax = np.array([np.cos(a), -np.sin(a)])
        for dx in range(-30, 31, 3):
            view.cx = base_cx + dx
            kept = pts[view.inside(pts[:, 0], pts[:, 1], pts[:, 2], 2)]
            if len(kept) < 50:
                continue
            cov = [coverage(o, kept, kilt) for o in others]
            removed = 1 - len(kept) / len(pts)
            score = min(cov) + 0.15 * removed + 0.5 * coverage(view, kept, kilt)
            if score > best[0]:
                best = (score, (yaw, dx, min(cov), removed))
    yaw, dx, cov, removed = best[1]
    a = np.radians(yaw)
    view.ax = np.array([np.cos(a), -np.sin(a)])
    view.dir = np.array([np.sin(a), 0, np.cos(a)])
    view.cx = base_cx + dx
    print(f"{view.name}: yaw {yaw}, offset {dx}px, main-view coverage {cov:.2f}, "
          f"removed {removed:.0%} of lower-body voxels")


def build(sheet_path, out_path):
    sheet = np.array(Image.open(sheet_path).convert("RGB"))
    views = {n: View(n, sheet) for n in PANELS}
    for v in views.values():
        v.mask_d = {k: cv2.dilate(v.mask.astype(np.uint8), np.ones((2 * k + 1,) * 2,
                    np.uint8)).astype(bool) for k in (0, 1, 2, 3)}
        print(f"{v.name}: height {v.bottom - v.top}px, centre {v.cx:.0f}")

    step = HEIGHT_M * 1.04 / RES
    xs = np.arange(-0.75, 0.75, step)
    ys = np.arange(-0.02, HEIGHT_M * 1.04, step)
    zs = np.arange(-0.6, 0.6, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = np.ones(X.shape, bool)
    # opposite views see the same silhouette (mirrored): union them to absorb
    # small inconsistencies in the sheet; different directions intersect.
    for group in (("front", "back"), ("right", "left")):
        sil = np.zeros(X.shape, bool)
        for g in group:
            sil |= views[g].inside(X, Y, Z, 2)
        vol &= sil
    # two orthogonal directions leave "ghost" legs (legs apart in front view x
    # legs in stride in side view = 4 legs). The 3/4 views remove them; they
    # are only used below the kilt, after fitting their exact yaw and offset.
    kilt = 0.62
    low = Y < kilt
    main = [views[k] for k in ("front", "back", "right", "left")]
    # props (bow held at the hip, quiver) also form blocks above the kilt:
    # carve the upper body with the 3/4 views too, fitted separately.
    import copy
    upper = {n: copy.copy(views[n]) for n in ("front34", "back34")}
    for name in ("front34", "back34"):
        fit_view(views[name], main, vol, X, Y, Z, low, kilt)
        fit_view(upper[name], main, vol, X, Y, Z, ~low, HEIGHT_M * 1.1)
    for name in ("front34", "back34"):
        vol &= np.where(low, views[name].inside(X, Y, Z, 2),
                        upper[name].inside(X, Y, Z, 3))
    del X, Y, Z
    # keep the main connected body
    lab = measure.label(vol, connectivity=1)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    vol = np.isin(lab, np.nonzero(sizes > sizes.max() * 0.01)[0])
    print("solid voxels:", vol.sum())

    volf = gaussian_filter(vol.astype(np.float32), 0.8)
    verts, faces, _, _ = measure.marching_cubes(
        np.pad(volf, 1), 0.5, spacing=(step, step, step))
    verts -= step
    verts += np.array([xs[0], ys[0], zs[0]])
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=12)
    mesh.fix_normals()
    v = mesh.vertices; n = mesh.vertex_normals

    # per-view visibility: splat vertex depths into a z-buffer per view
    vis = {}
    for name, view in views.items():
        u, vv = view.project(v[:, 0], v[:, 1], v[:, 2])
        d = v @ view.dir
        ui = np.round(u).astype(int); vi = np.round(vv).astype(int)
        hh, ww = view.mask.shape
        ok = (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
        zb = np.full((hh, ww), -np.inf)
        np.maximum.at(zb, (vi[ok], ui[ok]), d[ok])
        zb = cv2.dilate(zb.astype(np.float32), np.ones((3, 3), np.uint8))
        vis[name] = np.zeros(len(v), bool)
        vis[name][ok] = d[ok] >= zb[vi[ok], ui[ok]] - 0.02

    # colour only from the 4 orthographic views (the 3/4 views are not
    # pixel-consistent with them); a high power keeps one dominant view
    colour_views = {k: views[k] for k in ("front", "back", "right", "left")}
    acc = np.zeros((len(v), 3)); wsum = np.zeros(len(v))
    for name, view in colour_views.items():
        w = np.clip(n @ view.dir, 0, 1) ** 6 * vis[name]
        u, vv = view.project(v[:, 0], v[:, 1], v[:, 2])
        col = sample(view.tex, u, vv)
        acc += col * w[:, None]; wsum += w
    # vertices no view sees well: nearest view by normal, ignore visibility
    miss = wsum < 1e-3
    if miss.any():
        for name, view in colour_views.items():
            w = (np.clip(n[miss] @ view.dir, 0, 1) + 1e-3) ** 3
            u, vv = view.project(v[miss, 0], v[miss, 1], v[miss, 2])
            col = sample(view.tex, u, vv)
            acc[miss] += col * w[:, None]; wsum[miss] += w
    rgb = (acc / wsum[:, None]).clip(0, 255).astype(np.uint8)
    mesh.apply_translation([0, -mesh.bounds[0, 1], 0])
    write_glb(out_path, mesh.vertices, mesh.vertex_normals, mesh.faces, rgb)
    print(f"verts={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} -> {out_path}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
