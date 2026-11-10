"""Convert a green-screen character image into a textured, volumetric GLB.

Method (no GPU / no AI weights needed):
  1. Chroma-key the green background -> silhouette mask, remove green spill.
  2. Solve a Poisson equation inside the silhouette and take sqrt -> rounded
     "inflated" depth whose thickness follows local body width (limbs round,
     torso wide, bow/arrows thin).
  3. Build a closed mesh: front surface textured with the original photo,
     back surface textured with a version where the face is smoothed away.
  4. Export as binary glTF (.glb), 1.80 m tall, feet on the ground (y = 0).

Usage: python image_to_glb.py input.png output.glb
"""
import sys

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
from PIL import Image

STRIDE = 2            # mesh resolution (pixels per vertex)
HEIGHT_M = 1.80       # final model height
FRONT_DEPTH = 1.25    # depth scale of front half relative to width
BACK_DEPTH = 1.10     # depth scale of back half


def chroma_mask(img):
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    green = (g > 90) & (g > r * 1.35) & (g > b * 1.35)
    mask = (~green).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # keep the largest few components (character + bow pieces), drop specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            keep[lab == i] = 1
    # fill holes inside the body
    filled = keep.copy()
    inv = (1 - keep).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        touches = x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]
        if not touches and a < 400:
            filled[lab == i] = 1
    return filled.astype(bool)


def despill(img, mask):
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    g2 = np.minimum(g, np.maximum(r, b))
    out = np.stack([r, g2, b], -1)
    # bleed edge colours outward so texture filtering never samples green
    m = mask.astype(np.uint8)
    out = cv2.inpaint(np.clip(out, 0, 255).astype(np.uint8), 1 - m, 5, cv2.INPAINT_TELEA)
    return out


def poisson_depth(mask):
    h, w = mask.shape
    idx = -np.ones(mask.shape, np.int64)
    ys, xs = np.nonzero(mask)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    rows, cols, vals = [], [], []
    diag = np.full(n, 4.0)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nb = np.full(n, -1)
        nb[ok] = idx[ny[ok], nx[ok]]
        inside = nb >= 0
        rows.append(np.nonzero(inside)[0])
        cols.append(nb[inside])
        vals.append(np.full(inside.sum(), -1.0))
    A = sp.csr_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [np.arange(n)]),
                        np.concatenate(cols + [np.arange(n)]))), shape=(n, n))
    sol = spla.spsolve(A.tocsc(), np.ones(n))
    d = np.zeros(mask.shape)
    d[ys, xs] = np.sqrt(np.maximum(sol, 0))
    return d


def back_texture(tex, img_h):
    """Back of the body: same colours, but face features smoothed into skin."""
    back = tex.copy()
    sy = img_h / 1376.0
    y0, y1, x0, x1 = int(105 * sy), int(225 * sy), int(335 * sy), int(445 * sy)
    region = back[y0:y1, x0:x1]
    skin = cv2.GaussianBlur(region, (0, 0), 18)
    yy, xx = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    cy, cx = region.shape[0] / 2, region.shape[1] / 2
    wgt = np.clip(1.4 - np.hypot((yy - cy) / cy, (xx - cx) / cx), 0, 1)[..., None]
    back[y0:y1, x0:x1] = (skin * wgt + region * (1 - wgt)).astype(np.uint8)
    return cv2.GaussianBlur(back, (0, 0), 1.2)


def build(img_path, out_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    mask_full = chroma_mask(img)
    tex = despill(img, mask_full)

    mask = cv2.resize(mask_full.astype(np.uint8), (W // STRIDE, H // STRIDE),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((2, 2), np.uint8)).astype(bool)
    h, w = mask.shape
    depth = poisson_depth(mask)
    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.0) * mask

    # boundary pixels get zero depth so front and back meet at the silhouette
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = mask & ~eroded
    depth[boundary] = 0

    ys, xs = np.nonzero(mask)
    fidx = -np.ones((h, w), np.int64)
    fidx[ys, xs] = np.arange(len(ys))
    nf = len(ys)
    # back gets its own vertices (same position on the silhouette edge, but
    # separate UVs so back faces never stretch across the texture atlas)
    interior = np.ones(nf, bool)
    bidx = -np.ones((h, w), np.int64)
    bidx[ys, xs] = nf + np.arange(nf)

    scale = HEIGHT_M / h
    X = (xs - w / 2) * scale
    Y = (h - ys) * scale
    Y -= Y.min()
    Zf = depth[ys, xs] * FRONT_DEPTH * scale
    Zb = -depth[ys, xs] * BACK_DEPTH * scale

    verts = np.concatenate([np.stack([X, Y, Zf], 1),
                            np.stack([X[interior], Y[interior], Zb[interior]], 1)])
    u = (xs + 0.5) * STRIDE / W
    v = 1 - (ys + 0.5) * STRIDE / H
    # atlas: left half = front texture, right half = back texture
    uv = np.concatenate([np.stack([u * 0.5, v], 1),
                         np.stack([0.5 + u[interior] * 0.5, v[interior]], 1)])

    # quads -> triangles where all 4 corners are inside
    a = fidx[:-1, :-1]; b = fidx[:-1, 1:]; c = fidx[1:, :-1]; d = fidx[1:, 1:]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    A, B, C, D = a[ok], b[ok], c[ok], d[ok]
    front = np.concatenate([np.stack([A, C, B], 1), np.stack([B, C, D], 1)])
    a = bidx[:-1, :-1]; b = bidx[:-1, 1:]; c = bidx[1:, :-1]; d = bidx[1:, 1:]
    A, B, C, D = a[ok], b[ok], c[ok], d[ok]
    back = np.concatenate([np.stack([A, B, C], 1), np.stack([B, D, C], 1)])
    faces = np.concatenate([front, back])

    atlas = np.concatenate([tex, back_texture(tex, H)], 1)
    material = trimesh.visual.material.PBRMaterial(
        name="archer", baseColorTexture=Image.fromarray(atlas),
        metallicFactor=0.0, roughnessFactor=0.8, doubleSided=True)
    mesh = trimesh.Trimesh(verts, faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    mesh.export(out_path)
    print(f"verts={len(verts)} faces={len(faces)} -> {out_path}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
