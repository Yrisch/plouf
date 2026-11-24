# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Frame compositing for plouf: gas column density colormap, proxy splats, cores, labels.

Only numpy / scipy / matplotlib (colormaps) / PIL: used both by the render server
(shamrock process) and by the editor (proxy previews).
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ..path.camera import ZNEAR, pixel_size, project
from ..path.campath import DEFAULT_LOOK

LABEL_REF_HEIGHT = 1080  # label_size is in pixels of a 1080 px tall frame (same look at any resolution)

_LUTS = {}
_LUTS8 = {}


def lut(name):
    if name not in _LUTS:
        import matplotlib

        _LUTS[name] = (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3]).astype(np.float32)
    return _LUTS[name]


def lut8(name):
    """The colormap as the uint8 pixels compose() makes of it (same rounding)."""
    if name not in _LUTS8:
        _LUTS8[name] = _quantize(lut(name).astype(np.float64))
    return _LUTS8[name]


def _quantize(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def gas_index(sigma, clim):
    """Colormap index (uint8) of each pixel of a column density map [Msun/pc^2], log scale between 10**clim."""
    lo, hi = clim
    v = np.maximum(sigma, 1e-40)  # one float64 work array, everything else in place (full frames: ~4x faster)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(v, out=v)
        v -= lo
        v /= max(hi - lo, 1e-6)
    np.nan_to_num(v, copy=False, nan=0.0)
    np.clip(v, 0, 1, out=v)
    v *= 255
    return v.astype(np.uint8)


def gas_rgb(sigma, clim, cmap):
    """Column density [Msun/pc^2] -> float RGB in [0,1], log scale between 10**clim."""
    return lut(cmap)[gas_index(sigma, clim)]


def screen_blend(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


# ---------------------------------------------------------------------- proxy


def splat_column(pts, weights, state, nx, ny, smooth=0.7):
    """
    Cheap column density estimate from a particle subsample: each particle deposits
    weight / (pixel area at its depth) in its pixel. Same units as the real render
    if weights are particle masses in Msun and positions in code units with the
    area converted by the caller.
    """
    sx, sy, z = project(state, pts, nx, ny)
    ok = (z > ZNEAR) & (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
    ix = sx[ok].astype(np.int64)
    iy = sy[ok].astype(np.int64)
    area = pixel_size(state, z[ok], ny) ** 2
    w = weights[ok] if np.ndim(weights) else np.full(ix.shape, float(weights))
    img = np.bincount(iy * nx + ix, weights=w / area, minlength=nx * ny).reshape(ny, nx)
    if smooth:
        img = gaussian_filter(img, smooth)
    return img

SPLAT_SIGMA_H = 0.548  # Gaussian with the second moment of the M4 column kernel, in smoothing lengths
SPLAT_SIGMA_MIN = 0.7  # px: the smallest blur (unresolved particles)


def splat_sph(pts, h, weights, state, nx, ny):
    """
    Column density estimate from a particle subsample with smoothing lengths (the proxy
    sample's, grown for the sampling): each particle is a Gaussian of its kernel's width
    on screen (at least SPLAT_SIGMA_MIN px), sharp where the gas is dense and smooth where
    it is diffuse, at any resolution. Particles are sorted by width into classes a factor
    sqrt(2) apart, each deposited (bilinear) on a grid where its width is 0.7-1.4 px
    (halving the resolution per factor 2) and blurred once per class; the grids are
    upsampled and summed. Much cheaper than the kernel splatting of render/sph.py (tens of
    ms for 400k particles in a full HD frame), within ~0.1 dex of it. Units as splat_column.
    """
    sx, sy, z = project(state, pts, nx, ny)
    ok = np.isfinite(sx) & (z > ZNEAR)
    pix = pixel_size(state, np.where(ok, z, 1.0), ny)
    sig = np.maximum(SPLAT_SIGMA_H * h / pix, SPLAT_SIGMA_MIN)
    ok &= (sx + 3 * sig > 0) & (sx - 3 * sig < nx) & (sy + 3 * sig > 0) & (sy - 3 * sig < ny)
    sx, sy, sig, pix = sx[ok], sy[ok], sig[ok], pix[ok]
    w = (weights[ok] if np.ndim(weights) else np.full(len(sx), float(weights))) / pix**2
    nlev = max(0, int(math.log2(max(nx, ny) / 8)))
    lam = np.log2(sig / SPLAT_SIGMA_MIN)
    lev = np.clip(np.floor(lam), 0, nlev).astype(np.int16)
    cls = 2 * lev + np.clip(np.floor(2 * (lam - lev)), 0, 1).astype(np.int16)  # two classes per level, sqrt(2) apart
    cls[cls == 1] = 0  # one blur on the full resolution grid (0.7-1.4 px wide: the difference does not show)
    order = np.argsort(cls, kind="stable")  # radix sort of small ints
    bounds = np.searchsorted(cls[order], np.arange(2 * nlev + 3))
    acc = None
    for L in range(nlev, -1, -1):  # coarse to fine: upsample what is accumulated, add this level
        f = 2**L
        mx, my = -(-nx // f), -(-ny // f)
        if acc is not None:
            acc = _upsample2(acc)[:my, :mx]
        for k in (0, 1):
            g = order[bounds[2 * L + k] : bounds[2 * L + k + 1]]
            if not len(g):
                continue
            img = _deposit_bilinear(sx[g] / f - 0.5, sy[g] / f - 0.5, w[g] / f**2, mx, my)
            s_level = SPLAT_SIGMA_MIN * 2 ** ((k + 0.5) / 2) if L else SPLAT_SIGMA_MIN * 1.2
            gaussian_filter(img, math.sqrt(max(s_level**2 - 0.17, 0.0)), output=img, truncate=3.0)  # bilinear adds ~0.4 px
            if acc is None:
                acc = img
            else:
                acc += img
    return acc.astype(np.float64) if acc is not None else np.zeros((ny, nx))


def _deposit_bilinear(x, y, w, nx, ny):
    """Cloud in cell: w spread on the 4 pixels around (x, y) (pixel centres at integers), (ny, nx)."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    # one bincount on an image with a 1 px border (no bounds checks), cropped
    ix = np.clip(ix, -1, nx).astype(np.int64) + 1
    iy = np.clip(iy, -1, ny).astype(np.int64) + 1
    W = nx + 3
    base = iy * W + ix
    idx = np.concatenate((base, base + 1, base + W, base + W + 1))
    wts = np.concatenate(((1 - fx) * (1 - fy) * w, fx * (1 - fy) * w, (1 - fx) * fy * w, fx * fy * w))
    img = np.bincount(idx, weights=wts, minlength=(ny + 3) * W).astype(np.float32).reshape(ny + 3, W)
    return img[1 : ny + 1, 1 : nx + 1]


def _upsample2(a):
    """Linear upsampling by 2 of a pixel grid (pixel centres aligned as scipy zoom grid_mode), edges held."""
    p = np.pad(a, 1, mode="edge")
    rows = np.empty((2 * a.shape[0], p.shape[1]), a.dtype)
    rows[0::2] = 0.75 * p[1:-1] + 0.25 * p[:-2]
    rows[1::2] = 0.75 * p[1:-1] + 0.25 * p[2:]
    out = np.empty((rows.shape[0], 2 * a.shape[1]), a.dtype)
    out[:, 0::2] = 0.75 * rows[:, 1:-1] + 0.25 * rows[:, :-2]
    out[:, 1::2] = 0.75 * rows[:, 1:-1] + 0.25 * rows[:, 2:]
    return out


# ---------------------------------------------------------------------- cores


def disc_cover(ny, nx, sx, sy, rad, alpha=1.0):
    """
    Anti-aliased filled discs of one colour on a (ny, nx) image: (flat pixel indices, T),
    T the fraction of the background left on each covered pixel (the product of 1 - coverage
    over the discs on it, so the result does not depend on the drawing order). Vectorised
    over the discs, grouped by window size (thousands of sink markers in a few ms).
    """
    ok = np.isfinite(sx) & np.isfinite(sy)
    sx, sy = np.asarray(sx, float)[ok], np.asarray(sy, float)[ok]
    r = np.maximum(np.broadcast_to(np.asarray(rad, float), ok.shape)[ok], 0.8)
    half = np.ceil(r).astype(int) + 1  # window: the pixels within r + 1 of the centre
    idx, keep = [], []
    for k in np.unique(half):
        g = half == k
        x, y, rr = sx[g], sy[g], r[g]
        off = np.arange(-k, k + 1)
        px = np.floor(x).astype(np.int64)[:, None] + off  # (n, 2k+1)
        py = np.floor(y).astype(np.int64)[:, None] + off
        dist = np.hypot((px + 0.5 - x[:, None])[:, None, :], (py + 0.5 - y[:, None])[:, :, None])
        cov = np.clip(rr[:, None, None] + 0.5 - dist, 0.0, 1.0) * alpha
        inside = (cov > 0) & ((px >= 0) & (px < nx))[:, None, :] & ((py >= 0) & (py < ny))[:, :, None]
        idx.append((py[:, :, None] * nx + px[:, None, :])[inside])
        keep.append(1.0 - cov[inside])
    if not idx:
        return np.zeros(0, np.int64), np.zeros(0)
    idx, keep = np.concatenate(idx), np.concatenate(keep)
    order = np.argsort(idx, kind="stable")
    idx, keep = idx[order], keep[order]
    first = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    return idx[first], np.multiply.reduceat(keep, first)


def draw_discs(img, sx, sy, rad, color, alpha=1.0):
    """Anti-aliased filled discs of one colour on a float RGB image, in place."""
    ny, nx, _ = img.shape
    pix, T = disc_cover(ny, nx, sx, sy, rad, alpha)
    flat = img.reshape(-1, 3)
    color = np.asarray(color, dtype=float)
    flat[pix] = color + (flat[pix] - color) * T[:, None]


def core_discs(state, pos, nx, ny, radius):
    """radius: scalar or one value per core (code units)."""
    sx, sy, z = project(state, pos, nx, ny)
    ok = z > ZNEAR
    rad = np.broadcast_to(radius, z.shape) / pixel_size(state, np.where(ok, z, 1.0), ny)
    return sx[ok], sy[ok], rad[ok]


def core_radius(look, sinks):
    """World radius of the core discs: look["core_radius"] if set, else the sink accretion radii."""
    r = look.get("core_radius")
    if r is None or r <= 0:
        return sinks.get("core_racc", 0.04)
    return r


def sink_discs(state, look, sinks, nx, ny):
    """
    Discs of the sinks drawn as discs (cores, or every sink with look["sinks"] == "markers"):
    (sx, sy, radius [px], colour). Markers are at least look["marker_size"] px in radius on a
    LABEL_REF_HEIGHT tall frame, so tiny accretion radii stay visible.
    """
    look = {**DEFAULT_LOOK, **look}
    sx, sy, rad = core_discs(state, sinks["core_pos"], nx, ny, core_radius(look, sinks))
    if look["sinks"] == "markers":
        return sx, sy, np.maximum(rad, look["marker_size"] * ny / LABEL_REF_HEIGHT), look["marker_color"]
    return sx, sy, rad, look["core_color"]


# ---------------------------------------------------------------------- full frame


def compose(sigma, state, look, sinks=None, star_layer=None, label=None):
    """
    Final RGB uint8 frame.

    sigma: (ny, nx) column density [Msun/pc^2], top row first.
    sinks: dict with "core_pos" (and "core_racc") in code units if discs are drawn
    (cores, or the markers of look["sinks"] == "markers", see sink_discs).
    star_layer: from StarLight.render (RGB: screen blend, RGBA: drawn over) or None.
    """
    ny, nx = sigma.shape
    show_gas = look.get("show_gas", True)
    idx = gas_index(sigma, state.clim) if show_gas else None
    discs = None
    if look.get("show_cores", True) and sinks is not None and len(sinks["core_pos"]):
        discs = sink_discs(state, look, sinks, nx, ny)
    if star_layer is None:
        # no star layer (the instant views): pixels straight from the uint8 colormap, the
        # discs blended in float on the pixels they cover only (the same pixels as below)
        out = lut8(look["cmap"])[idx] if show_gas else np.zeros((ny, nx, 3), np.uint8)
        if discs is not None:
            sx, sy, rad, col = discs
            pix, T = disc_cover(ny, nx, sx, sy, rad)
            bg = lut(look["cmap"])[idx.reshape(-1)[pix]].astype(np.float64) if show_gas else np.zeros((len(pix), 3))
            col = np.asarray(col, dtype=float)
            out.reshape(-1, 3)[pix] = _quantize(col + (bg - col) * T[:, None])
    else:
        img = lut(look["cmap"])[idx].astype(np.float64) if show_gas else np.zeros((ny, nx, 3))
        if discs is not None:
            draw_discs(img, *discs)
        if star_layer.shape[-1] == 4:  # dots, premultiplied RGBA: over the gas
            img = img * (1 - star_layer[..., 3:]) + star_layer[..., :3]
        else:  # PSF glow: screen blend
            img = screen_blend(img, star_layer)
        out = _quantize(img)
    if label:
        out = draw_label(out, label, look.get("label_size", 36.0))
    return out


def draw_label(img, text, label_size=36.0):
    """White text in the bottom right corner, label_size px high on a LABEL_REF_HEIGHT frame."""
    from PIL import Image, ImageDraw, ImageFont

    ny, nx, _ = img.shape
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    size = max(4, round(label_size * ny / LABEL_REF_HEIGHT))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            font = ImageFont.load_default(size)  # Pillow's embedded scalable font (Pillow >= 10.1, FreeType)
        except (TypeError, ImportError, OSError):
            font = ImageFont.load_default()  # fixed bitmap font, last resort
    w = dr.textlength(text, font=font)
    dr.text((nx - w - size, ny - 2 * size), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def save_png(img, filename):
    from PIL import Image

    Image.fromarray(img).save(filename)
