# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Numpy SPH renderer: the fallback when shamrock is not installed.

Column density maps through the perspective camera by kernel splatting, as splash does:
each gas particle adds its M4 (cubic spline) kernel integrated along the line of sight of
each pixel, ``m / h^2 F(b / h)``, b being the distance from the particle to the pixel's ray
(exact in perspective: off the view axis the footprint is stretched radially). The dumps
are read with the fast readers (data/readers.py), so only numpy / scipy are needed.
Slower than shamrock (a few seconds per frame for a few million particles);
the images match shamrock's to ~1% (tests/test_shamrock.py).

Splatting cost is bounded per particle: a particle whose footprint is larger than
STAMP_MAX pixels (across, off axis it is longer radially) is drawn on a coarser level
(pixels 2, 4, 8... times larger, where its footprint is small again), and the levels are
summed after cubic upsampling. A kernel
that wide is smooth at the scale of the coarse pixels, so nothing visible is lost.
Particles smaller than a few pixels spread exactly their mass over the pixels they
touch (normalised stamps; point sampling would alias them), larger ones are sampled at
the pixel centres like shamrock's rays.

NumpyScene has the interface of backend.SimScene (load, sinks, column, column_to_points,
gas_sample, meta) so FrameRenderer and the render process use either.
"""

import math
import os
import time

import numpy as np
from scipy.ndimage import zoom

from ..constants import MSUN_G, MYR_S, PC_CM
from ..data.readers import get_reader
from ..path.camera import ZNEAR, project

SUPPORT = 2.0  # M4 kernel support, in smoothing lengths
STAMP_MAX = 8  # largest footprint radius drawn on a level, px of that level (8: kernels within ~0.2% of exact; 6: 3%)
STRETCH_MAX = 3  # off axis, footprints stretched radially up to this factor still use stamps (half width <= 3 STAMP_MAX)
NORM_BELOW = 1.5  # particles with h below this many pixels deposit their exact mass (normalised stamps)
UPSAMPLE_ORDER = 3  # spline order of the coarse levels' upsampling
CHUNK = 1_000_000  # kernel samples evaluated at once per thread (memory ~ 10 x 4 bytes each)
THREADS = min(8, os.cpu_count() or 1)


def _column_kernel(n=512):
    """F(q), q in [0, SUPPORT]: the M4 kernel integrated along a line at impact parameter q h, times h^2."""

    def w3(r):  # 3D M4 kernel times h^3
        return np.where(r < 1, 1 - 1.5 * r**2 + 0.75 * r**3, np.where(r < 2, 0.25 * (2 - r) ** 3, 0.0)) / math.pi

    q = np.linspace(0.0, SUPPORT, n)
    s = np.linspace(0.0, SUPPORT, 2001)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2
    F = 2 * trapz(w3(np.hypot(q[:, None], s[None, :])), s, axis=1)
    F /= trapz(F * 2 * math.pi * q, q)  # a particle's column integrates to its mass
    F[-1] = 0.0
    return q, F


_Q, _F = _column_kernel()


def kernel_column(q):
    """F(q) with q = b / h; the column of a particle is m / h^2 F(b / h)."""
    return np.interp(q, _Q, _F, right=0.0)


# F tabulated in q^2 for the stamps (a lookup, no square root); the last entry is 0, beyond the support
_NQ2 = 4096
_Q2_SCALE = (_NQ2 - 1) / SUPPORT**2
_F_Q2 = np.append(kernel_column(np.sqrt(np.arange(_NQ2) / _Q2_SCALE)), 0.0).astype(np.float32)


def _kernel_q2(q2, scaled=False):
    """F(sqrt(q2)) for the stamps (nearest of the table; 0 beyond the support). scaled: q2 is already times _Q2_SCALE."""
    if not scaled:
        q2 = q2 * np.float32(_Q2_SCALE)
    q2 += np.float32(0.5)
    return np.take(_F_Q2, q2.astype(np.int32), mode="clip")  # the last entry is 0: beyond the support


def _stamp_q2(dx, dy, a, b, kpix, hp, scale=1.0):
    """
    (d / h)^2 on a stamp, d the distance from the particle to the ray of each pixel.
    dx (n, K), dy (n, K): pixel centre minus particle position [px, y down]; a, b (n,):
    tan coordinates of the particle (x / z, y / z in the camera frame, y up); kpix: tan
    units per pixel; hp (n,): smoothing length [px at the particle depth]. With the
    particle at p = z (a, b, 1) and the ray along (X, Y, 1),
    d = |p x ray| / |ray| = z kpix sqrt(dx^2 + dy^2 + (a dy' - b dx)^2) / sqrt(1 + X^2 + Y^2),
    dy' = -dy (up), X = a + kpix dx, Y = b + kpix dy', and z kpix is the pixel size at depth z.
    Returns (n, K, K), rows along dy, times scale.
    """
    a, b = a[:, None], b[:, None]
    up = -dy
    X2 = (a + kpix * dx) ** 2  # (n, K)
    Y2 = (b + kpix * up) ** 2
    cross = (a * up)[:, :, None] - (b * dx)[:, None, :]  # (n, K, K)
    num = (dy**2)[:, :, None] + (dx**2)[:, None, :] + cross**2
    den = (1 + Y2)[:, :, None] + X2[:, None, :]
    den *= (hp**2 / scale)[:, None, None]
    num /= den
    return num


def _screen(state, pos, h, nx, ny):
    """
    Where the particles are on screen. Returns (sx, sy [px], z [depth], a, b [tan coords,
    x / z and y / z with y up], hp [h in px at the particle depth], R [px], kpix). R bounds
    the footprint on screen (the support cone seen through the image plane, stretched
    radially off axis); it is infinite when the camera is inside the support or the
    footprint reaches 90 degrees off axis.
    """
    sx, sy, z = project(state, pos, nx, ny)
    kpix = 2 * math.tan(math.radians(state.fov) / 2) / ny  # tan units per pixel
    a = (sx - nx / 2 * (1 - state.shift)) * kpix  # true direction x / z: the lens shift only moves the image (project)
    b = (ny / 2 - sy) * kpix
    front = np.isfinite(sx) & (z > ZNEAR)
    zz = np.where(front, z, 1.0)
    hp = h / (zz * kpix)
    t = np.hypot(a, b)
    r = zz * np.sqrt(1 + t**2)  # distance to the camera
    with np.errstate(invalid="ignore"):
        alpha = np.arcsin(np.minimum(SUPPORT * h / r, 1.0))  # half angle of the support seen from the camera
        edge = np.arctan(t) + alpha
        R = np.where((SUPPORT * h < r) & (edge < math.radians(89)), (np.tan(edge) - t) / kpix, np.inf)
    R = np.where(front, np.maximum(R, SUPPORT * hp), np.nan)
    return sx, sy, z, a, b, hp, R, kpix


def _deposit_stamps(img, pad, x, y, a, b, kpix, hp, amp, norm, k, check=None):
    """
    Add kernel stamps (half width k px) to img, in place. img has pad extra pixels on each
    side (>= 2k + 1, so no stamp of a visible particle falls outside it); x, y are in the
    unpadded frame. Particle i adds amp[i] F(q) at each pixel, or, where norm[i] (too small
    for point sampling), spreads amp[i] (its column times pixel area, in pixel units) in
    proportion to F(q), so it deposits exactly that. Returns the mask of the normalised
    particles too small to touch a pixel centre (nothing deposited for them).
    Chunks of particles run in threads (numpy releases the GIL in these array operations).
    check: called before each chunk (it raises to drop the render).
    """
    nxp = img.shape[1]
    f32 = np.float32
    off = np.arange(-k, k + 1, dtype=f32) + f32(0.5)
    stamp = (np.arange(2 * k + 1)[:, None] * nxp + np.arange(2 * k + 1)[None, :]).ravel()
    miss = np.zeros(len(x), bool)

    def chunk(sl):
        if check:
            check()
        cx, cy = np.floor(x[sl]), np.floor(y[sl])
        dx = off + (cx - x[sl]).astype(f32)[:, None]  # (n, 2k+1)
        dy = off + (cy - y[sl]).astype(f32)[:, None]
        q2 = _stamp_q2(dx, dy, a[sl].astype(f32), b[sl].astype(f32), f32(kpix), hp[sl].astype(f32), f32(_Q2_SCALE))
        f = _kernel_q2(q2, scaled=True)
        s = f.sum(axis=(1, 2), dtype=np.float64)
        nm = norm[sl]
        miss[sl] = nm & (s <= 0)
        f *= np.where(nm, amp[sl] / np.where(s > 0, s, 1.0), amp[sl]).astype(f32)[:, None, None]
        corner = (cy.astype(np.int64) + pad - k) * nxp + cx.astype(np.int64) + pad - k
        flat = (corner[:, None] + stamp[None, :]).ravel()
        return np.bincount(flat, weights=f.ravel(), minlength=img.size)

    step = max(1, CHUNK // len(off) ** 2)
    slices = [slice(s0, s0 + step) for s0 in range(0, len(x), step)]
    if len(slices) > 1:
        for part in _pool().map(chunk, slices):
            img += part.reshape(img.shape)
    elif slices:
        img += chunk(slices[0]).reshape(img.shape)
    return miss


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=THREADS, thread_name_prefix="sph")
    return _POOL


def _deposit_whole(img, pad, x, y, a, b, kpix, hp, amp):
    """Particles wider than a stamp (coarsest level only): amp F(q) at every pixel of img (pad px around the frame)."""
    ny, nx = img.shape
    cx = (np.arange(nx) + 0.5 - pad).astype(np.float32)
    cy = (np.arange(ny) + 0.5 - pad).astype(np.float32)
    step = max(1, CHUNK // (nx * ny))
    f32 = np.float32
    for s0 in range(0, len(x), step):
        sl = slice(s0, s0 + step)
        dx = cx[None, :] - x[sl].astype(f32)[:, None]
        dy = cy[None, :] - y[sl].astype(f32)[:, None]
        q2 = _stamp_q2(dx, dy, a[sl].astype(f32), b[sl].astype(f32), f32(kpix), hp[sl].astype(f32))
        img += np.einsum("nij,n->ij", _kernel_q2(q2), amp[sl].astype(f32))


def _deposit_points(img, x, y, w):
    ny, nx = img.shape
    ix, iy = x.astype(np.int64), y.astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    img += np.bincount(iy[ok] * nx + ix[ok], weights=w[ok], minlength=nx * ny).reshape(ny, nx)


def column_render(pos, h, mass, state, nx, ny, check=None):
    """
    Column density (ny, nx), top row first, in code units (mass / length^2), of the
    particles at pos (n, 3) with smoothing lengths h (n,) and masses (scalar or (n,)),
    seen by the camera state (lens shift included, as project() applies it).
    check: called between chunks of work; it may raise to drop the render.
    """
    sx, sy, z, a, b, hp, R, kpix = _screen(state, pos, h, nx, ny)
    vis = np.isfinite(sx) & (sx + R > 0) & (sx - R < nx) & (sy + R > 0) & (sy - R < ny)  # NaN R: behind
    sx, sy, z, a, b, hp, R = (v[vis] for v in (sx, sy, z, a, b, hp, R))
    hh = h[vis]
    m = mass[vis] if np.ndim(mass) else np.full(len(sx), float(mass))
    amp = m / hh**2  # column through the centre / F(0)
    # mass seen through the image plane, per pixel: m / pixel area, the area of a pixel
    # projected perpendicular to the line of sight being (z kpix)^2 cos(theta)
    spread = m / (z * kpix) ** 2 * np.sqrt(1 + a**2 + b**2)

    # level from the footprint's width (2 hp across, it is longer radially off axis), so
    # that the kernel stays resolved across on every level: hp >= STAMP_MAX / 4 coarse px
    nlev = max(0, int(math.log2(max(nx, ny) / STAMP_MAX)))  # the coarsest level is about 2 STAMP_MAX px wide
    kmax = STRETCH_MAX * STAMP_MAX
    lev = np.clip(np.ceil(np.log2(SUPPORT * hp / STAMP_MAX)), 0, nlev).astype(int)
    lev[np.minimum(R, 1e30) / 2.0**lev > kmax] = nlev  # extreme stretch (or camera inside): whole coarsest level
    out = np.zeros((ny, nx))
    small = (R < 0.5) & (lev == 0)
    _deposit_points(out, sx[small], sy[small], spread[small])
    pad = 2 * kmax + 2
    mg = 6  # margin of a coarse level upsampled with the frame (stamps spill into it; the cubic prefilter decays as 0.27^px)
    for L in range(nlev + 1):
        if check:
            check()
        sel = (lev == L) & ~small
        if not sel.any():
            continue
        f = 2**L
        mx, my = -(-nx // f), -(-ny // f)
        padded = np.zeros((my + 2 * pad, mx + 2 * pad))
        img = padded[pad:-pad, pad:-pad]
        sub = padded[pad - mg : pad + my + mg, pad - mg : pad + mx + mg]  # what is upsampled
        x, y, al, bl, hl, Rl = sx[sel] / f, sy[sel] / f, a[sel], b[sel], hp[sel] / f, R[sel] / f
        norm = hl < NORM_BELOW
        amp_l = np.where(norm, spread[sel] / f**2, amp[sel])  # same mass on pixels f^2 larger
        k = np.ceil(np.minimum(Rl, 1e9)).astype(int)
        huge = k > kmax  # only on the coarsest level, which is about as wide as a stamp
        _deposit_whole(sub, mg, x[huge], y[huge], al[huge], bl[huge], kpix * f, hl[huge], amp[sel][huge])
        miss = np.zeros(len(x), bool)
        for kk in np.unique(k[~huge]):
            g = (k == kk) & ~huge
            miss[g] = _deposit_stamps(padded, pad, x[g], y[g], al[g], bl[g], kpix * f, hl[g], amp_l[g], norm[g], int(kk), check)
        _deposit_points(img, x[miss], y[miss], amp_l[miss])
        if L == 0:
            out += img
        else:  # the kernels here are >= STAMP_MAX / 2 coarse pixels wide: smooth at this scale
            # the frame edges are interpolated (with the margin), not extrapolated
            up = zoom(sub, f, order=UPSAMPLE_ORDER, mode="nearest", grid_mode=True)
            out += up[mg * f : mg * f + ny, mg * f : mg * f + nx]
    return np.maximum(out, 0.0)  # cubic upsampling may undershoot slightly next to steep edges


def column_to_points(pos, h, mass, state, pts, n=1024, check=None):
    """
    Gas column between the camera of state and each point of pts (in front of it), in
    code units: the particles nearer than the point (in depth) whose kernel the line of
    sight crosses, as column_render sees them. n only sets the screen grid (n x n) used to
    find the candidates.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    out = np.zeros(len(pts))
    if not len(pts):
        return out
    st = state.copy()
    st.shift = 0.0
    px, py, pz = project(st, pts, n, n)
    ok = np.isfinite(px)
    if not ok.any():
        return out
    sx, sy, z, a, b, hp, R, kpix = _screen(st, pos, h, n, n)
    front = np.isfinite(sx) & (z < pz[ok].max())
    R = np.where(front, np.minimum(R, 4.0 * n), 0.0)  # an infinite footprint covers the whole (square) screen
    m = np.broadcast_to(mass, z.shape)
    stars = cKDTree(np.stack([px[ok], py[ok]], axis=1))
    sid = np.flatnonzero(ok)
    # candidates by footprint size (powers of 2): pairs closer than the largest footprint of the bin
    lev = np.floor(np.log2(np.maximum(R, 1e-30))).astype(int)
    lev[~front] = -999
    for L in np.unique(lev[front]):
        if check:
            check()
        g = np.flatnonzero(lev == L)
        parts = cKDTree(np.stack([sx[g], sy[g]], axis=1))
        pairs = stars.sparse_distance_matrix(parts, max_distance=2.0 ** (L + 1), output_type="ndarray")
        if not len(pairs):
            continue
        i, j = sid[pairs["i"]], g[pairs["j"]]
        keep = z[j] < pz[i]
        i, j = i[keep], j[keep]
        dx, dy = (px[i] - sx[j])[:, None], (py[i] - sy[j])[:, None]
        q2 = _stamp_q2(dx, dy, a[j], b[j], kpix, hp[j])[:, 0, 0]
        out += np.bincount(i, weights=m[j] / h[j] ** 2 * kernel_column(np.sqrt(q2)), minlength=len(pts))
    return out


class NumpyScene:
    """One dump read with the fast reader, rendered with numpy (interface of backend.SimScene)."""

    name = "numpy"

    def __init__(self, fmt=None):
        self.reader = get_reader(fmt)
        self.path = None
        self.gas = None

    @property
    def loaded(self):
        return self.gas is not None

    def load(self, path, fmt=None):
        if fmt and fmt != self.reader.name:
            self.reader = get_reader(fmt)
            self.path = None
        if path == self.path:
            return False
        t0 = time.perf_counter()
        self.gas = None
        g = self.reader.read_gas(path)
        alive = g["alive"]
        m = g["mass"]
        self.gas = {
            "pos": np.ascontiguousarray(g["pos"][alive], dtype=float),
            "h": np.abs(np.asarray(g["h"][alive], dtype=float)),
            "mass": m[alive].astype(float) if np.ndim(m) else float(m),
        }
        self.path = path
        self.pc_per_unit = g["udist"] / PC_CM
        self.msun_per_unit = g["umass"] / MSUN_G
        self.sigma_unit = self.msun_per_unit / self.pc_per_unit**2  # code column -> Msun/pc^2
        self.time = g["time"]
        self.time_myr = g["time"] * g["utime"] / MYR_S
        self.npart = int(alive.sum())
        self.particle_mass = float(np.mean(m[alive])) if np.ndim(m) else float(m)
        self.sinks = alive_sinks(self.reader.read_sinks(path), self.msun_per_unit)
        self.load_time = time.perf_counter() - t0
        return True

    def gas_sample(self, n=400_000, seed=0):
        xyz = self.gas["pos"]
        N = len(xyz)
        if N > n:
            xyz = xyz[np.random.default_rng(seed).choice(N, n, replace=False)]
        w = self.particle_mass * N / max(len(xyz), 1) * self.sigma_unit
        return xyz.astype(np.float32), w

    def meta(self):
        return {
            "path": self.path,
            "time": self.time,
            "time_myr": self.time_myr,
            "pc_per_unit": self.pc_per_unit,
            "msun_per_unit": self.msun_per_unit,
            "npart": self.npart,
            "load_time": self.load_time,
            "backend": self.name,
        }

    def column(self, state, nx, ny, check=None):
        """Column density [Msun/pc^2] seen by the camera, (ny, nx), top row first."""
        g = self.gas
        return column_render(g["pos"], g["h"], g["mass"], state, nx, ny, check) * self.sigma_unit

    def column_to_points(self, origin, pts, state=None, check=None):
        """Gas column [Msun/pc^2] between the camera (state, at origin) and each point."""
        g = self.gas
        return column_to_points(g["pos"], g["h"], g["mass"], state, pts, check=check) * self.sigma_unit


def alive_sinks(s, msun_per_unit):
    """The alive sinks of a reader's read_sinks() result: {"id", "pos", "mass" [Msun], "racc" [code units]}."""
    m = np.asarray(s["mass"], dtype=float)
    alive = m > 0
    return {
        "id": np.flatnonzero(alive),  # index in the dump sink arrays, stable across dumps
        "pos": np.asarray(s["pos"], dtype=float)[alive],
        "mass": m[alive] * msun_per_unit,
        "racc": np.asarray(s["racc"], dtype=float)[alive],
    }
