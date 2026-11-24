# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Render backends for plouf.

SimScene owns one Phantom dump loaded in a shamrock SPH model and renders
column density maps through a perspective camera (shamrock.math.Camera3d_f64_3).
Without shamrock, make_scene falls back to sph.NumpyScene (same interface, numpy only).

server_main is the entry point of the render process used by the editor:
shamrock (MPI + SYCL) lives only there so the GUI never blocks on it.
"""

import atexit
import gc
import math
import os
import time
import traceback

import numpy as np

from ..constants import MSUN_G, MYR_S, PC_CM
from ..data.sinktracks import SinkTracks, split_sinks
from ..path.camera import ZNEAR, CamState
from .compose import compose
from .sph import NumpyScene, alive_sinks
from .stars import StarLight

_shamrock = None
BACKENDS = ("auto", "shamrock", "numpy")
RENDER_NICE = 10  # the editor's render process runs at a lower priority: the GUI (proxy views) comes first


class Cancelled(Exception):
    """A render dropped on the way: the view it was for changed (see server_main)."""


def init_shamrock(sycl_cfg="0:0", loglevel=0):
    global _shamrock
    if _shamrock is None:
        import shamrock

        _shamrock = shamrock
    if not _shamrock.sys.is_initialized():
        _shamrock.change_loglevel(loglevel)
        _shamrock.sys.init(sycl_cfg)
        atexit.register(_shamrock.sys.close)
    return _shamrock


def make_scene(backend="auto", sycl_cfg="0:0", fmt=None):
    """
    The scene of a backend: "shamrock", "numpy" (sph.NumpyScene), or "auto": shamrock
    when its python module imports, numpy otherwise.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (known: {', '.join(BACKENDS)})")
    if backend != "numpy":
        try:
            init_shamrock(sycl_cfg)
        except ImportError:
            if backend == "shamrock":
                raise
        else:
            return SimScene()
    return NumpyScene(fmt)


class SimScene:
    name = "shamrock"

    def __init__(self):
        self.sham = init_shamrock()
        self.path = None
        self.ctx = None
        self.model = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self, path, fmt=None):
        """fmt: ignored, shamrock reads Phantom dumps only."""
        if path == self.path:
            return False
        sham = self.sham
        self.model = None
        self.ctx = None
        gc.collect()

        t0 = time.perf_counter()
        dump = sham.load_phantom_dump(path)
        ctx = sham.Context()
        ctx.pdata_layout_new()
        model = sham.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
        cfg = model.gen_config_from_phantom_dump(dump, bypass_error=True)
        model.set_solver_config(cfg)
        model.init_scheduler(int(1e8), 1)
        model.init_from_phantom_dump(dump)
        self.ctx, self.model, self.path = ctx, model, path

        u = model.get_units().to_json()
        self.pc_per_unit = u["unit_length"] / PC_CM
        self.msun_per_unit = u["unit_mass"] / MSUN_G
        self.sigma_unit = self.msun_per_unit / self.pc_per_unit**2  # code column -> Msun/pc^2
        self.time = model.get_time()
        self.time_myr = self.time * u["unit_time"] / MYR_S
        self.particle_mass = model.get_particle_mass()
        self.npart = model.get_total_part_count()
        self._load_sinks()
        self.load_time = time.perf_counter() - t0
        return True

    def _load_sinks(self):
        sinks = self.model.get_sinks()
        if sinks:
            pos = np.array([s["pos"] for s in sinks], dtype=float)
            m = np.array([s["mass"] for s in sinks], dtype=float)
            racc = np.array([s["accretion_radius"] for s in sinks], dtype=float)
        else:
            pos, m, racc = np.zeros((0, 3)), np.zeros(0), np.zeros(0)
        self.sinks = alive_sinks({"pos": pos, "mass": m, "racc": racc}, self.msun_per_unit)  # split by the look when drawn

    def gas_sample(self, n=400_000, seed=0):
        xyz = np.asarray(self.ctx.collect_data()["xyz"])
        N = len(xyz)
        if N > n:
            idx = np.random.default_rng(seed).choice(N, n, replace=False)
            xyz = xyz[idx]
        # weight such that splat_column gives Msun/pc^2 with code-unit areas
        w = self.particle_mass * N / len(xyz) * self.sigma_unit
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

    # -- rendering

    def column(self, state, nx, ny, check=None):
        """Column density [Msun/pc^2] seen by the camera, (ny, nx), top row first. check: see server_main."""
        if check:
            check()  # one shamrock call: it cannot be interrupted once started
        d, r, u = state.basis()
        # a lens shift (stereo eye) of a whole number of pixels: a centred camera wide enough, cropped
        sh = int(round(state.shift * nx / 2))
        cam = self.sham.math.Camera3d_f64_3(
            pos=tuple(state.pos),
            dir=tuple(d),
            up=tuple(u),
            nx=int(nx + 2 * abs(sh)),
            ny=int(ny),
            fovy=math.radians(state.fov),
            znear=ZNEAR,
            zfar=1e30,
        )
        arr = np.asarray(self.model.render_column_integ("rho", "f64", cam.get_rays()))
        x0 = abs(sh) + sh
        return arr.reshape(ny, nx + 2 * abs(sh))[::-1, x0 : x0 + nx] * self.sigma_unit

    def column_to_points(self, origin, pts, state=None, check=None):
        """Gas column [Msun/pc^2] between origin and each point (two half-line integrals; state unused)."""
        if check:
            check()
        if len(pts) == 0:
            return np.zeros(0)
        Ray = self.sham.math.Ray_f64_3
        dirs = pts - origin
        o = tuple(origin)
        rays_cam = [Ray(o, tuple(dv), 0.0) for dv in dirs]
        rays_beyond = [Ray(tuple(p), tuple(dv), 0.0) for p, dv in zip(pts, dirs)]
        a = np.asarray(self.model.render_column_integ("rho", "f64", rays_cam))
        b = np.asarray(self.model.render_column_integ("rho", "f64", rays_beyond))
        return np.maximum(a - b, 0.0) * self.sigma_unit


class FrameRenderer:
    """Full frame (gas + cores + stars + label) from a SimScene, shared by editor and batch."""

    def __init__(self, scene, tracks=None):
        self.scene = scene
        self.starlight = StarLight()
        self._lum_cache = (None, None)
        self.tracks = tracks  # sinktracks.SinkTracks: sinks drawn at their position between dumps

    def sinks_at(self, dump, look=None):
        """
        Sinks of the loaded dump to draw with this look (sinktracks.split_sinks), moved to
        their smooth position at a fractional dump.
        """
        raw = self.scene.sinks
        tr = self.tracks
        if tr is not None and abs(dump - round(dump)) > 1e-6:
            raw = dict(raw)
            ids = raw["id"]
            ok = ids < tr.nsink
            pos = raw["pos"].copy()
            if ok.any():
                pos[ok] = tr.positions_at(ids[ok], dump)
            raw["pos"] = pos
        return split_sinks(**raw, pc_per_unit=self.scene.pc_per_unit, look=look)

    def star_lum(self, star_mass):
        key = (self.scene.path, star_mass.tobytes())
        if self._lum_cache[0] != key:
            self._lum_cache = (key, self.starlight.luminosities(star_mass)[0])
        return self._lum_cache[1]

    def render(self, state, nx, ny, look, timings=None, exposure=None, check=None):
        """
        exposure: star exposure to use instead of state.exposure (auto exposure), dex.
        check: called now and then, raises Cancelled to drop the render (see server_main).
        """
        timings = {} if timings is None else timings
        sc = self.scene
        t0 = time.perf_counter()
        sigma = sc.column(state, nx, ny, check=check) if look.get("show_gas", True) else np.zeros((ny, nx))
        timings["gas"] = time.perf_counter() - t0
        if check:
            check()

        sinks = self.sinks_at(state.dump, look)
        star_layer = None
        if look.get("show_stars", True) and len(sinks["star_pos"]):
            t0 = time.perf_counter()
            pos = sinks["star_pos"]
            sig = None
            if state.extinction > 0:
                # only stars in front of the camera matter
                d = state.basis()[0]
                front = (pos - state.pos) @ d > ZNEAR
                sig = np.zeros(len(pos))
                sig[front] = sc.column_to_points(state.pos, pos[front], state=state, check=check)
            lum = self.star_lum(sinks["star_mass"])
            star_layer = self.starlight.render(state, pos, lum, sig, nx, ny, look, sc.pc_per_unit, exposure)
            timings["stars"] = time.perf_counter() - t0

        label = f"t = {sc.time_myr:.2f} Myr" if look.get("time_label", True) else None
        return compose(sigma, state, look, sinks, star_layer, label)


# ---------------------------------------------------------------------- render server


def server_main(conn, sycl_cfg="0:0", sample_n=400_000, backend="auto", generation=None):
    """
    Message loop of the render process (backend: see make_scene).

    requests:  {"cmd": "render", "id", "dump", "state", "nx", "ny", "look", "exposure" (optional),
                "format" (optional, dump format of the numpy backend), "gen" (optional)}
               {"cmd": "tracks", "path"}  (sink tracks cache, see data/sinktracks.py)
               {"cmd": "quit"}
    replies:   {"type": "backend", "name"} once ready (the scene's name: "shamrock" or "numpy")
               {"type": "loaded", "meta", "sample", "weight", "sinks" (alive sinks: "id", "pos", "mass" [Msun], "racc")}
               {"type": "frame", "id", "rgb", "info"}
               {"type": "cancelled", "id"}
               {"type": "status", "msg"} / {"type": "error", "id", "msg"}

    Pending render requests are collapsed: only the most recent one is rendered,
    except those flagged "keep" (export of a full resolution frame).

    generation: a shared multiprocessing.Value the editor increments when the view it
    asked a render for has changed (workers.Backend.cancel). A render whose "gen" is older
    is dropped as soon as the renderer checks (between steps, often in the numpy renderer)
    and answered "cancelled", so it does not compete with the GUI for nothing; "keep"
    renders (exports) are always finished. The process also runs at a lower priority.
    """
    try:
        os.nice(RENDER_NICE)
    except (AttributeError, OSError):
        pass
    try:
        scene = make_scene(backend, sycl_cfg)
        renderer = FrameRenderer(scene)
    except Exception:
        conn.send({"type": "error", "id": None, "msg": traceback.format_exc()})
        return
    conn.send({"type": "backend", "name": scene.name})
    if scene.name == "shamrock":
        conn.send({"type": "status", "msg": "shamrock ready"})
    elif backend == "auto":
        conn.send({"type": "status", "msg": "shamrock not available: renders with the numpy SPH renderer (slower)"})

    while True:
        msgs = [conn.recv()]
        while conn.poll():
            msgs.append(conn.recv())
        if any(m["cmd"] == "quit" for m in msgs):
            break
        for m in msgs:
            if m["cmd"] == "tracks":
                try:
                    renderer.tracks = SinkTracks.load(m["path"]) if m["path"] else None
                except Exception:
                    conn.send({"type": "error", "id": None, "msg": traceback.format_exc()})
        renders = [m for m in msgs if m["cmd"] == "render"]
        todo = [m for m in renders if m.get("keep")]
        latest = [m for m in renders if not m.get("keep")]
        if latest:
            todo.append(latest[-1])

        for m in todo:

            def check(m=m):
                if generation is not None and not m.get("keep") and generation.value > m.get("gen", 0):
                    raise Cancelled

            try:
                check()
                if m["dump"] and m["dump"] != scene.path:
                    conn.send({"type": "status", "msg": f"loading {os.path.basename(m['dump'])} ..."})
                    scene.load(m["dump"], m.get("format"))
                    xyz, w = scene.gas_sample(sample_n)
                    conn.send(
                        {"type": "loaded", "meta": scene.meta(), "sample": xyz, "weight": w, "sinks": scene.sinks}
                    )
                if not scene.loaded:
                    raise RuntimeError("no dump loaded")
                state = CamState.from_dict(m["state"])
                timings = {}
                t0 = time.perf_counter()
                rgb = renderer.render(state, m["nx"], m["ny"], m["look"], timings, m.get("exposure"), check=check)
                timings["total"] = time.perf_counter() - t0
                timings["backend"] = scene.name
                conn.send({"type": "frame", "id": m["id"], "rgb": rgb, "info": timings, "keep": m.get("keep")})
            except Cancelled:
                conn.send({"type": "cancelled", "id": m["id"]})
            except Exception:
                conn.send({"type": "error", "id": m.get("id"), "msg": traceback.format_exc()})
