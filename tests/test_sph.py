# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""The numpy SPH renderer (render/sph.py), the fallback without shamrock."""

import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pytest
from conftest import NGAS, sink_pos

from plouf.path.camera import CamState, pixel_size, project
from plouf.render import sph

CAM = CamState(pos=[0, -20, 0], target=[0, 0, 0], up=[0, 0, 1], fov=20)


def pixel_rays(state, nx, ny):
    """Unit direction of the ray through each pixel centre, (ny, nx, 3)."""
    d, r, u = state.basis()
    th = math.tan(math.radians(state.fov) / 2)
    ys, xs = np.mgrid[0:ny, 0:nx] + 0.5
    X = (2 * xs / nx - 1 + state.shift) * th * nx / ny
    Y = (1 - 2 * ys / ny) * th
    ray = d + X[..., None] * r + Y[..., None] * u
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def ray_distance(p, ray):
    """Distance from points p (n, 3) relative to the camera to the lines along the unit rays (..., 3), (n, ...)."""
    return np.linalg.norm(np.cross(p[:, None, :], ray.reshape(-1, 3)[None]), axis=-1).reshape((len(p),) + ray.shape[:-1])


def exact_column(pos, h, state, nx, ny):
    """Every particle's kernel integrated along every pixel's ray (no normalisation, no levels)."""
    rays = pixel_rays(state, nx, ny)
    rel = np.asarray(pos, float) - state.pos
    out = np.zeros((ny, nx))
    for i in range(len(rel)):
        out += sph.kernel_column(ray_distance(rel[i : i + 1], rays)[0] / h[i]) / h[i] ** 2
    return out


def test_column_kernel_integrates_to_one():
    q = np.linspace(0, sph.SUPPORT, 4001)
    F = sph.kernel_column(q)
    assert np.trapezoid(F * 2 * np.pi * q, q) == pytest.approx(1.0, abs=1e-4)
    assert F[-1] == 0 and np.all(np.diff(F) <= 1e-12)  # decreasing to zero at the support
    # through the centre: twice the integral of the M4 kernel along a radius, 3 / (2 pi) h^-2
    assert sph.kernel_column(0.0) == pytest.approx(3 / (2 * np.pi), rel=1e-3)


@pytest.mark.parametrize("h", [1e-4, 0.005, 0.05, 0.3, 1.0])
def test_a_particle_deposits_its_mass_at_any_size(h):
    """From far below a pixel to several coarse levels: the image holds the particle's mass."""
    img = sph.column_render(np.array([[0.13, 0.0, 0.07]]), np.array([h]), 2.5, CAM, 320, 180)
    # exact for the small (normalised) ones; large ones are point sampled (and spline upsampled on coarse levels)
    assert img.sum() * pixel_size(CAM, 20.0, 180) ** 2 == pytest.approx(2.5, rel=1e-3 if h < 0.1 else 5e-3)


@pytest.mark.parametrize("h", [0.2, 2.0, 5.0, 20.0])
def test_large_particles_match_the_exact_kernel(h):
    """Kernels drawn on the coarse levels (or the whole coarsest one) and upsampled."""
    p = np.array([[0.3, 0.0, 0.2]])
    img = sph.column_render(p, np.array([h]), 1.0, CAM, 320, 180)
    ref = exact_column(p, [h], CAM, 320, 180)
    assert np.abs(img - ref).max() < 0.01 * ref.max()


def test_resolved_particles_match_the_exact_kernel():
    rng = np.random.default_rng(1)
    pos = rng.normal(scale=1.0, size=(1500, 3))
    h = 0.1 * np.exp(rng.normal(scale=1.0, size=1500))
    sx, sy, z = project(CAM, pos, 320, 180)
    keep = h / pixel_size(CAM, z, 180) >= 2  # point sampling is only exact for resolved kernels
    pos, h = pos[keep], h[keep]
    img = sph.column_render(pos, h, 1.0, CAM, 320, 180)
    ref = exact_column(pos, h, CAM, 320, 180)
    assert np.abs(img - ref).max() < 0.01 * ref.max()


def test_lens_shift_translates_the_image():
    """A stereo eye (CamState.shift, whole pixels) sees the gas moved as project() moves points."""
    rng = np.random.default_rng(2)
    pos = rng.normal(scale=1.0, size=(3000, 3))
    h = np.full(3000, 0.08)
    a = sph.column_render(pos, h, 1.0, CAM, 160, 90)
    st = CAM.copy()
    st.shift = 2 * 6 / 160  # 6 px: what was at column j + 6 is now at column j
    b = sph.column_render(pos, h, 1.0, st, 160, 90)
    np.testing.assert_allclose(b[:, :-6], a[:, 6:], rtol=1e-9, atol=1e-12 * a.max())


@pytest.mark.parametrize("fov", [20, 70])
def test_column_to_points_matches_brute_force(fov):
    rng = np.random.default_rng(3)
    pos = rng.normal(scale=1.0, size=(20000, 3))
    h = 0.05 * np.exp(rng.normal(scale=0.8, size=20000))
    st = CamState(pos=[0, -4, 0], target=[0, 0, 0], up=[0, 0, 1], fov=fov)
    pts = rng.normal(scale=0.7, size=(40, 3))
    got = sph.column_to_points(pos, h, 1.0, st, pts)
    d = st.basis()[0]
    rel = pos - st.pos
    z = rel @ d
    ref = []
    for p in pts:
        ray = (p - st.pos) / np.linalg.norm(p - st.pos)
        b = np.linalg.norm(np.cross(rel, ray), axis=1)
        ok = (z > 1e-3) & (z < (p - st.pos) @ d)
        ref.append(np.sum(sph.kernel_column(b[ok] / h[ok]) / h[ok] ** 2))
    np.testing.assert_allclose(got, ref, rtol=1e-3)  # the renderer's kernel table vs np.interp
    assert got.max() > 0


def test_wide_angle_columns_are_exact_off_axis():
    """Off the view axis the footprint is stretched radially: rays, not the image plane, decide."""
    rng = np.random.default_rng(5)
    st = CamState(pos=[0, -3, 0], target=[0, 0, 0], up=[0, 0, 1], fov=90)
    pos = rng.uniform(-2.5, 2.5, size=(400, 3)) * [1, 0.3, 1]
    h = rng.uniform(0.05, 0.4, size=400)
    sx, sy, z = project(st, pos, 160, 90)
    keep = h / pixel_size(st, z, 90) >= 2  # resolved (point sampling is exact for those)
    img = sph.column_render(pos[keep], h[keep], 1.0, st, 160, 90)
    ref = exact_column(pos[keep], h[keep], st, 160, 90)
    assert np.abs(img - ref).max() < 0.01 * ref.max()
    np.testing.assert_allclose(img.sum(), ref.sum(), rtol=3e-3)


# ---------------------------------------------------------------------- scene on the synthetic dumps


@pytest.fixture
def scene(sim):
    sc = sph.NumpyScene()
    assert sc.load(sim.dumps[1]) and not sc.load(sim.dumps[1])  # loaded once
    return sc


def test_scene_reads_the_dump(scene, sim):
    assert scene.loaded and scene.npart == NGAS - 150
    assert scene.time_myr == pytest.approx(0.1)  # code units are pc, Msun, Myr
    assert scene.pc_per_unit == pytest.approx(1.0) and scene.sigma_unit == pytest.approx(1.0)
    sk = scene.sinks
    assert list(sk["id"]) == [0, 1, 2]  # sink 3 forms at dump 2
    np.testing.assert_allclose(sk["pos"][0], sink_pos(0, 1), atol=1e-12)
    assert sk["mass"][2] == pytest.approx(50.0)
    xyz, w = scene.gas_sample(500)
    assert xyz.shape == (500, 3) and w * 500 == pytest.approx(scene.npart * 1e-3)


def test_scene_column_holds_the_gas_mass(scene):
    """Seen from far, the whole ball is in the frame: the image holds the living gas mass [Msun]."""
    st = CamState(pos=[0, -60, 0], target=[0, 0, 0], up=[0, 0, 1], fov=12)
    sig = scene.column(st, 200, 200)  # Msun / pc^2
    assert sig.sum() * pixel_size(st, 60.0, 200) ** 2 == pytest.approx(scene.npart * 1e-3, rel=0.02)


def test_frame_renderer_with_the_numpy_scene(scene):
    from plouf.path.campath import DEFAULT_LOOK
    from plouf.render.backend import FrameRenderer

    st = CamState(pos=[0, -4, 1], target=[0, 0, 0], up=[0, 0, 1], extinction=1.0)
    tm = {}
    img = FrameRenderer(scene).render(st, 96, 54, dict(DEFAULT_LOOK, sinks="stars"), tm)
    assert img.shape == (54, 96, 3) and img.dtype == np.uint8 and img.max() > 0
    assert "stars" in tm  # stars with extinction through the gas


def test_make_scene(monkeypatch):
    from plouf.render import backend

    assert isinstance(backend.make_scene("numpy"), sph.NumpyScene)

    def no_shamrock(cfg="0:0", loglevel=0):
        raise ImportError("No module named 'shamrock'")

    monkeypatch.setattr(backend, "init_shamrock", no_shamrock)
    assert backend.make_scene("auto").name == "numpy"  # the fallback
    with pytest.raises(ImportError):
        backend.make_scene("shamrock")
    with pytest.raises(ValueError):
        backend.make_scene("povray")


def test_render_process_without_shamrock(sim):
    """The editor's render process with the numpy backend: spawned from the package, answers a render."""
    from plouf.path.campath import DEFAULT_LOOK
    from plouf.render.backend import server_main

    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    proc = ctx.Process(target=server_main, args=(child, "0:0", 500, "numpy"), daemon=True)
    proc.start()
    try:
        st = CamState(pos=[0, -4, 1], target=[0, 0, 0], up=[0, 0, 1])
        conn.send({"cmd": "render", "id": 1, "dump": sim.dumps[0], "state": st.to_dict(), "nx": 64, "ny": 36,
                   "look": dict(DEFAULT_LOOK), "format": "phantom"})  # fmt: skip
        got = {}
        while "frame" not in got and "error" not in got:
            assert conn.poll(60), "no answer from the render process"
            m = conn.recv()
            got[m["type"]] = m
        assert "error" not in got, got.get("error", {}).get("msg")
        assert got["backend"]["name"] == "numpy"
        assert got["loaded"]["sample"].shape == (500, 3)
        assert got["frame"]["rgb"].shape == (36, 64, 3) and got["frame"]["info"]["backend"] == "numpy"
        conn.send({"cmd": "quit"})
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.terminate()


def test_movie_render_without_shamrock(sim, tmp_path, monkeypatch):
    from plouf import render_movie
    from plouf.path.campath import CameraPath, Keyframe

    p = CameraPath(keyframes=[Keyframe(0, CamState(pos=[0, -4, 1], target=[0, 0, 0], up=[0, 0, 1], dump=0)),
                              Keyframe(1, CamState(pos=[1, -4, 1], target=[0, 0, 0], up=[0, 0, 1], dump=4))],
                   duration=1, fps=3, resolution=(64, 36), dumps=sim.dumps)  # fmt: skip
    f = tmp_path / "p.json"
    p.save(f)
    out = tmp_path / "frames"
    monkeypatch.setattr(sys, "argv", ["plouf render", str(f), "--out", str(out), "--backend", "numpy"])
    render_movie.main()
    assert sorted(os.listdir(out)) == [f"frame_{i:05d}.png" for i in range(p.n_frames)]


def test_huge_particle_and_camera_inside_the_gas():
    """Close-ups: kernels much wider than the frame, particles around the camera; finite, non-negative."""
    rng = np.random.default_rng(4)
    pos = rng.normal(scale=0.3, size=(5000, 3))
    h = np.full(5000, 0.2)
    st = CamState(pos=[0, -0.05, 0], target=[0, 0, 0], up=[0, 0, 1], fov=70)
    img = sph.column_render(pos, h, 1.0, st, 128, 72)
    assert np.all(np.isfinite(img)) and img.min() >= 0 and img.max() > 0
    assert math.isfinite(img.sum())


def test_render_process_drops_stale_renders_but_not_exports(sim):
    from plouf.path.campath import DEFAULT_LOOK
    from plouf.render.backend import server_main

    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    gen = ctx.Value("i", 1)  # the view changed since generation 0
    proc = ctx.Process(target=server_main, args=(child, "0:0", 500, "numpy", gen), daemon=True)
    proc.start()
    try:
        st = CamState(pos=[0, -4, 1], target=[0, 0, 0], up=[0, 0, 1]).to_dict()
        base = {"cmd": "render", "dump": sim.dumps[0], "state": st, "nx": 32, "ny": 18, "look": dict(DEFAULT_LOOK)}
        conn.send({**base, "id": 1, "gen": 0})
        conn.send({**base, "id": 2, "gen": 0, "keep": True})  # an export: always finished
        got = {}
        while not {1, 2} <= got.keys():
            assert conn.poll(60), "no answer from the render process"
            m = conn.recv()
            if "id" in m and m["type"] in ("frame", "cancelled", "error"):
                got[m["id"]] = m
        assert got[1]["type"] == "cancelled"
        assert got[2]["type"] == "frame"
        conn.send({"cmd": "quit"})
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.terminate()


def test_the_numpy_renderer_checks_for_cancellation_while_rendering():
    from plouf.render.backend import Cancelled

    rng = np.random.default_rng(6)
    pos = rng.normal(size=(200_000, 3))
    h = 0.05 * np.exp(rng.normal(scale=0.7, size=200_000))
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 3:
            raise Cancelled

    with pytest.raises(Cancelled):
        sph.column_render(pos, h, 1.0, CAM, 640, 360, check=check)
    assert len(calls) == 3  # stopped at the third check, mid-frame
