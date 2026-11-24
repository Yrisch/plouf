# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Gas colours, proxy splats, cores and labels (render/compose.py)."""

import numpy as np
import pytest

from plouf.path.camera import CamState, pixel_size
from plouf.render.compose import compose, core_radius, draw_label, gas_rgb, lut, splat_column

LOOK = {"cmap": "inferno", "show_gas": True, "show_cores": True, "core_radius": None, "core_color": [1.0, 0.27, 0.0],
        "sinks": "cluster"}


def view():
    return CamState(pos=[0, -10, 0], target=[0, 0, 0], up=[0, 0, 1], clim=(0.0, 2.0))


def test_gas_rgb_maps_the_log_range_to_the_colormap():
    rgb = gas_rgb(np.array([[1e-5, 1.0, 100.0, 1e9]]), (0.0, 2.0), "inferno")
    np.testing.assert_array_equal(rgb[0, 0], lut("inferno")[0])
    np.testing.assert_array_equal(rgb[0, 1], lut("inferno")[0])
    np.testing.assert_array_equal(rgb[0, 2], lut("inferno")[255])
    np.testing.assert_array_equal(rgb[0, 3], lut("inferno")[255])


def test_splat_conserves_mass():
    """Particles in the plane of the target: sum of column x pixel area = total mass."""
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-1, 1, 5000), np.zeros(5000), rng.uniform(-1, 1, 5000)]
    st = view()
    img = splat_column(pts, 0.01, st, 200, 100, smooth=0.7)
    area = pixel_size(st, 10.0, 100) ** 2
    assert img.sum() * area == pytest.approx(50.0, rel=1e-6)


def test_core_radius_physical_or_override():
    sinks = {"core_racc": np.array([0.02, 0.03])}
    np.testing.assert_array_equal(core_radius(LOOK, sinks), [0.02, 0.03])
    assert core_radius(dict(LOOK, core_radius=0.3), sinks) == 0.3
    np.testing.assert_array_equal(core_radius(dict(LOOK, core_radius=0.0), sinks), [0.02, 0.03])


def test_compose_draws_cores_stars_and_label():
    st = view()
    sigma = np.full((100, 200), 10.0)
    sinks = {"core_pos": np.array([[0.0, 0.0, 0.0]]), "core_racc": np.array([0.5])}
    out = compose(sigma, st, LOOK, sinks, None, None)
    assert out.shape == (100, 200, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[50, 100], [255, 69, 0])  # core colour at the centre
    assert (out[5, 5] == np.round(lut("inferno")[127] * 255)).all()  # gas half way in the log range
    stars = np.zeros((100, 200, 3))
    stars[10, 10] = 1.0
    assert (compose(sigma, st, LOOK, sinks, stars, None)[10, 10] == 255).all()  # screen blend
    dots = np.zeros((100, 200, 4))
    dots[10, 10] = [0.5, 0.25, 0.0, 0.5]  # premultiplied RGBA: half opaque (1, 0.5, 0)
    gas = lut("inferno")[127].astype(float)
    expected = np.round((gas * 0.5 + [0.5, 0.25, 0.0]) * 255 + 1e-9)
    np.testing.assert_allclose(compose(sigma, st, LOOK, sinks, dots, None)[10, 10], expected, atol=1)  # drawn over
    lab = compose(sigma, st, dict(LOOK, label_size=200), None, None, "t = 1.00 Myr")  # 18 px on this 100 px frame
    changed = (lab != compose(sigma, st, LOOK, None, None, None)).any(axis=2)
    assert changed[60:, 100:].any() and not changed[:60].any()  # text bottom right
    assert lab[changed].max() > 240  # in white
    dark = compose(sigma, st, dict(LOOK, show_gas=False, show_cores=False), sinks, None, None)
    assert dark.max() == 0


def test_label_size_is_a_parameter_scaled_with_the_frame_height():
    """label_size px on a 1080 px tall frame: twice the size -> twice as tall, same look in a draft."""

    def text_height(ny, size):
        img = draw_label(np.zeros((ny, 2 * ny, 3), np.uint8), "t = 1.00 Myr", size)
        rows = np.flatnonzero(img.max(axis=(1, 2)) > 128)
        return rows[-1] - rows[0] + 1

    h = text_height(1080, 36)
    assert h > 20  # a scalable font, not the 11 px bitmap fallback
    assert text_height(1080, 72) == pytest.approx(2 * h, rel=0.1)
    assert text_height(540, 36) == pytest.approx(h / 2, rel=0.1)


def test_markers_have_a_minimum_size_and_their_colour():
    st = view()
    sigma = np.full((108, 192), 1e-6)  # no gas visible
    sinks = {"core_pos": np.array([[0.0, 0.0, 0.0]]), "core_racc": np.array([1e-6])}  # a tiny accretion radius
    look = dict(LOOK, sinks="markers", marker_size=10.0, marker_color=[0.0, 1.0, 0.0])
    out = compose(sigma, st, look, sinks, None, None)
    green = (out[..., 1] > 200) & (out[..., 0] < 50)
    assert 1 <= green.sum() <= 8  # 10 px radius on a 1080 px frame: 1 px here, still visible
    big = compose(np.full((1080, 1920), 1e-6), st, look, sinks, None, None)
    ys, xs = np.nonzero((big[..., 1] > 200) & (big[..., 0] < 50))
    assert 9 <= (xs.max() - xs.min() + 1) / 2 <= 11


def test_compose_without_star_layer_is_the_same_frame_as_with_the_float_path():
    """The fast path of the instant views (uint8 colormap, discs blended in their box) changes no pixel."""
    from plouf.render.compose import _quantize, draw_discs, gas_rgb, sink_discs

    st = view()
    rng = np.random.default_rng(0)
    sigma = 10 ** rng.uniform(-1, 4, (90, 160))
    sinks = {"core_pos": rng.normal(scale=0.5, size=(6, 3)), "core_racc": rng.uniform(0.01, 0.3, 6)}
    for look in (LOOK, dict(LOOK, sinks="markers"), dict(LOOK, show_gas=False)):
        ref = gas_rgb(sigma, st.clim, "inferno").astype(np.float64) if look["show_gas"] else np.zeros((90, 160, 3))
        draw_discs(ref, *sink_discs(st, look, sinks, 160, 90))
        np.testing.assert_array_equal(compose(sigma, st, look, sinks, None, None), _quantize(ref))


def test_proxy_splat_with_smoothing_lengths():
    """splat_sph: the column of the particles (mass kept), each as wide as its kernel on screen."""
    from plouf.render.compose import SPLAT_SIGMA_H, splat_sph

    st = view()
    for h in (1e-4, 0.05, 0.4):  # below a pixel, a few pixels, tens of pixels (coarse grids)
        img = splat_sph(np.zeros((1, 3)), np.array([h]), 2.0, st, 320, 180)
        assert img.sum() * pixel_size(st, st.distance, 180) ** 2 == pytest.approx(2.0, rel=0.01)
        sig_px = max(SPLAT_SIGMA_H * h / pixel_size(st, st.distance, 180), 0.7)
        prof = img[90]
        width = np.sqrt(np.sum(prof * (np.arange(320) + 0.5 - 160) ** 2) / prof.sum())
        assert width == pytest.approx(np.sqrt(sig_px**2 + 1 / 12), rel=0.25)  # Gaussian of the kernel's width
