# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Editor tests, offscreen. The shamrock render process is replaced by a stub, so they need
neither shamrock nor a GPU: they cover the GUI logic on the synthetic dumps.
"""

import argparse
import os
import time

import numpy as np
import pytest
from conftest import sink_pos

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import QtCore, QtGui  # noqa: E402
from PyQt6.QtCore import QPointF, Qt, pyqtSignal  # noqa: E402

pytestmark = pytest.mark.gui


class StubBackend(QtCore.QObject):
    """Stands for gui.workers.Backend: records the render requests, never answers."""

    loaded = pyqtSignal(object)
    frame = pyqtSignal(object)
    error = pyqtSignal(object)
    status = pyqtSignal(str)
    cancelled = pyqtSignal(object)

    def __init__(self, sycl, sample_n, renderer="auto"):
        super().__init__()
        self.cancels = 0
        self.requests = []
        self.busy = False
        self.tracks_path = None
        self.name = "stub"

    def request(self, state, dump, nx, ny, look, keep=False, exposure=None, fmt=None):
        self.requests.append(dict(state=state.to_dict(), dump=dump, nx=nx, ny=ny, look=dict(look), keep=keep, exposure=exposure,
                                  fmt=fmt))  # fmt: skip
        return len(self.requests)

    def answer(self, i=-1):
        """Reply to request i with a blank frame, as the render process would."""
        r = self.requests[i]
        rid = len(self.requests) + i + 1 if i < 0 else i + 1
        rgb = np.zeros((r["ny"], r["nx"], 3), np.uint8)
        self.frame.emit({"id": rid, "rgb": rgb, "info": {"total": 0.1, "backend": "stub"}, "keep": r["keep"]})

    def cancel(self):
        self.cancels += 1

    def send_tracks(self, path):
        self.tracks_path = path

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp(tmp_path_factory):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("config"))  # QSettings of the tests only
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(app, cond, tmax=20.0):
    t = time.perf_counter()
    while not cond() and time.perf_counter() - t < tmax:
        app.processEvents()
        time.sleep(0.002)
    return cond()


def make_editor(qapp, sim, monkeypatch, path=None):
    from plouf.gui import app as gui_app

    monkeypatch.setattr(gui_app, "Backend", StubBackend)
    args = argparse.Namespace(dumps=sim.dumps, path=path, sycl="0:0", sample=1000, keyboard="qwerty", format=None,
                              proxy_cache=100, backend="auto")  # fmt: skip
    ed = gui_app.Editor(args)
    ed.resize(1200, 800)
    ed.show()
    return ed


def close_editor(qapp, ed):
    ed.commit()
    ed.saved = ed.history[ed.hpos]  # no "save?" dialog
    ed.close()
    ed.deleteLater()
    qapp.processEvents()


@pytest.fixture
def editor(qapp, sim, monkeypatch):
    """The movie editor."""
    ed = make_editor(qapp, sim, monkeypatch)
    ed.set_mode("movie")
    assert wait(qapp, lambda: ed.path.tracks is not None and ed.sample is not None)
    yield ed
    close_editor(qapp, ed)


@pytest.fixture
def viewer(qapp, sim, monkeypatch):
    """The editor in its default mode: view / navigate."""
    ed = make_editor(qapp, sim, monkeypatch)
    vm = ed.viewmode
    assert wait(qapp, lambda: ed.path.tracks is not None and vm.sample is not None)
    yield vm
    close_editor(qapp, ed)


def test_startup_reads_tracks_and_proxy(editor, sim):
    assert editor.path.tracks.nsink == 4
    assert editor.proxy_path == sim.dumps[0]
    assert len(editor.sample) == 1000
    assert editor.live_dirty and not editor.path.keyframes  # a first camera framing the gas
    assert editor.backend.tracks_path is not None  # the render process gets the tracks too
    assert editor.backend.requests  # a first preview was asked


def test_key_delete_undo_redo(editor):
    editor.key_current()
    editor.set_time(1.0)
    editor.live.dolly(0.5)
    editor.live_changed()
    editor.key_current()
    assert [k.t for k in editor.path.keyframes] == [0.0, 1.0]
    editor.delete_key(editor.key_index())
    assert len(editor.path.keyframes) == 1
    editor.undo()
    assert len(editor.path.keyframes) == 2
    editor.redo()
    assert len(editor.path.keyframes) == 1


def test_editing_on_a_key_changes_the_key(editor):
    editor.key_current()
    editor.live.fov = 30.0
    editor.live_changed()
    assert editor.path.keyframes[0].state.fov == 30.0 and not editor.live_dirty


def test_play_to_next_key_stops_on_it(qapp, editor):
    editor.key_current()
    editor.set_time(0.3)
    editor.live.dolly(0.8)
    editor.key_current()
    editor.set_time(0.0)
    editor.play_to_next_key()
    assert editor.playing
    assert wait(qapp, lambda: not editor.playing, 5)
    assert editor.t == 0.3 and editor.key_index() == 1


def test_select_and_follow(editor):
    editor.live.dump = 1.0
    editor.select_sink(0)
    assert "sink #0" in editor.sink_label(0)
    editor.path.look["sinks"] = "cluster"
    assert "star #0" in editor.sink_label(0)
    editor.follow("look")
    assert editor.live.follow == 0 and not editor.live.ride
    np.testing.assert_allclose(editor.live.target, sink_pos(0, 1.0), atol=1e-6)
    off = editor.live.pos - editor.live.target
    editor.follow("ride")
    assert editor.live.ride
    np.testing.assert_allclose(editor.live.pos - editor.live.target, off, atol=1e-9)
    editor.follow("stop")
    assert editor.live.follow is None


def test_click_selects_a_sink(qapp, editor):
    vf = editor.viewfinder
    st = editor.live
    st.target = sink_pos(2, st.dump)
    editor.live_changed()
    qapp.processEvents()
    from plouf.gui.qtutil import fit_rect

    r = fit_rect(vf.width(), vf.height(), editor.aspect())
    c = QPointF(r.center())
    for kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease):
        ev = QtGui.QMouseEvent(kind, c, vf.mapToGlobal(c), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier)  # fmt: skip
        QtWidgets.QApplication.sendEvent(vf, ev)
    assert editor.selected == 2  # the core at the centre of the view


@pytest.mark.parametrize("layout, forward", [("qwerty", Qt.Key.Key_W), ("azerty", Qt.Key.Key_Z)])
def test_fly_keys_follow_the_keyboard_layout(editor, layout, forward):
    editor.set_keyboard(layout)
    d = editor.live.basis()[0]
    p0, dist = editor.live.pos.copy(), editor.live.distance
    ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, forward, Qt.KeyboardModifier.NoModifier, "")
    QtWidgets.QApplication.sendEvent(editor.viewfinder, ev)
    np.testing.assert_allclose(editor.live.pos - p0, d * 0.02 * dist, atol=1e-9)


def test_axis_gizmo_reorients_the_world_view(qapp, editor):
    w = editor.world
    ends = {axis: e for e, _, axis, _, _ in w._gizmo_ends()}
    pt = ends[(2, 1)]  # +Z end
    for kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease):
        ev = QtGui.QMouseEvent(kind, pt, w.mapToGlobal(pt), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier)  # fmt: skip
        QtWidgets.QApplication.sendEvent(w, ev)
    assert wait(qapp, lambda: not w._anim_timer.isActive(), 5)
    np.testing.assert_allclose(-w.view.basis()[0], [0, 0, 1], atol=1e-9)  # looking down from +Z


def test_movie_panel_edits_the_look(editor):
    editor.movie_insp.checks["star_psf"].setChecked(False)
    editor.movie_insp.checks["auto_exposure"].setChecked(True)
    assert editor.path.look["star_psf"] is False and editor.path.look["auto_exposure"] is True
    editor.request_refine()
    assert editor.backend.requests[-1]["look"]["star_psf"] is False
    assert not editor.movie_insp.spike.isEnabled()  # spikes are part of the PSF
    editor.movie_insp.checks["star_psf"].setChecked(True)
    editor.movie_insp.spike.setValue(22.5)
    assert editor.path.look["spike_angle"] == 22.5
    editor.movie_insp.glow.setValue(0.4)
    assert editor.path.look["glow_gamma"] == 0.4
    editor.movie_insp.label.setValue(60)
    assert editor.path.look["label_size"] == 60
    editor.movie_insp.checks["time_label"].setChecked(False)
    assert not editor.movie_insp.label.isEnabled()


def test_up_buttons_set_up_exactly(editor):
    editor.live.up = np.array([0.1, 0.2, 0.9])
    editor.cam_insp._set_up(2)
    np.testing.assert_array_equal(editor.live.up, [0, 0, 1])


def test_proxy_follows_the_dump(qapp, editor, sim):
    editor.key_current()
    editor.live.dump = 3
    editor.live_changed()
    assert wait(qapp, lambda: editor.proxy_path == sim.dumps[3])
    img = editor.viewfinder.proxy_image()
    assert img is not None and img.width() > 0


def test_retime_segment_from_the_editor_and_by_shift_dragging_a_key(qapp, editor):
    for t, dolly in ((0.0, None), (1.0, 0.7), (2.0, 0.7)):
        editor.set_time(t)
        if dolly:
            editor.live.dolly(dolly)
        editor.key_current()
    editor.set_time(0.5)
    editor.set_segment_duration(editor.path.segment_at(editor.t), 3.0)
    assert [k.t for k in editor.path.keyframes] == [0, 3, 4] and editor.path.duration == 12
    editor.undo()
    assert [k.t for k in editor.path.keyframes] == [0, 1, 2]

    tl = editor.timeline
    qapp.processEvents()
    x1, x2 = tl.tx(1.0), tl.tx(3.0)  # shift-drag key 2 from 1 s to 3 s: key 3 follows
    for kind, x, buttons in ((QtCore.QEvent.Type.MouseButtonPress, x1, Qt.MouseButton.LeftButton),
                             (QtCore.QEvent.Type.MouseMove, x2, Qt.MouseButton.LeftButton),
                             (QtCore.QEvent.Type.MouseButtonRelease, x2, Qt.MouseButton.NoButton)):  # fmt: skip
        pt = QPointF(x, 40)
        ev = QtGui.QMouseEvent(kind, pt, tl.mapToGlobal(pt), Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.ShiftModifier)
        QtWidgets.QApplication.sendEvent(tl, ev)
    assert [k.t for k in editor.path.keyframes] == pytest.approx([0, 3, 4])
    assert editor.path.keyframes[0].stretch == pytest.approx(3.0)
    assert editor.history[editor.hpos] == editor.path.to_dict()  # committed for undo


def test_render_dialog_stereo_option(qapp):
    from plouf.gui.inspectors import RenderDialog

    dlg = RenderDialog(None, "/tmp/out", 10)
    assert not dlg.angle.isEnabled() and dlg.mp4.isEnabled()
    dlg.stereo.setChecked(True)
    assert dlg.angle.isEnabled() and not dlg.mp4.isEnabled() and dlg.angle.value() == 4.0


def test_a_camera_path_picked_in_the_dumps_dialog_is_opened(qapp, editor, sim, tmp_path, monkeypatch):
    """The dialog shown at startup without dumps: picking a path file opens it, not as a dump."""
    from plouf.path.camera import CamState
    from plouf.path.campath import CameraPath, Keyframe

    p = CameraPath(keyframes=[Keyframe(0, CamState(pos=[0, -3, 1], target=[0, 0, 0], up=[0, 0, 1], dump=2))], dumps=sim.dumps[:3])
    f = str(tmp_path / "p.json")
    p.save(f)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([f], "")))
    editor.add_dumps()
    assert editor.path_file == f and editor.path.dumps == sim.dumps[:3] and len(editor.path.keyframes) == 1
    assert wait(qapp, lambda: editor.backend.requests[-1]["dump"] == sim.dumps[2], 5)  # rendered from its dumps

    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([f, sim.dumps[0]], "")))
    editor.add_dumps()
    assert warned and editor.path.dumps == sim.dumps[:3]  # mixed selection: nothing changes


# ---------------------------------------------------------------------- view / navigate mode


def test_starts_in_view_mode(viewer, sim):
    ed = viewer.ed
    assert ed.mode == "view" and ed.stack.currentWidget() is viewer
    assert ed.toolbar.isHidden() and not ed.movie_menus[0].menuAction().isVisible()
    assert all(not a.isEnabled() for a in ed.movie_actions) and all(not s.isEnabled() for s in ed.movie_shortcuts)
    assert viewer.proxy_path == sim.dumps[0] and viewer.live.dump == 0
    assert ed.sample is None  # the movie editor's proxy is not read until it is shown


def test_path_option_starts_in_the_movie_editor(qapp, sim, monkeypatch, tmp_path):
    ed = make_editor(qapp, sim, monkeypatch, path=str(tmp_path / "new.json"))
    try:
        assert ed.mode == "movie" and ed.stack.currentWidget() is ed.movie_page
        assert all(a.isEnabled() for a in ed.movie_actions) and not any(a.isEnabled() for a in ed.viewmode.actions.values())
    finally:
        close_editor(qapp, ed)


def test_view_keys_are_splash_like_and_follow_the_layout(viewer):
    keys = {n: [k.toString() for k in a.shortcuts()] for n, a in viewer.actions.items()}
    assert "Space" in keys["next"] and "B" in keys["prev"] and "P" in keys["play"]
    assert keys["view1"] == ["1"]
    viewer.ed.set_keyboard("azerty")
    assert [k.toString() for k in viewer.actions["view1"].shortcuts()] == ["1", "&"]


def test_step_through_the_dumps(qapp, viewer, sim):
    viewer.actions["next"].trigger()
    assert viewer.dump_index() == 1
    assert wait(qapp, lambda: viewer.proxy_path == sim.dumps[1])  # its gas is shown
    viewer.actions["prev"].trigger()
    assert viewer.dump_index() == 0
    viewer.actions["next10"].trigger()
    assert viewer.dump_index() == 4  # clamped to the last dump
    viewer.actions["first"].trigger()
    assert viewer.dump_index() == 0
    assert viewer.panel.slider.value() == 0


def test_space_and_b_keys(qapp, viewer):
    from PyQt6.QtTest import QTest

    ed = viewer.ed
    ed.activateWindow()
    if not QTest.qWaitForWindowActive(ed, 2000):
        pytest.skip("no active window on this Qt platform: shortcuts are not delivered")
    QTest.keyClick(viewer.canvas, Qt.Key.Key_Space)
    QTest.keyClick(viewer.canvas, Qt.Key.Key_Space)
    assert viewer.dump_index() == 2
    QTest.keyClick(viewer.canvas, Qt.Key.Key_B)
    assert viewer.dump_index() == 1
    assert not ed.playing  # the movie editor's Space is off in the view mode


def test_zoom_and_axis_views(viewer):
    fov = viewer.live.fov
    viewer.actions["zoom_in"].trigger()
    assert viewer.live.fov < fov
    d = viewer.live.distance
    viewer.actions["view7"].trigger()
    np.testing.assert_allclose(viewer.live.basis()[0], [0, 0, -1], atol=1e-12)  # looking down from +Z
    assert viewer.live.distance == pytest.approx(d)


def test_follow_a_sink_through_the_dumps(viewer):
    viewer.select_sink(0)
    viewer.follow("ride")
    off = viewer.live.pos - viewer.live.target
    np.testing.assert_allclose(viewer.live.target, sink_pos(0, 0), atol=1e-6)
    viewer.step_dump(2)
    np.testing.assert_allclose(viewer.live.target, sink_pos(0, 2), atol=1e-6)  # the star stays in the centre
    np.testing.assert_allclose(viewer.live.pos - viewer.live.target, off, atol=1e-9)
    viewer.actions["look_at"].trigger()  # T while following: stop
    assert viewer.live.follow is None
    viewer.actions["look_at"].trigger()
    assert viewer.live.follow == 0 and not viewer.live.ride
    p = viewer.live.pos.copy()
    viewer.step_dump(1)
    np.testing.assert_allclose(viewer.live.pos, p)  # look at: the camera stays, it turns
    np.testing.assert_allclose(viewer.live.target, sink_pos(0, 3), atol=1e-6)


def test_play_through_the_dumps_stops_at_the_end(qapp, viewer):
    viewer.panel.rate.setValue(60)
    viewer.actions["play"].trigger()
    assert viewer.playing and viewer.panel.buttons["play"].text() == "⏸"
    assert wait(qapp, lambda: not viewer.playing, 10)
    assert viewer.dump_index() == 4


def test_modes_have_independent_cameras(qapp, viewer):
    ed = viewer.ed
    ed.set_mode("movie")
    assert wait(qapp, lambda: ed.sample is not None)
    movie = ed.live.copy()
    ed.set_mode("view")
    viewer.live.dolly(0.3)
    viewer.step_dump(3)
    viewer.live_changed()
    assert ed.live.same_view(movie)
    ed.set_mode("movie")
    ed.live.fov = 20.0
    ed.live_changed()
    assert viewer.live.fov != 20.0 and viewer.dump_index() == 3


def test_view_renders_and_saves_images(qapp, viewer, sim, tmp_path, monkeypatch):
    be = viewer.backend
    viewer.step_dump(2)
    viewer.request_refine()
    r = be.requests[-1]
    assert r["dump"] == sim.dumps[2] and r["fmt"] == "phantom" and not r["keep"]
    assert r["nx"] == int(viewer.canvas.width() * viewer.render_scale)
    be.answer()
    assert viewer.preview_valid()
    viewer.toggle_help()
    assert not viewer.canvas.grab().isNull()  # the HUD and the key help paint

    f = str(tmp_path / "view.png")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (f, "")))
    viewer.panel.width.setValue(320)
    viewer.actions["save_image"].trigger()
    r = be.requests[-1]
    assert r["keep"] and r["nx"] == 320 and r["ny"] == round(320 / viewer.aspect())
    be.answer()
    assert os.path.exists(f)


def test_opening_a_path_switches_to_the_movie_editor(viewer, sim, tmp_path):
    from plouf.path.camera import CamState
    from plouf.path.campath import CameraPath, Keyframe

    p = CameraPath(keyframes=[Keyframe(0, CamState(pos=[0, -3, 1], target=[0, 0, 0], up=[0, 0, 1]))], dumps=sim.dumps)
    f = str(tmp_path / "p.json")
    p.save(f)
    viewer.ed.open_path(f)
    assert viewer.ed.mode == "movie" and viewer.ed.mode_actions["movie"].isChecked()


def test_proxy_follows_the_view_size_and_is_coarser_while_moving(qapp, viewer):
    from plouf.gui.qtutil import fit_rect
    from plouf.gui.viewfinder import PROXY_FAST_PIXELS, PROXY_PIXELS

    cv = viewer.canvas
    viewer.ed.resize(1800, 1000)
    qapp.processEvents()
    rect = fit_rect(cv.width(), cv.height(), viewer.aspect())
    w, h = cv.proxy_size(rect, False)
    dpr = cv.devicePixelRatioF()
    assert abs(w - min(rect.width() * dpr, (PROXY_PIXELS * rect.width() / rect.height()) ** 0.5)) <= 1  # not 480 px
    img = cv.proxy_image(rect)
    assert img.width() == w  # at rest
    viewer.live.dolly(0.9)
    img = cv.proxy_image(rect)  # the view changed: coarse
    assert img.width() * img.height() <= PROXY_FAST_PIXELS * 1.01 and cv._fast
    assert wait(qapp, lambda: not cv._fast, 3)  # settled
    assert cv.proxy_image(rect).width() == w


def test_sink_mode_in_the_panels(viewer):
    panel = viewer.panel
    assert viewer.look["sinks"] == "markers" and panel.star_racc.isHidden()
    panel.sinks.setCurrentIndex(panel.sinks.findData("cluster"))
    assert viewer.look["sinks"] == "cluster" and not panel.star_racc.isHidden()
    panel.star_racc.setValue(0.05)
    assert viewer.look["star_max_racc_pc"] == 0.05
    sk = viewer.display_sinks()
    assert sorted(sk["star_id"]) == [0, 1, 2]  # the 0.02 pc core is now under the threshold
    ed = viewer.ed
    assert ed.path.look["sinks"] == "markers"  # the movie's look is independent
    ed.movie_insp.sinks.setCurrentIndex(ed.movie_insp.sinks.findData("stars"))
    assert ed.path.look["sinks"] == "stars" and viewer.look["sinks"] == "cluster"


@pytest.mark.parametrize("mode, stars, discs, threshold, markers", [
    ("markers", False, True, False, True),
    ("cluster", True, True, True, False),
    ("stars", True, False, False, False),
    ("hidden", False, False, False, False),
])  # fmt: skip
def test_sink_panels_show_only_the_controls_of_the_mode(viewer, mode, stars, discs, threshold, markers):
    ed = viewer.ed
    ed.set_mode("movie")
    for panel, sl, expo in ((viewer.panel, viewer.panel.sink_look, viewer.panel.expo),
                            (ed.movie_insp, ed.movie_insp.sink_look, ed.cam_insp.expo)):  # fmt: skip
        sl.mode.setCurrentIndex(sl.mode.findData(mode))
        ed.cam_insp.refresh()
        shown = lambda w: not w.isHidden()  # noqa: E731
        assert shown(sl.star_row) == shown(sl.spike) == shown(sl.glow) == shown(sl.psf) == stars
        assert shown(sl.radius) == shown(sl.discs) == discs
        assert shown(sl.thr) == threshold and shown(sl.marker_size) == markers
        assert shown(expo) == stars  # star exposure only with stars
        if discs:
            assert sl.discs.text() == ("draw markers" if markers else "draw cores")


def test_view_panel_spikes(viewer):
    sl = viewer.panel.sink_look
    sl.mode.setCurrentIndex(sl.mode.findData("stars"))
    sl.checks["star_spikes"].setChecked(False)
    assert viewer.look["star_spikes"] is False and not sl.spike.isEnabled()
    sl.checks["star_spikes"].setChecked(True)
    sl.spike.setValue(30.0)
    assert viewer.look["spike_angle"] == 30.0 and sl.spike.isEnabled()
    viewer.request_refine()
    r = viewer.backend.requests[-1]["look"]
    assert r["star_spikes"] and r["spike_angle"] == 30.0 and r["sinks"] == "stars"
    sl.checks["star_psf"].setChecked(False)
    assert not sl.checks["star_spikes"].isEnabled() and not sl.spike.isEnabled()  # spikes are part of the PSF


def test_disc_radius_hint_follows_the_sink_mode(editor):
    editor.sinks = {"id": np.array([0, 1]), "pos": np.zeros((2, 3)), "mass": np.ones(2), "racc": np.array([1e-4, 0.02])}
    editor.path.tracks = None  # the loaded dump's sinks (pc units here)
    assert "4.13e+03 au" in editor.disc_radius_hint({**editor.path.look, "sinks": "cluster"})  # the core only
    assert "2.07e+03 au" in editor.disc_radius_hint({**editor.path.look, "sinks": "markers"})  # median of both


def test_a_render_for_a_view_that_changed_is_cancelled(qapp, viewer):
    """Moving (or stepping dumps) while a render is computing drops it instead of letting it compete."""
    be = viewer.backend
    viewer.request_refine()
    rid = len(be.requests)
    assert rid in viewer.pending
    c0 = be.cancels  # the startup render, before the camera was framed, may have been dropped already
    viewer.canvas.repaint()
    assert be.cancels == c0  # still the view it is for
    viewer.step_dump(1)
    viewer.canvas.repaint()
    viewer.canvas.repaint()
    assert be.cancels == c0 + 1  # once
    be.cancelled.emit({"id": rid})  # the render process answers
    assert not viewer.pending
    n = len(be.requests)
    assert wait(qapp, lambda: len(be.requests) > n, 3)  # asked again once the view settles
    assert be.requests[-1]["dump"] == viewer.dump_path()
