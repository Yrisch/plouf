# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
View / navigate mode: explore the dumps with a free camera, no keyframes.

A large view (ViewCanvas, a Viewfinder: same mouse / wheel / fly navigation as in the
movie editor) and a side panel (ViewPanel): dump navigation, camera, display. Keys are
in keymaps.VIEW_KEYS (Space / B next / previous dump, as in splash).

The mode has its own camera, look and selection, independent from the movie editor. It
shares the editor's data: dumps, proxy cache, sink tracks, render process. ViewMode
provides its canvas with the interface Viewfinder expects from the editor (live, aspect,
sample, preview, sinks...).
"""

import json
import math
import os

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor

from ..constants import MYR_S
from ..path.camera import CamState, normalize
from ..path.campath import DEFAULT_LOOK
from ..render.compose import save_png, splat_column
from .inspectors import CMAPS, SinkLook, dspin, has_stars
from .keymaps import keymap_hint, view_shortcuts
from .qtutil import np_to_qimage
from .viewfinder import Viewfinder

RENDER_SCALES = {"25 %": 0.25, "50 %": 0.5, "75 %": 0.75, "100 %": 1.0}  # render size / view size
AXIS_DIRS = {"1": (0, -1, 0), "3": (1, 0, 0), "7": (0, 0, 1)}  # camera side of the target: front, side, top

HELP = (  # (keys, action) rows of the key help (H); "{fly}" is the layout's fly keys
    ("Space / →", "next dump"),
    ("B / ←", "previous dump"),
    ("Shift+Space / Shift+B", "10 dumps forward / back"),
    ("Home / End", "first / last dump"),
    ("P", "play through the dumps (rate in the panel)"),
    ("LMB · Shift+LMB / MMB · RMB", "orbit · pan · look around"),
    ("wheel · Ctrl+wheel · Alt+wheel", "dolly · fov · dolly zoom"),
    ("+ / -", "zoom in / out (fov)"),
    ("1 / 3 / 7", "front / side / top view"),
    ("{fly}", "fly, roll (Shift ×5)"),
    ("O", "recentre on the gas"),
    ("C", "auto colour range"),
    ("click", "select a star / core"),
    ("T / Shift+T", "look at / ride along it through the dumps (T again: stop)"),
    ("Esc", "deselect"),
    ("G", "thirds grid"),
    ("Ctrl+E", "save image"),
    ("H", "this help"),
)


class ViewCanvas(Viewfinder):
    """The view of the view mode: a Viewfinder filling the widget, with its own HUD."""

    def look(self):
        return self.ed.look

    def _paint_hud(self, p, rect, mode):
        vm = self.ed
        f = p.font()
        f.setPointSizeF(9)
        p.setFont(f)
        p.setPen(QColor(220, 220, 220))
        p.drawText(QPointF(rect.left() + 8, rect.top() + 16), mode)
        p.drawText(QPointF(rect.left() + 8, rect.bottom() - 8), vm.info_text())
        hint = "H: keys"
        if vm.playing:
            hint = f"▶ playing {vm.play_rate:g} dumps/s · " + hint
        p.setPen(QColor(160, 160, 170))
        p.drawText(QPointF(rect.right() - p.fontMetrics().horizontalAdvance(hint) - 8, rect.top() + 16), hint)
        if vm.live.follow is not None:
            p.setPen(QColor(80, 230, 230))
            txt = f"◎ {vm.sink_label(vm.live.follow)} · {'ride along' if vm.live.ride else 'look at'}"
            p.drawText(QPointF(rect.right() - p.fontMetrics().horizontalAdvance(txt) - 8, rect.top() + 32), txt)
        if vm.show_help:
            self._paint_help(p, rect)

    def _paint_help(self, p, rect):
        rows = [(k.format(fly=keymap_hint(self.ed.keyboard)), t) for k, t in HELP]
        fm = p.fontMetrics()
        lh = fm.height() + 2
        kw = max(fm.horizontalAdvance(k) for k, _ in rows) + 24
        w = kw + max(fm.horizontalAdvance(t) for _, t in rows) + 24
        h = lh * len(rows) + 24
        box = QtCore.QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2, w, h)
        p.fillRect(box, QColor(0, 0, 0, 200))
        for i, (k, t) in enumerate(rows):
            y = box.top() + 12 + lh * (i + 0.8)
            p.setPen(QColor(255, 210, 60))
            p.drawText(QPointF(box.left() + 12, y), k)
            p.setPen(QColor(230, 230, 230))
            p.drawText(QPointF(box.left() + kw, y), t)


class ViewMode(QtWidgets.QWidget):
    def __init__(self, ed):
        super().__init__()
        self.ed = ed
        self.live = CamState(pos=[0, -40, 15], target=[0, 0, 0], up=[0, 0, 1])
        self.look = dict(DEFAULT_LOOK)
        self.sample = None
        self.sample_h = None
        self.weight = None
        self.sample_id = 0
        self.proxy_path = None
        self._framed = False  # camera framed on the gas once
        self.selected = None
        self.preview_img = None
        self.preview_info = {}
        self.preview_key = None
        self.pending = {}
        self.exports = {}
        self.render_scale = 0.5
        self.export_width = 1920
        self.playing = False
        self.play_rate = 4.0  # dumps per second
        self.show_help = False

        self.canvas = ViewCanvas(self)
        self.panel = ViewPanel(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(self.panel.sizeHint().width() + 24)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(scroll)

        self.refine_timer = QTimer(self, singleShot=True, interval=250)
        self.refine_timer.timeout.connect(self.request_refine)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self.actions = {}  # Navigate menu actions, by VIEW_KEYS name (built by the editor, see build_menu)

    # ------------------------------------------------------------------ menu and keys

    def build_menu(self, menu):
        """Fill the editor's Navigate menu: every view mode key is an action there."""
        items = (
            ("next", "Next dump", lambda: self.step_dump(1)),
            ("prev", "Previous dump", lambda: self.step_dump(-1)),
            ("next10", "Forward 10 dumps", lambda: self.step_dump(10)),
            ("prev10", "Back 10 dumps", lambda: self.step_dump(-10)),
            ("first", "First dump", lambda: self.set_dump(0)),
            ("last", "Last dump", lambda: self.set_dump(self.n_dumps - 1)),
            ("play", "Play / pause through the dumps", self.toggle_play),
            None,
            ("zoom_in", "Zoom in (narrower fov)", lambda: self.zoom(1 / 1.15)),
            ("zoom_out", "Zoom out (wider fov)", lambda: self.zoom(1.15)),
            ("view1", "Front view (from -Y)", lambda: self.axis_view("1")),
            ("view3", "Side view (from +X)", lambda: self.axis_view("3")),
            ("view7", "Top view (from +Z)", lambda: self.axis_view("7")),
            ("recentre", "Recentre on the gas", self.recentre),
            ("auto_clim", "Auto colour range", self.auto_clim),
            None,
            ("look_at", "Look at the selected sink / stop", lambda: self.follow("stop" if self.live.follow is not None else "look")),
            ("ride", "Ride along the selected sink", lambda: self.follow("ride")),
            ("deselect", "Deselect", lambda: self.select_sink(None)),
            None,
            ("grid", "Thirds grid", self.toggle_grid),
            ("help", "Key help", self.toggle_help),
            ("save_image", "Save image…", self.save_image),
        )
        for it in items:
            if it is None:
                menu.addSeparator()
                continue
            name, text, fn = it
            a = QtGui.QAction(text, self.ed)
            a.triggered.connect(fn)
            menu.addAction(a)
            self.ed.addAction(a)  # window-wide shortcut, also while the menu bar is hidden
            self.actions[name] = a
        self.update_shortcuts()

    def update_shortcuts(self):
        for name, keys in view_shortcuts(self.keyboard).items():
            if name in self.actions:
                self.actions[name].setShortcuts([QtGui.QKeySequence(k) for k in keys])

    def set_enabled(self, on):
        for a in self.actions.values():
            a.setEnabled(on)

    # ------------------------------------------------------------------ interface used by the canvas (see Viewfinder)

    @property
    def keyboard(self):
        return self.ed.keyboard

    @property
    def backend(self):
        return self.ed.backend

    @property
    def starlight(self):
        return self.ed.starlight

    def aspect(self):
        return max(self.canvas.width(), 1) / max(self.canvas.height(), 1)

    def preview_size(self, cap=None):
        w = max(16, int(self.canvas.width() * self.render_scale))
        if cap is not None:
            w = min(w, cap)
        return w, max(1, int(round(w / self.aspect())))

    def live_changed(self, from_panel=False):
        self.refine_timer.start()
        self.canvas.update()
        if not from_panel:
            self.panel.refresh()

    def commit(self):
        pass  # no undo history in the view mode

    def commit_soon(self):
        pass

    def display_sinks(self):
        return self.ed.display_sinks(self.live.dump, self.look)

    def pick_sink(self, state, w, h, x, y):
        return self.ed.pick_sink(state, w, h, x, y)

    def sink_position(self, sink):
        return self.ed.sink_position(sink, self.live.dump)

    def sink_label(self, sink):
        return self.ed.sink_label(sink, self.live.dump, self.look)

    def pc_per_unit(self):
        return self.ed.pc_per_unit()

    def live_exposure(self, nx, ny):
        """Star exposure: the camera's, plus the automatic part with auto exposure."""
        tr = self.ed.path.tracks
        if not self.look.get("auto_exposure", False) or tr is None:
            return self.live.exposure
        sk = self.display_sinks()
        lum, _ = self.starlight.luminosities(sk["star_mass"])
        e = self.starlight.exposure_for(self.live, sk["star_pos"], lum, nx, ny, tr.pc_per_unit)
        return self.live.exposure + (e or 0.0)

    def preview_valid(self):
        return self.preview_img is not None and self.preview_key == self.look_key()

    def proxy_status(self):
        if self.ed.proxy is None or not self.dumps:
            return ""
        cur = self.dump_path()
        if cur == self.proxy_path:
            return ""
        return f"gas of {os.path.basename(self.proxy_path) if self.proxy_path else '—'}, reading {os.path.basename(cur)}…"

    def select_sink(self, sink):
        self.selected = sink
        if sink is not None:
            self.ed.statusBar().showMessage(f"selected {self.sink_label(sink)}: T look at · Shift+T ride along", 5000)
        self.canvas.update()
        self.panel.refresh()

    # ------------------------------------------------------------------ dumps

    @property
    def dumps(self):
        return self.ed.path.dumps

    @property
    def n_dumps(self):
        return len(self.dumps)

    def dump_index(self):
        return int(np.clip(round(self.live.dump), 0, max(self.n_dumps - 1, 0)))

    def dump_path(self):
        return self.dumps[self.dump_index()] if self.dumps else None

    def dump_time_myr(self, i):
        tr = self.ed.path.tracks
        if tr is not None and i < tr.ndump and "utime" in tr.units:
            return tr.time[i] * tr.units["utime"] / MYR_S
        m = self.ed.meta
        if m and self.dumps and m["path"] == self.dumps[i]:
            return m["time_myr"]
        return None

    def info_text(self):
        if not self.dumps:
            return "no simulation dumps (File → Set simulation dumps…)"
        i = self.dump_index()
        txt = f"dump {i} / {self.n_dumps - 1}  ·  {os.path.basename(self.dumps[i])}"
        t = self.dump_time_myr(i)
        if t is not None:
            txt += f"  ·  t = {t:.3f} Myr"
        return txt + f"  ·  fov {self.live.fov:.1f}°  ·  distance {self.live.distance:.4g}"

    def set_dump(self, i):
        """Show dump i; a followed sink keeps its place in the view (the camera moves with it)."""
        if not self.dumps:
            return
        i = int(np.clip(i, 0, self.n_dumps - 1))
        st = self.live
        if i == st.dump:
            return
        tr = self.ed.path.tracks
        if st.follow is not None and tr is not None and tr.has(st.follow):
            a0, a1 = tr.position(st.follow, st.dump), tr.position(st.follow, i)
            if st.ride:
                st.pos = st.pos + (a1 - a0)
            st.target = a1.copy()
        st.dump = float(i)
        self.update_proxy()
        self.live_changed()

    def step_dump(self, n):
        self.set_dump(self.dump_index() + n)

    def dumps_changed(self):
        """New dump list (File → Set simulation dumps): back to the first one, framed again."""
        self.live.dump = 0.0
        self.live.follow, self.live.ride = None, False
        self.selected = None
        self.sample = None
        self.proxy_path = None
        self._framed = False
        self.update_proxy()
        self.live_changed()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            if self.dump_index() >= self.n_dumps - 1:
                self.set_dump(0)
            self.play_timer.start(int(1000 / self.play_rate))
        else:
            self.play_timer.stop()
            self.refine_timer.start()
        self.update_proxy()
        self.panel.refresh()
        self.canvas.update()

    def set_play_rate(self, rate):
        self.play_rate = rate
        if self.playing:
            self.play_timer.setInterval(int(1000 / rate))

    def _play_tick(self):
        if self.dump_index() >= self.n_dumps - 1:
            self.toggle_play()
            return
        proxy = self.ed.proxy
        nxt = self.dumps[self.dump_index() + 1]
        if proxy is not None and not proxy.has(nxt) and nxt not in proxy.errors:
            return  # wait for its gas: the play never runs ahead of the reading
        self.step_dump(1)

    # ------------------------------------------------------------------ proxy samples

    def update_proxy(self):
        """Show the cached sample of the current dump, and prefetch the next ones."""
        proxy = self.ed.proxy
        if proxy is None or not self.dumps:
            return
        cur = self.dump_path()
        if cur != self.proxy_path:
            e = proxy.get(cur)
            if e is not None:
                self._use_sample(e["pos"], e["weight"], cur, e["h"])
        i, n = self.dump_index(), self.n_dumps
        ahead = 30 if self.playing else 4
        order = [i] + [j for d in range(1, ahead + 1) for j in ((i + d,) if d > 2 else (i + d, i - d)) if 0 <= j < n]
        proxy.want([self.dumps[j] for j in order])

    def _use_sample(self, pos, weight, path, h=None):
        self.sample, self.weight, self.proxy_path, self.sample_h = pos, weight, path, h
        self.sample_id += 1
        if not self._framed:
            self._framed = True
            self.recentre(keep_direction=False)

    def on_proxy_ready(self, path, err):
        if err:
            return  # reported by the editor
        if path == self.dump_path() and path != self.proxy_path:
            e = self.ed.proxy.get(path)
            if e is not None:
                self._use_sample(e["pos"], e["weight"], path, e["h"])
                self.canvas.update()

    def on_loaded(self, msg):
        """A dump loaded by the render process: its sample when the fast reader cannot read it."""
        proxy = self.ed.proxy
        if msg["meta"]["path"] == self.dump_path() and (proxy is None or msg["meta"]["path"] in proxy.errors):
            self._use_sample(msg["sample"], msg["weight"], msg["meta"]["path"])
        self.canvas.update()
        self.panel.refresh()

    # ------------------------------------------------------------------ camera

    def recentre(self, keep_direction=True):
        """Target the centre of the gas, at a distance that shows most of it."""
        if self.sample is None:
            return
        c = np.median(self.sample, axis=0).astype(float)
        R = float(np.percentile(np.linalg.norm(self.sample - c, axis=1), 90))
        st = self.live
        d = normalize(st.pos - st.target) if keep_direction else normalize([0, -2.6, 0.9])
        st.target = c
        st.pos = c + d * R * 2.75 * math.tan(math.radians(22.5)) / math.tan(math.radians(st.fov) / 2)
        st.follow, st.ride = None, False
        self.live_changed()

    def zoom(self, factor):
        self.live.fov = float(np.clip(self.live.fov * factor, 1.0, 150.0))
        self.live_changed()

    def axis_view(self, digit):
        """Look at the target from an axis, at the same distance (up +Z, or +Y from the top)."""
        st = self.live
        d = np.array(AXIS_DIRS[digit], dtype=float)
        st.pos = st.target + d * st.distance
        st.up = np.array([0.0, 1.0, 0.0]) if digit == "7" else np.array([0.0, 0.0, 1.0])
        self.live_changed()

    def follow(self, mode):
        """Look at / ride along the selected sink through the dumps, or stop."""
        st = self.live
        if mode == "stop":
            st.follow, st.ride = None, False
        else:
            if self.selected is None:
                self.ed.statusBar().showMessage("click a star or a core first", 3000)
                return
            a = self.ed.path.anchor(self.selected, st.dump)
            if a is None:
                self.ed.statusBar().showMessage("sink trajectories not read yet", 3000)
                return
            offset = st.pos - st.target
            st.follow, st.ride = int(self.selected), mode == "ride"
            st.target = a.copy()
            if st.ride:
                st.pos = a + offset
        self.live_changed()

    def auto_clim(self):
        if self.sample is None:
            return
        nx, ny = self.preview_size(cap=320)
        sig = splat_column(self.sample, self.weight, self.live, nx, ny)
        nz = sig[sig > 0]
        if nz.size:
            lo, hi = np.log10(np.percentile(nz, [20, 99.7]))
            self.live.clim = (round(float(lo), 2), round(float(hi), 2))
            self.live_changed()

    def toggle_grid(self):
        self.canvas.show_grid = not self.canvas.show_grid
        self.panel.refresh()
        self.canvas.update()

    def toggle_help(self):
        self.show_help = not self.show_help
        self.canvas.update()

    # ------------------------------------------------------------------ rendering

    def look_key(self):
        return (json.dumps(self.live.to_dict()), json.dumps(self.look, sort_keys=True), self.preview_size(), self.dump_path())

    def request_refine(self):
        if self.ed.mode != "view" or self.playing or not self.dumps:
            return
        key = self.look_key()
        if key == self.preview_key or key in self.pending.values():
            return
        nx, ny = self.preview_size()
        rid = self.backend.request(
            self.live, self.dump_path(), nx, ny, self.look, exposure=self.live_exposure(nx, ny), fmt=self.ed.path.format
        )
        self.pending = {rid: key}
        self.canvas.update()

    def on_frame(self, msg):
        if msg.get("keep"):
            fname = self.exports.pop(msg["id"], None)
            if fname:
                save_png(msg["rgb"], fname)
                self.ed.statusBar().showMessage(f"saved {fname} ({msg['info'].get('total', 0):.1f} s)", 8000)
            return
        key = self.pending.pop(msg["id"], None)
        if key is None:
            return
        self.preview_img = np_to_qimage(msg["rgb"])
        self.preview_info = msg["info"]
        self.preview_key = key
        self.canvas.update()
        if key != self.look_key():
            self.refine_timer.start()

    def on_error(self, msg):
        self.pending.pop(msg.get("id"), None)
        self.exports.pop(msg.get("id"), None)

    def save_image(self):
        if not self.dumps:
            return
        name = os.path.splitext(os.path.basename(self.dump_path()))[0]
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", f"{name}.png", "PNG (*.png)")
        if not fname:
            return
        nx = int(self.export_width)
        ny = max(1, int(round(nx / self.aspect())))
        rid = self.backend.request(
            self.live, self.dump_path(), nx, ny, self.look, keep=True, exposure=self.live_exposure(nx, ny), fmt=self.ed.path.format
        )
        self.exports[rid] = fname
        self.ed.statusBar().showMessage(f"rendering {nx}×{ny} image…")

    # ------------------------------------------------------------------ mode switch

    def activate(self):
        """The view mode becomes the shown one."""
        self.update_proxy()
        self.panel.refresh()
        self.canvas.setFocus()
        self.request_refine()

    def deactivate(self):
        if self.playing:
            self.toggle_play()


class ViewPanel(QtWidgets.QWidget):
    """Side panel of the view mode: dumps, camera, display."""

    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        self._updating = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # -- dumps
        g = QtWidgets.QGroupBox("Dump")
        v = QtWidgets.QVBoxLayout(g)
        self.dump_label = QtWidgets.QLabel()
        self.dump_label.setWordWrap(True)
        v.addWidget(self.dump_label)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Left / Right stay dump steps
        self.slider.valueChanged.connect(lambda i: None if self._updating else vm.set_dump(i))
        v.addWidget(self.slider)
        row = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for name, text, fn in (
            ("first", "⏮", lambda: vm.set_dump(0)),
            ("prev", "◀", lambda: vm.step_dump(-1)),
            ("play", "▶", vm.toggle_play),
            ("next", "▶|", lambda: vm.step_dump(1)),
            ("last", "⏭", lambda: vm.set_dump(vm.n_dumps - 1)),
        ):
            b = QtWidgets.QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space is "next dump", not "press the button"
            b.setFixedWidth(42)
            b.clicked.connect(fn)
            row.addWidget(b)
            self.buttons[name] = b
        v.addLayout(row)
        self.rate = dspin(0.25, 60, 2, 1, " dumps/s")
        self.rate.setValue(vm.play_rate)
        self.rate.valueChanged.connect(vm.set_play_rate)
        form = QtWidgets.QFormLayout()
        form.addRow("play rate", self.rate)
        v.addLayout(form)
        lay.addWidget(g)

        # -- camera
        g = QtWidgets.QGroupBox("Camera")
        form = QtWidgets.QFormLayout(g)
        self.fov = dspin(1, 150, 1, 1, "°")
        self.dist = dspin(1e-6, 1e9, 4, 0.5)
        self.fov.valueChanged.connect(self._camera_edited)
        self.dist.valueChanged.connect(self._camera_edited)
        form.addRow("fov (vertical)", self.fov)
        form.addRow("distance", self.dist)
        row = QtWidgets.QHBoxLayout()
        for text, fn, tip in (
            ("Front", lambda: vm.axis_view("1"), "look from -Y (1)"),
            ("Side", lambda: vm.axis_view("3"), "look from +X (3)"),
            ("Top", lambda: vm.axis_view("7"), "look from +Z (7)"),
            ("Recentre", vm.recentre, "target the centre of the gas (O)"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        form.addRow(row)
        self.follow_label = QtWidgets.QLabel()
        self.follow_label.setWordWrap(True)
        form.addRow("sink", self.follow_label)
        row = QtWidgets.QHBoxLayout()
        for mode, text, tip in (
            ("look", "Look at", "keep the selected star / core at the centre through the dumps (T)"),
            ("ride", "Ride along", "move with the selected star / core through the dumps (Shift+T)"),
            ("stop", "Stop", "stop following (T)"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, m=mode: vm.follow(m))
            row.addWidget(b)
        form.addRow(row)
        lay.addWidget(g)

        # -- display
        g = QtWidgets.QGroupBox("Display")
        form = QtWidgets.QFormLayout(g)
        self.clo = dspin(-10, 10, 2, 0.1)
        self.chi = dspin(-10, 10, 2, 0.1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.clo)
        row.addWidget(self.chi)
        b = QtWidgets.QPushButton("Auto")
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setToolTip("colour range from the gas in view (C)")
        b.clicked.connect(vm.auto_clim)
        row.addWidget(b)
        form.addRow("log Σ [M☉/pc²]", row)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(CMAPS)
        form.addRow("colormap", self.cmap)
        self.checks = {}
        row = QtWidgets.QHBoxLayout()
        for key, text in (("show_gas", "gas"), ("time_label", "time label")):
            c = QtWidgets.QCheckBox(text)
            c.toggled.connect(self._look_edited)
            self.checks[key] = c
            row.addWidget(c)
        self.grid = QtWidgets.QCheckBox("thirds grid")
        self.grid.toggled.connect(lambda on: None if self._updating else vm.toggle_grid())
        row.addWidget(self.grid)
        form.addRow(row)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(list(RENDER_SCALES))
        self.scale.setToolTip("size of the real render relative to the view (lower: faster)")
        form.addRow("render size", self.scale)
        lay.addWidget(g)

        # -- sinks: only the controls of the sink mode (inspectors.SinkLook, as in the movie panel)
        g = QtWidgets.QGroupBox("Sinks")
        v = QtWidgets.QVBoxLayout(g)
        self.sink_look = sl = SinkLook(self._look_edited)
        self.sinks, self.star_racc = sl.mode, sl.thr
        v.addWidget(sl)
        self.expo = dspin(-10, 10, 2, 0.25, " dex")
        self.expo_form = QtWidgets.QFormLayout()
        self.expo_form.addRow("star exposure", self.expo)
        v.addLayout(self.expo_form)
        for w in (self.clo, self.chi, self.expo):
            w.valueChanged.connect(self._look_edited)
        for w in (self.cmap, self.scale):
            w.currentIndexChanged.connect(self._look_edited)
        lay.addWidget(g)

        # -- image
        g = QtWidgets.QGroupBox("Image")
        form = QtWidgets.QFormLayout(g)
        self.width = QtWidgets.QSpinBox()
        self.width.setRange(64, 16384)
        self.width.setSingleStep(160)
        self.width.setSuffix(" px")
        self.width.setValue(vm.export_width)
        self.width.valueChanged.connect(lambda w: setattr(vm, "export_width", w))
        form.addRow("width", self.width)
        b = QtWidgets.QPushButton("Save image… (Ctrl+E)")
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setToolTip("render the view at this width (the view's aspect) and save it as PNG")
        b.clicked.connect(vm.save_image)
        form.addRow(b)
        lay.addWidget(g)

        self.renderer = QtWidgets.QLabel()
        self.renderer.setWordWrap(True)
        lay.addWidget(self.renderer)
        lay.addStretch(1)

    def refresh(self):
        vm = self.vm
        st = vm.live
        self._updating = True
        n = vm.n_dumps
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(vm.dump_index())
        self.slider.setEnabled(n > 1)
        if n:
            i = vm.dump_index()
            t = vm.dump_time_myr(i)
            self.dump_label.setText(f"{i} / {n - 1}  ·  {os.path.basename(vm.dumps[i])}" + (f"\nt = {t:.3f} Myr" if t is not None else ""))
        else:
            self.dump_label.setText("no simulation dumps")
        self.buttons["play"].setText("⏸" if vm.playing else "▶")
        self.fov.setValue(st.fov)
        self.dist.setValue(st.distance)
        self.clo.setValue(st.clim[0])
        self.chi.setValue(st.clim[1])
        self.expo.setValue(st.exposure)
        if vm.look["cmap"] not in CMAPS:
            self.cmap.addItem(vm.look["cmap"])
        self.cmap.setCurrentText(vm.look["cmap"])
        for k, c in self.checks.items():
            c.setChecked(bool(vm.look[k]))
        self.sink_look.set_look(vm.look)
        self.sink_look.radius.setSpecialValueText(vm.ed.disc_radius_hint(vm.look))
        self.expo_form.setRowVisible(self.expo, has_stars(vm.look))
        self.grid.setChecked(vm.canvas.show_grid)
        self.scale.setCurrentText(next(k for k, v in RENDER_SCALES.items() if v == vm.render_scale))
        if st.follow is not None:
            self.follow_label.setText(f"◎ {vm.sink_label(st.follow)} · {'ride along' if st.ride else 'look at'}")
        elif vm.selected is not None:
            self.follow_label.setText(f"selected {vm.sink_label(vm.selected)}")
        else:
            self.follow_label.setText("click a star / core in the view")
        name = getattr(vm.backend, "name", None)
        self.renderer.setText(
            {"shamrock": "renderer: shamrock", "numpy": "renderer: numpy SPH (slower than shamrock)"}.get(
                name, "renderer: starting…"
            )
        )
        self._updating = False

    def _camera_edited(self):
        if self._updating:
            return
        st = self.vm.live
        st.fov = self.fov.value()
        if abs(self.dist.value() - st.distance) > 1e-9 * max(st.distance, 1e-30):
            st.dolly(self.dist.value() / st.distance)
        self.vm.live_changed(from_panel=True)

    def _look_edited(self):
        if self._updating:
            return
        vm = self.vm
        vm.live.clim = (self.clo.value(), self.chi.value())
        vm.live.exposure = self.expo.value()
        vm.look["cmap"] = self.cmap.currentText()
        for k, c in self.checks.items():
            vm.look[k] = c.isChecked()
        self.sink_look.write(vm.look)
        self.sink_look.set_look(vm.look)  # the controls of the (new) sink mode
        self.sink_look.radius.setSpecialValueText(vm.ed.disc_radius_hint(vm.look))
        self.expo_form.setRowVisible(self.expo, has_stars(vm.look))
        vm.render_scale = RENDER_SCALES[self.scale.currentText()]
        vm.live_changed(from_panel=True)
