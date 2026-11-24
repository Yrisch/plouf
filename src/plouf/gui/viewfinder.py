# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Viewfinder: what the camera sees (instant particle proxy, then the real render).

The view mode's canvas (gui/viewmode.py) is a Viewfinder too: its ``ed`` is then the
view mode, which provides the same small interface (live, look, sample, preview, sinks...).
"""

import json
import math
import os

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen

from ..path.camera import ZNEAR, normalize, project
from ..render.compose import compose, splat_column, splat_sph
from .keymaps import KEYMAPS, key_char, keymap_hint
from .qtutil import draw_dots, fit_rect, np_to_qimage


PROXY_PIXELS = 1920 * 1080  # the proxy at rest: the view's resolution (device pixels), up to this
PROXY_FAST_PIXELS = 400_000  # while the view changes (moving, playing): coarser, and half of the sample
PROXY_SETTLE_MS = 200  # the view unchanged this long: the proxy is redrawn at rest resolution


class Viewfinder(QtWidgets.QWidget):

    def __init__(self, ed):
        super().__init__()
        self.ed = ed
        self.setMinimumSize(320, 180)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._drag = None
        self._press = None
        self._proxy = (None, None)
        self._view = None  # what the proxy shows, to tell a changing view from a still one
        self._cancelled = set()  # render requests already cancelled
        self._fast = False
        self._settle = QTimer(self, singleShot=True, interval=PROXY_SETTLE_MS)
        self._settle.timeout.connect(self._settled)
        self.show_grid = False

    def _settled(self):
        self._fast = False
        self.update()

    def look(self):
        return self.ed.path.look

    def proxy_size(self, rect, fast):
        """Proxy image size for a view rect: its device pixels, within the pixel budget."""
        dpr = self.devicePixelRatioF()
        w, h = max(rect.width() * dpr, 1.0), max(rect.height() * dpr, 1.0)
        k = min(1.0, math.sqrt((PROXY_FAST_PIXELS if fast else PROXY_PIXELS) / (w * h)))
        return max(16, int(w * k)), max(9, int(h * k))

    def proxy_image(self, rect=None):
        """
        The instant view of the particle sample. While the view changes it is drawn coarser
        and from half of the sample; PROXY_SETTLE_MS after the last change, at the view's
        resolution (with the sample's smoothing lengths: sharp where the gas is dense).
        """
        ed = self.ed
        if ed.sample is None:
            return None
        if rect is None:
            rect = fit_rect(self.width(), self.height(), ed.aspect())
        st = ed.live
        view = (json.dumps(st.to_dict()), ed.sample_id, json.dumps(self.look(), sort_keys=True))
        if view != self._view:
            if self._view is not None:
                self._fast = True
                self._settle.start()
            self._view = view
        fast = self._fast or ed.playing
        nx, ny = self.proxy_size(rect, fast)
        key = (view, nx, ny, fast)
        if self._proxy[0] != key:
            k = 2 if fast else 1
            h = getattr(ed, "sample_h", None)
            if h is not None:  # half of the sample: twice the mass, kernels 2^(1/3) wider
                sig = splat_sph(ed.sample[::k], h[::k] * k ** (1 / 3), ed.weight * k, st, nx, ny)
            else:  # a sample without smoothing lengths (from the render process): plain splat
                sig = splat_column(ed.sample[::k], ed.weight * k, st, nx, ny, smooth=max(0.7, nx / 640))
            rgb = compose(sig, st, self.look(), ed.display_sinks(), None, None)
            self._proxy = (key, np_to_qimage(rgb))
        return self._proxy[1]

    def paintEvent(self, ev):
        ed = self.ed
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        rect = fit_rect(self.width(), self.height(), ed.aspect())
        p.fillRect(rect, QColor(0, 0, 0))
        self._drop_stale_render()
        mode = self._paint_image(p, rect)
        self._paint_marks(p, rect)
        self._paint_hud(p, rect, mode)

    def _drop_stale_render(self):
        """The render in progress is for a view that is gone (moved, next dump, playing): cancel it."""
        ed = self.ed
        if not ed.pending or ed.backend is None:
            return
        rid = next(iter(ed.pending))
        if rid not in self._cancelled and ed.look_key() not in ed.pending.values():
            self._cancelled.add(rid)
            ed.backend.cancel()

    def _paint_image(self, p, rect):
        """The real render if it is up to date, else the proxy. Returns the HUD text of what is shown."""
        ed = self.ed
        if ed.preview_valid():
            p.drawImage(rect, ed.preview_img)
            name = ed.preview_info.get("backend", "render").upper()
            return f"{name} {ed.preview_img.width()}×{ed.preview_img.height()} · {ed.preview_info.get('total', 0):.2f} s"
        img = self.proxy_image(rect)
        if img is not None:
            p.drawImage(rect, img)
        sinks = ed.display_sinks()
        if self.look().get("show_stars", True) and sinks is not None and len(sinks["star_pos"]):
            self._proxy_stars(p, rect)
        mode = "PROXY" + ("  ·  rendering…" if ed.backend and ed.backend.busy else "")
        if ed.proxy_status():
            mode += "  ·  " + ed.proxy_status()
        return mode

    def _paint_marks(self, p, rect):
        """Selected / followed sink rings and the thirds grid (editor only, never in the renders)."""
        ed = self.ed
        for sink, col, rad in ((ed.selected, QColor(255, 255, 255, 200), 8.0), (ed.live.follow, QColor(80, 230, 230), 11.0)):
            pt = ed.sink_position(sink)
            if pt is None:
                continue
            sx, sy, z = project(ed.live, pt[None], rect.width(), rect.height())
            if z[0] > ZNEAR:
                p.setPen(QPen(col, 1.5))
                p.drawEllipse(QPointF(rect.left() + sx[0], rect.top() + sy[0]), rad, rad)

        if self.show_grid:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for f in (1 / 3, 2 / 3):
                x = rect.left() + rect.width() * f
                y = rect.top() + rect.height() * f
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.PenStyle.DashLine))
            p.drawRect(rect.adjusted(rect.width() * 0.05, rect.height() * 0.05, -rect.width() * 0.05, -rect.height() * 0.05))

    def _paint_hud(self, p, rect, mode):
        ed = self.ed
        p.setPen(QColor(220, 220, 220))
        f = p.font()
        f.setPointSizeF(9)
        p.setFont(f)
        p.drawText(QPointF(rect.left() + 8, rect.top() + 16), mode)
        if self.hasFocus():
            p.setPen(QColor(160, 160, 170))
            p.drawText(QPointF(rect.left() + 8, rect.top() + 32), keymap_hint(ed.keyboard))
            p.setPen(QColor(220, 220, 220))
        meta = ed.meta
        frame = int(round(ed.t * ed.path.fps))
        info = f"t = {ed.t:6.2f} s  ·  frame {frame}"
        if ed.path.dumps:
            di = ed.path.dump_index(ed.live)
            info += f"  ·  dump {di} {os.path.basename(ed.path.dumps[di])}"
        if meta:
            info += f"  ·  {meta['time_myr']:.2f} Myr"
        p.drawText(QPointF(rect.left() + 8, rect.bottom() - 8), info)
        on = ed.key_index()
        if on is not None:
            p.setPen(QColor(255, 210, 60))
            txt = f"◆ editing keyframe {on + 1}"
        elif ed.live_dirty:
            p.setPen(QColor(255, 110, 90))
            txt = "● camera not keyed — press K"
        else:
            p.setPen(QColor(160, 160, 170))
            txt = "interpolated"
        p.drawText(QPointF(rect.right() - p.fontMetrics().horizontalAdvance(txt) - 8, rect.top() + 16), txt)
        if ed.live.follow is not None:
            p.setPen(QColor(80, 230, 230))
            txt = f"◎ {ed.sink_label(ed.live.follow)} · {'ride along' if ed.live.ride else 'look at'}"
            p.drawText(QPointF(rect.right() - p.fontMetrics().horizontalAdvance(txt) - 8, rect.top() + 32), txt)

    def _proxy_stars(self, p, rect):
        ed = self.ed
        sinks = ed.display_sinks()
        pos = sinks["star_pos"]
        sx, sy, z = project(ed.live, pos, rect.width(), rect.height())
        ok = (z > ZNEAR) & (sx >= 0) & (sx < rect.width()) & (sy >= 0) & (sy < rect.height())
        if not ok.any():
            return
        # the dots of the renders without PSF (StarLight.dots), sizes relative to the frame
        # width; a few brightness levels and temperature bins, each drawn in one call
        sl = ed.starlight
        lum, T = sl.luminosities(sinks["star_mass"][ok])
        F = sl.fluxes(ed.live, pos[ok], lum, None, ed.pc_per_unit())
        b, _, _, _ = sl.dots(F, ed.live_exposure(rect.width(), rect.height()))
        scale = rect.width() / sl.DOT_REF_WIDTH
        level = np.minimum((b * 6).astype(int), 5)
        tbin = np.clip(np.digitize(np.log10(T), np.linspace(3.4, 4.5, 7)), 0, 7)
        cols = np.clip(sl.colours(10 ** np.linspace(3.3, 4.6, 8)), 0, 1) ** (1 / 2.2)
        X, Y = rect.left() + sx[ok], rect.top() + sy[ok]
        for lv in range(6):
            for tb in range(8):
                m = (level == lv) & (tbin == tb) & (b > 0.5 / 255)
                if m.any():
                    v = (lv + 0.5) / 6
                    r = (sl.DOT_R0 + sl.DOT_R1 * v) * scale
                    draw_dots(p, X[m], Y[m], r, QColor.fromRgbF(*cols[tb], sl.DOT_A0 + sl.DOT_A1 * v))

    # -- events

    def mousePressEvent(self, ev):
        self.setFocus()
        mods = ev.modifiers()
        b = ev.button()
        x, y = ev.position().x(), ev.position().y()
        if b == Qt.MouseButton.MiddleButton or (b == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier):
            self._drag = ("pan", x, y, None)
        elif b == Qt.MouseButton.RightButton or (b == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ControlModifier):
            self._drag = ("look", x, y, None)
        elif b == Qt.MouseButton.LeftButton:
            # orbit around the camera up as set by the user (+Z: classic level turntable)
            self._drag = ("orbit", x, y, normalize(self.ed.live.up))
            self._press = (x, y, False)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        kind, x0, y0, extra = self._drag
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - x0, y - y0
        if self._press and not self._press[2]:
            if math.hypot(x - self._press[0], y - self._press[1]) <= 3:
                return  # maybe a click: do not move the camera yet
            dx, dy = x - self._press[0], y - self._press[1]
            self._press = (self._press[0], self._press[1], True)
        self._drag = (kind, x, y, extra)
        live = self.ed.live
        H = fit_rect(self.width(), self.height(), self.ed.aspect()).height()
        if kind == "orbit":
            live.turntable(dx * 0.005, dy * 0.005, world_up=extra)
        elif kind == "pan":
            k = 2 * math.tan(math.radians(live.fov) / 2) / H
            live.pan(-dx * k, dy * k)
        elif kind == "look":
            k = math.radians(live.fov) / H
            live.look(dx * k, -dy * k)
        self.ed.live_changed()

    def mouseReleaseEvent(self, ev):
        if self._press and not self._press[2]:  # click without drag: select a star / core
            rect = fit_rect(self.width(), self.height(), self.ed.aspect())
            x, y = self._press[0] - rect.left(), self._press[1] - rect.top()
            self.ed.select_sink(self.ed.pick_sink(self.ed.live, rect.width(), rect.height(), x, y))
        elif self._drag is not None:
            self.ed.commit()
        self._press = None
        self._drag = None

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120
        mods = ev.modifiers()
        live = self.ed.live
        if mods & Qt.KeyboardModifier.ControlModifier:
            live.fov = float(np.clip(live.fov * 0.93**steps, 1.0, 150.0))
        elif mods & Qt.KeyboardModifier.AltModifier:
            # dolly zoom: keep the target plane the same size on screen
            steps = (ev.angleDelta().x() or ev.angleDelta().y()) / 120
            new_fov = float(np.clip(live.fov * 0.93**steps, 2.0, 140.0))
            w = live.distance * math.tan(math.radians(live.fov) / 2)
            live.fov = new_fov
            live.dolly(w / math.tan(math.radians(new_fov) / 2) / live.distance)
        else:
            live.dolly(0.9**steps)
        self.ed.live_changed()
        self.ed.commit_soon()

    def focusInEvent(self, ev):
        self.update()
        super().focusInEvent(ev)

    def focusOutEvent(self, ev):
        self.update()
        super().focusOutEvent(ev)

    def keyPressEvent(self, ev):
        c = key_char(ev)
        km = KEYMAPS[self.ed.keyboard]
        fast = 5.0 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        live = self.ed.live
        if c in km["fly"]:
            f, r, u = km["fly"][c]
            s = 0.02 * fast
            live.fly(f * s, r * s, u * s)
        elif c in km["roll"]:
            live.roll(km["roll"][c] * math.radians(2 * fast))
        else:
            return super().keyPressEvent(ev)
        self.ed.live_changed()

    def keyReleaseEvent(self, ev):
        c = key_char(ev)
        km = KEYMAPS[self.ed.keyboard]
        if not ev.isAutoRepeat() and (c in km["fly"] or c in km["roll"]):
            self.ed.commit()
        super().keyReleaseEvent(ev)
