# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Background work of the editor: the render process (shamrock, or the numpy renderer), the sink tracks reader.
Workers never touch widgets; they report through Qt signals to the GUI thread.
"""

import multiprocessing as mp

from PyQt6 import QtCore
from PyQt6.QtCore import QTimer, pyqtSignal

from ..data.sinktracks import SinkTracks
from ..render import backend


class TracksBuilder(QtCore.QThread):
    """Reads the sink block of every dump in the background (cached in ~/.cache/plouf)."""

    progress = pyqtSignal(int, int)
    done = pyqtSignal(object, object)  # dumps tuple, SinkTracks or exception

    def __init__(self, dumps, fmt):
        super().__init__()
        self.dumps = tuple(dumps)
        self.fmt = fmt

    def run(self):
        try:
            tr = SinkTracks.build(
                list(self.dumps), progress=lambda i, n: self.progress.emit(i, n) if i % 20 == 0 else None, fmt=self.fmt
            )
            self.done.emit(self.dumps, tr)
        except Exception as e:  # unreadable dump: following is disabled
            self.done.emit(self.dumps, e)


class ProxyBridge(QtCore.QObject):
    """Delivers the proxy cache notifications (worker thread) to the GUI thread."""

    ready = pyqtSignal(str, object)


class Backend(QtCore.QObject):
    loaded = pyqtSignal(object)
    frame = pyqtSignal(object)
    error = pyqtSignal(object)
    status = pyqtSignal(str)
    cancelled = pyqtSignal(object)  # {"id"}: a render dropped because the view changed (cancel)

    def __init__(self, sycl, sample_n, renderer="auto"):
        super().__init__()
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.generation = ctx.Value("i", 0)  # incremented by cancel(): renders asked before are dropped
        self.proc = ctx.Process(
            target=backend.server_main, args=(child, sycl, sample_n, renderer, self.generation), daemon=True
        )
        self.name = None  # renderer of the process once it is ready: "shamrock" or "numpy"
        self.proc.start()
        child.close()
        self.last_id = 0
        self.busy = False
        self._dead = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(15)

    def request(self, state, dump, nx, ny, look, keep=False, exposure=None, fmt=None):
        self.last_id += 1
        self.conn.send(
            {
                "cmd": "render",
                "id": self.last_id,
                "dump": dump,
                "state": state.to_dict(),
                "nx": int(nx),
                "ny": int(ny),
                "look": dict(look),
                "keep": keep,
                "exposure": exposure,
                "format": fmt,
                "gen": self.generation.value,
            }
        )
        if not keep:
            self.busy = True
        return self.last_id

    def cancel(self):
        """Drop the renders asked so far (not the exports): the view they were for is gone."""
        with self.generation.get_lock():
            self.generation.value += 1

    def send_tracks(self, path):
        try:
            self.conn.send({"cmd": "tracks", "path": path})
        except (OSError, ValueError):
            pass

    def _poll(self):
        try:
            while self.conn.poll():
                msg = self.conn.recv()
                t = msg["type"]
                if t == "loaded":
                    self.loaded.emit(msg)
                elif t == "frame":
                    if msg["id"] == self.last_id:
                        self.busy = False
                    self.frame.emit(msg)
                elif t == "error":
                    self.busy = False
                    self.error.emit(msg)
                elif t == "cancelled":
                    if msg["id"] == self.last_id:
                        self.busy = False
                    self.cancelled.emit(msg)
                elif t == "status":
                    self.status.emit(msg["msg"])
                elif t == "backend":
                    self.name = msg["name"]
        except (EOFError, OSError):
            pass
        if not self._dead and not self.proc.is_alive():
            self._dead = True
            self.error.emit({"id": None, "msg": f"render process died (exit code {self.proc.exitcode})"})

    def close(self):
        try:
            self.conn.send({"cmd": "quit"})
        except Exception:
            pass
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.terminate()
