# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Sink trajectories through a series of dumps, for cameras that follow a sink.

The sinks are read with the fast reader of the dump format (data/readers.py: only the
header and the sink block, a few ms per dump instead of seconds for a full load).

SinkTracks holds the positions / velocities of every sink in every dump and gives a
smooth position at a fractional dump index (cubic Hermite with the sink velocities).
A sink is identified by its index in the sink arrays, which Phantom keeps across dumps
(accreted / merged sinks stay in place with a mass <= 0).
"""

import hashlib
import os

import numpy as np

from ..constants import DEFAULT_SINK_MODE, MSUN_G, PC_CM, STAR_MAX_RACC_PC
from .readers import DEFAULT_FORMAT, get_reader


def split_sinks(id, pos, mass, racc, pc_per_unit, look=None):  # noqa: A002 (the key of the sink dicts)
    """
    Sinks to draw, by look["sinks"] (constants.SINK_MODES): {"star_id", "core_id",
    "star_pos", "core_pos", "star_mass", "core_mass" [Msun], "core_racc" [code units]}.
    Stars get the star photometry (render/stars.py), "cores" are drawn as discs (the cores
    of "cluster", every sink with "markers", see compose.sink_discs). id, pos, mass [Msun],
    racc [code units]: the alive sinks (the dict of SinkTracks.raw_at, SimScene.sinks).
    """
    look = look or {}
    mode = look.get("sinks", DEFAULT_SINK_MODE)
    ids = np.asarray(id, dtype=int)
    if mode == "cluster":
        star = racc * pc_per_unit < look.get("star_max_racc_pc", STAR_MAX_RACC_PC)
    elif mode == "stars":
        star = np.ones(len(ids), bool)
    else:
        star = np.zeros(len(ids), bool)
    disc = ~star & (mode != "hidden")
    return {
        "star_id": ids[star],
        "core_id": ids[disc],
        "star_pos": pos[star],
        "core_pos": pos[disc],
        "star_mass": mass[star],
        "core_mass": mass[disc],
        "core_racc": racc[disc],
    }


def sink_kind(racc, pc_per_unit, look=None):
    """What one sink is drawn as, for the UI: "star", "core" or "sink"."""
    look = look or {}
    mode = look.get("sinks", DEFAULT_SINK_MODE)
    if mode == "cluster":
        return "star" if racc * pc_per_unit < look.get("star_max_racc_pc", STAR_MAX_RACC_PC) else "core"
    return "star" if mode == "stars" else "sink"


def _cache_key(dumps, fmt):
    h = hashlib.sha1(f"{fmt}\n".encode())
    for d in dumps:
        try:
            st = os.stat(d)
            h.update(f"{d}|{st.st_size}|{int(st.st_mtime)}\n".encode())
        except OSError:
            h.update(f"{d}|missing\n".encode())
    return h.hexdigest()[:20]


def cache_path(dumps, fmt=DEFAULT_FORMAT):
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base, "plouf", f"sinks_{_cache_key(dumps, fmt)}.npz")


class SinkTracks:
    """Every sink in every dump. Arrays are (ndump, nsink, ...) with NaN where a sink does not exist yet."""

    def __init__(self, time, pos, vel, mass, racc, units=None):
        self.time = time
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.racc = racc
        self.units = units or {}
        self.ndump, self.nsink = mass.shape
        # where a sink is not alive (not formed yet / accreted), hold its nearest alive position
        alive = np.isfinite(mass) & (mass > 0)
        self.alive = alive
        self._pos = pos.copy()
        self._vel = np.where(alive[..., None], vel, 0.0)
        for s in range(self.nsink):
            idx = np.flatnonzero(alive[:, s])
            if len(idx) == 0:
                self._pos[:, s] = 0.0
                continue
            near = idx[np.clip(np.searchsorted(idx, np.arange(self.ndump)), 0, len(idx) - 1)]
            prev = idx[np.clip(np.searchsorted(idx, np.arange(self.ndump), side="right") - 1, 0, len(idx) - 1)]
            d = np.arange(self.ndump)
            choose = np.where(np.abs(near - d) <= np.abs(prev - d), near, prev)
            dead = ~alive[:, s]
            self._pos[dead, s] = pos[choose[dead], s]

    # -- building

    @classmethod
    def load(cls, cp):
        z = np.load(cp)
        units = {k: float(z[k]) for k in ("udist", "umass", "utime") if k in z}
        f8 = [z[k].astype(float) for k in ("pos", "vel", "mass", "racc")]
        return cls(z["time"], *f8, units)

    @classmethod
    def build(cls, dumps, progress=None, use_cache=True, fmt=DEFAULT_FORMAT):
        cp = cache_path(dumps, fmt)
        reader = get_reader(fmt)
        if use_cache and os.path.exists(cp):
            try:
                return cls.load(cp)
            except Exception:
                pass
        recs = []
        for i, d in enumerate(dumps):
            recs.append(reader.read_sinks(d))
            if progress:
                progress(i + 1, len(dumps))
        S = max((len(r["mass"]) for r in recs), default=0)
        D = len(recs)
        time = np.array([r["time"] for r in recs])
        pos = np.full((D, S, 3), np.nan)
        vel = np.full((D, S, 3), np.nan)
        mass = np.full((D, S), np.nan)
        racc = np.full((D, S), np.nan)
        for i, r in enumerate(recs):
            n = len(r["mass"])
            pos[i, :n], vel[i, :n], mass[i, :n], racc[i, :n] = r["pos"], r["vel"], r["mass"], r["racc"]
        units = {k: recs[0][k] for k in ("udist", "umass", "utime")} if recs else {}
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                f4 = {k: v.astype(np.float32) for k, v in (("pos", pos), ("vel", vel), ("mass", mass), ("racc", racc))}
                np.savez_compressed(cp + ".tmp.npz", time=time, **f4, **units)
                os.replace(cp + ".tmp.npz", cp)
            except OSError:
                pass
        return cls(time, pos, vel, mass, racc, units)

    # -- units

    @property
    def pc_per_unit(self):
        return self.units.get("udist", PC_CM) / PC_CM

    @property
    def msun_per_unit(self):
        return self.units.get("umass", MSUN_G) / MSUN_G

    # -- queries

    def has(self, sink):
        return sink is not None and 0 <= int(sink) < self.nsink and self.ndump > 0

    def track(self, sink, dumps):
        """Smooth positions (m, 3) of one sink at fractional dump indices (exact at integers)."""
        f = np.clip(np.atleast_1d(np.asarray(dumps, dtype=float)), 0, self.ndump - 1)
        if self.ndump == 1:
            return np.repeat(self._pos[:1, sink], len(f), axis=0)
        i = np.minimum(np.floor(f).astype(int), self.ndump - 2)
        s = (f - i)[:, None]
        p0, p1 = self._pos[i, sink], self._pos[i + 1, sink]
        dt = (self.time[i + 1] - self.time[i])[:, None]
        dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
        m0, m1 = self._vel[i, sink] * dt, self._vel[i + 1, sink] * dt
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def position(self, sink, dump):
        """Smooth position of one sink (index) at a fractional dump index, None if unknown."""
        if not self.has(sink):
            return None
        return self.track(int(sink), [dump])[0]

    def positions_at(self, sinks, dump):
        """Smooth positions (len(sinks), 3) of several sinks at a fractional dump index."""
        sinks = np.asarray(sinks, dtype=int)
        f = float(np.clip(dump, 0, self.ndump - 1))
        i = min(int(np.floor(f)), max(self.ndump - 2, 0))
        if self.ndump == 1 or f == i:
            return self._pos[i, sinks].copy()
        s = f - i
        dt = self.time[i + 1] - self.time[i]
        dt = dt if np.isfinite(dt) and dt > 0 else 0.0
        p0, p1 = self._pos[i, sinks], self._pos[i + 1, sinks]
        m0, m1 = self._vel[i, sinks] * dt, self._vel[i + 1, sinks] * dt
        return (
            (2 * s**3 - 3 * s**2 + 1) * p0
            + (s**3 - 2 * s**2 + s) * m0
            + (-2 * s**3 + 3 * s**2) * p1
            + (s**3 - s**2) * m1
        )

    def raw_at(self, dump):
        """
        The sinks alive at a (fractional) dump, at their smooth position there: {"id", "pos",
        "mass" [Msun], "racc" [code units]} (the same dict as backend.SimScene.sinks).
        """
        i = int(np.clip(round(dump), 0, self.ndump - 1))
        ids = np.flatnonzero(self.alive[i])
        return {
            "id": ids,
            "pos": self.positions_at(ids, dump),
            "mass": self.mass[i, ids] * self.msun_per_unit,
            "racc": self.racc[i, ids],
        }

    def sinks_at(self, dump, look=None):
        """Sinks to draw at a (fractional) dump (split_sinks with the look's sink mode)."""
        return split_sinks(**self.raw_at(dump), pc_per_unit=self.pc_per_unit, look=look)

    def describe(self, sink, dump, look=None):
        """Short text for the UI: kind, mass, existence."""
        i = int(np.clip(round(dump), 0, self.ndump - 1))
        m = self.mass[i, sink]
        r = self.racc[i, sink]
        pc = self.pc_per_unit
        msun = self.msun_per_unit
        if not (np.isfinite(m) and m > 0):
            alive = np.flatnonzero(self.alive[:, sink])
            if len(alive) == 0:
                return f"sink #{sink} (never alive)"
            when = "forms at dump " + str(alive[0]) if alive[0] > i else "gone after dump " + str(alive[-1])
            return f"sink #{sink} ({when})"
        return f"{sink_kind(r, pc, look)} #{sink} · {m * msun:.3g} M☉"
