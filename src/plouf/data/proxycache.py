# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Proxy particle samples of the dumps, read in the background and kept in memory.

The instant views of the editor (viewfinder proxy, world view) splat a sample of the gas
particles. Each dump's sample is read directly with the fast reader of the dump format
(data/readers.py, no shamrock), so scrubbing and playing the timeline shows the gas of
the right dump at once, while shamrock only loads a dump for the real renders.

Sample: the n living particles with the lowest rank, the rank being a fixed hash of the
particle id. The same particles are kept from one dump to the next; when one is accreted
the next ranked particle replaces it, so the sample always has n particles and does not
flicker. Each sampled particle carries (living gas mass) / n, and a smoothing length
grown by (living / n)^(1/3) so the sample fills the same volume as the full gas (with
n times fewer, heavier particles the kernels must overlap as much as the original ones).

Several dumps are read at once (~3x faster with 4 readers than with 1), so playing
through the dumps keeps up. With processes=True (the editor) the reading runs in worker
processes, so it does not take the GUI thread's time (the GIL); the cache threads only
dispatch and wait.
"""

import collections
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..constants import MSUN_G, PC_CM


def particle_rank(ids, seed=0x5EED):
    """Fixed pseudo-random uint64 rank of each particle id (splitmix64)."""
    with np.errstate(over="ignore"):
        z = ids.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def proxy_sample(gas, n):
    """
    {"pos" (m,3) float32, "h" (m,) float32 or None, "weight", "time", "nalive"} from a
    reader's read_gas() result. h: smoothing lengths [code units] grown for the sampling.
    """
    alive = np.flatnonzero(gas["alive"])
    if len(alive) > n:
        ids = gas["ids"][alive] if gas["ids"] is not None else alive
        keep = np.argpartition(particle_rank(ids), n - 1)[:n]
        sel = np.sort(alive[keep])
    else:
        sel = alive
    m = gas["mass"]
    total = float(np.sum(m[alive])) if np.ndim(m) else float(m) * len(alive)
    msun_per_unit = gas["umass"] / MSUN_G
    pc_per_unit = gas["udist"] / PC_CM
    sigma_unit = msun_per_unit / pc_per_unit**2  # code column density -> Msun/pc^2
    h = gas.get("h")
    if h is not None:
        h = (np.abs(h[sel]) * (len(alive) / max(len(sel), 1)) ** (1 / 3)).astype(np.float32)
    return {
        "pos": gas["pos"][sel].astype(np.float32),
        "h": h,
        # splat weight such that frame.splat_column gives Msun/pc^2 with code-unit areas
        "weight": total / max(len(sel), 1) * sigma_unit,
        "time": gas["time"],
        "nalive": len(alive),
    }


def _exit_with_parent(parent):
    """Worker process initializer: exit when the editor is gone (killed or crashed), not linger as an orphan."""

    def watch():
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _read_sample(fmt, path, n):
    """Worker process job: the proxy sample of one dump."""
    from .readers import get_reader

    t0 = time.perf_counter()
    sample = proxy_sample(get_reader(fmt).read_gas(path), n)
    sample["load_time"] = time.perf_counter() - t0
    return sample


class ProxyCache:
    """
    LRU cache of proxy samples, filled by background threads.

    want(paths) gives the dumps to load, most urgent first (it replaces the previous
    list); the workers start them in that order. on_ready(path, error) is called from a
    worker thread when one is done.
    """

    def __init__(self, reader, n_sample=400_000, budget_mb=1024, on_ready=None, workers=4, processes=False):
        self.reader = reader
        self.n_sample = n_sample
        self.budget = budget_mb * 1e6
        self.on_ready = on_ready
        self.entries = collections.OrderedDict()  # path -> sample, least recently used first
        self.errors = {}
        self.queue = []
        self.loading = set()
        self.load_time = 0.0
        self._cond = threading.Condition()
        self._stop = False
        # worker processes rebuild the reader from its registered name (data/readers.py READERS)
        self._pool = None
        if processes:
            self._pool = ProcessPoolExecutor(
                max(1, workers), mp_context=mp.get_context("spawn"), initializer=_exit_with_parent, initargs=(os.getpid(),)
            )
        self._threads = [
            threading.Thread(target=self._run, name=f"proxy-cache-{i}", daemon=True) for i in range(max(1, workers))
        ]
        for t in self._threads:
            t.start()

    @property
    def capacity(self):
        """Number of samples that fit in the memory budget (positions and smoothing lengths, 16 bytes a particle)."""
        return max(int(self.budget // (self.n_sample * 16 + 1024)), 2)

    def get(self, path):
        with self._cond:
            e = self.entries.get(path)
            if e is not None:
                self.entries.move_to_end(path)
            return e

    def has(self, path):
        with self._cond:
            return path in self.entries

    def want(self, paths):
        with self._cond:
            seen = set()
            self.queue = [p for p in paths if p and not (p in seen or seen.add(p))][: self.capacity]
            for p in reversed(self.queue):  # keep the wanted ones out of the eviction
                if p in self.entries:
                    self.entries.move_to_end(p)
            self._cond.notify_all()

    def nbytes(self):
        with self._cond:
            return sum(e["pos"].nbytes + (e["h"].nbytes if e["h"] is not None else 0) for e in self.entries.values())

    def close(self):
        """Stop reading. No on_ready call after this returns (its receiver may be gone)."""
        with self._cond:
            self._stop = True
            self.on_ready = None
            self._cond.notify_all()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _next(self):
        for p in self.queue:
            if p not in self.entries and p not in self.errors and p not in self.loading:
                return p
        return None

    def _run(self):
        while True:
            with self._cond:
                while not self._stop and self._next() is None:
                    self._cond.wait()
                if self._stop:
                    return
                path = self._next()
                self.loading.add(path)
            err = None
            try:
                if self._pool is not None:
                    sample = self._pool.submit(_read_sample, self.reader.name, path, self.n_sample).result()
                else:
                    t0 = time.perf_counter()
                    sample = proxy_sample(self.reader.read_gas(path), self.n_sample)
                    sample["load_time"] = time.perf_counter() - t0
            except Exception as e:  # unreadable dump: remembered, not retried
                sample, err = None, f"{type(e).__name__}: {e}"
            with self._cond:
                if self._stop:  # closed meanwhile (the pool's jobs are cancelled): nothing to report
                    return
                self.loading.discard(path)
                self._cond.notify_all()  # the others may be waiting for a free path
                if sample is None:
                    self.errors[path] = err
                else:
                    self.entries[path] = sample
                    self.load_time = sample["load_time"]
                    while len(self.entries) > self.capacity:
                        self.entries.popitem(last=False)
                on_ready = self.on_ready
            if on_ready:
                on_ready(path, err)
