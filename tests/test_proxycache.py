# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

import threading

import numpy as np
import pytest
from conftest import NGAS

from plouf.data.proxycache import ProxyCache, particle_rank, proxy_sample
from plouf.data.readers import get_reader

N = 1000


def _ids(sim, sample_pos):
    """Particle ids of the sampled positions (positions are unique in the synthetic dumps)."""
    lut = {tuple(p): i for p, i in zip(sim.pos.astype(np.float32), sim.iorig)}
    return {lut[tuple(p)] for p in sample_pos}


def test_rank_is_fixed():
    ids = np.arange(100)
    np.testing.assert_array_equal(particle_rank(ids), particle_rank(ids.copy()))
    assert len(set(particle_rank(ids).tolist())) == 100


def test_sample_size_and_weight(sim):
    g = get_reader().read_gas(sim.dumps[2])
    s = proxy_sample(g, N)
    nalive = NGAS - len(sim.dead[2])
    assert s["pos"].shape == (N, 3) and s["pos"].dtype == np.float32
    assert s["nalive"] == nalive
    # each sampled particle carries (living gas mass) / n, in Msun/pc^2 splat units (code units are pc, Msun)
    assert s["weight"] == pytest.approx(1e-3 * nalive / N)


def test_sample_is_stable_and_replenished(sim):
    """Same particles from one dump to the next; accreted ones are replaced, the size stays n."""
    rd = get_reader()
    a = _ids(sim, proxy_sample(rd.read_gas(sim.dumps[0]), N)["pos"])
    b = _ids(sim, proxy_sample(rd.read_gas(sim.dumps[3]), N)["pos"])
    assert len(a) == len(b) == N
    dead = set(sim.iorig[sim.dead[3]].tolist())
    assert a - dead <= b  # every survivor is still in the sample
    assert not (b & dead)  # no accreted particle
    assert len(b - a) == len(a & dead)  # each accreted one replaced by exactly one newcomer


def test_cache_loads_wanted_dumps(sim):
    done = threading.Event()
    got = []

    def ready(path, err):
        got.append((path, err))
        if len(got) == 2:
            done.set()

    pc = ProxyCache(get_reader(), N, budget_mb=100, on_ready=ready, workers=1)  # one reader: in order
    try:
        pc.want([sim.dumps[1], sim.dumps[0]])
        assert done.wait(20)
        assert [p for p, _ in got] == [sim.dumps[1], sim.dumps[0]]  # most urgent first
        assert pc.has(sim.dumps[0]) and pc.get(sim.dumps[1])["pos"].shape == (N, 3)
        assert pc.nbytes() == 2 * N * 16  # positions and smoothing lengths, float32
    finally:
        pc.close()


def test_cache_evicts_least_recently_used(sim):
    ev = threading.Event()
    pc = ProxyCache(get_reader(), N, budget_mb=0, on_ready=lambda p, e: ev.set())  # capacity: 2 samples
    try:
        assert pc.capacity == 2
        for d in sim.dumps[:3]:
            ev.clear()
            pc.want([d])
            assert ev.wait(20)
        assert not pc.has(sim.dumps[0]) and pc.has(sim.dumps[1]) and pc.has(sim.dumps[2])
    finally:
        pc.close()


def test_cache_remembers_unreadable_dumps(tmp_path):
    ev = threading.Event()
    bad = str(tmp_path / "missing_dump")
    pc = ProxyCache(get_reader(), N, on_ready=lambda p, e: ev.set())
    try:
        pc.want([bad])
        assert ev.wait(20)
        assert bad in pc.errors and "FileNotFoundError" in pc.errors[bad]
        assert not pc.has(bad)
    finally:
        pc.close()


def test_several_readers_load_everything_once(sim):
    lock = threading.Lock()
    got = []
    done = threading.Event()

    def ready(path, err):
        with lock:
            got.append(path)
            if len(got) == len(sim.dumps):
                done.set()

    pc = ProxyCache(get_reader(), N, budget_mb=100, on_ready=ready, workers=4)
    try:
        pc.want(sim.dumps)
        assert done.wait(20)
        assert sorted(got) == sorted(sim.dumps)  # each dump read once
        assert all(pc.has(d) for d in sim.dumps) and not pc.loading
    finally:
        pc.close()


def test_sample_smoothing_lengths_grow_with_the_sampling(sim):
    """n of N particles: kernels (N/n)^(1/3) wider, so the sample fills the same volume."""
    g = get_reader().read_gas(sim.dumps[2])
    s = proxy_sample(g, N)
    nalive = NGAS - len(sim.dead[2])
    assert s["h"].shape == (N,) and s["h"].dtype == np.float32
    np.testing.assert_allclose(s["h"], 0.05 * (nalive / N) ** (1 / 3), rtol=1e-6)  # all h = 0.05 in the synthetic dumps


def test_reading_in_worker_processes(sim):
    done = threading.Event()
    got = []

    def ready(path, err):
        got.append((path, err))
        if len(got) == 2:
            done.set()

    pc = ProxyCache(get_reader(), N, budget_mb=100, on_ready=ready, workers=2, processes=True)
    try:
        pc.want(sim.dumps[:2])
        assert done.wait(60)
        assert all(err is None for _, err in got)
        a = pc.get(sim.dumps[0])
        b = proxy_sample(get_reader().read_gas(sim.dumps[0]), N)
        np.testing.assert_array_equal(a["pos"], b["pos"])  # the same sample as read in this process
        np.testing.assert_array_equal(a["h"], b["h"])
    finally:
        pc.close()


def test_reader_processes_exit_with_their_parent(sim, tmp_path):
    """A killed editor leaves no reader process behind."""
    import os
    import subprocess
    import sys
    import time

    pids = tmp_path / "pids"
    code = f"""
import os, threading
from plouf.data.proxycache import ProxyCache
from plouf.data.readers import get_reader
ev = threading.Event()
pc = ProxyCache(get_reader(), 100, on_ready=lambda p, e: ev.set(), workers=2, processes=True)
pc.want([{sim.dumps[0]!r}])
ev.wait(60)
open({str(pids)!r}, "w").write(" ".join(str(p) for p in pc._pool._processes))
os._exit(0)  # like a crash: no close()
"""
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)), timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    workers = [int(p) for p in pids.read_text().split()]
    assert workers

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    t = time.perf_counter()
    while any(alive(p) for p in workers) and time.perf_counter() - t < 10:
        time.sleep(0.1)
    assert not any(alive(p) for p in workers)
