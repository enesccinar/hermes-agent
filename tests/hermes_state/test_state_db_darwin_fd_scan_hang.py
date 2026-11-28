"""A kernel-blocked vnode must not wedge Hermes startup (#113187).

``_iter_darwin_fd_targets`` calls ``proc_pidfdinfo`` for every descriptor of
every process. If any process holds a descriptor whose vnode lookup blocks in
the kernel (e.g. uninterruptible I/O on a dead network share), that call never
returns -- and it used to run synchronously on the SessionDB connect path, so
one stuck unrelated process froze every Hermes start indefinitely.

``_iter_darwin_sidecar_holders`` must therefore bound the whole pass with a
wall-clock deadline (a per-iteration check cannot fire while blocked inside a
single call) and fail open with a warning. libproc is faked throughout, so
these run on every platform.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
import time

import hermes_state_dbfile as dbfile
from tests.waiting import wait_until
from hermes_state_dbfile import (
    _DARWIN_FD_INO_OFFSET,
    _DARWIN_FD_PATH_OFFSET,
    _DARWIN_FD_DEV_OFFSET,
    _DARWIN_FD_RECORD_SIZE,
    _iter_darwin_sidecar_holders,
)

# Far below the 5s production budget so the suite stays fast; far above zero
# so scheduling jitter cannot flake it.
BUDGET = 2.0
# Guard so a regression (scan blocks again) fails instead of hanging the suite.
HANG_GUARD = 30.0


class _HungLibproc:
    """One pid, one fd, whose vnode lookup never returns (state-U process)."""

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        payload = struct.pack("<iI", 7, 1)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        threading.Event().wait()
        return 0  # unreachable


class _HealthyLibproc:
    """One pid, one vnode fd naming ``target`` with the given identity."""

    def __init__(self, target, identity):
        self._target = target
        self._identity = identity

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        payload = struct.pack("<iI", 7, 1)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        blob = bytearray(_DARWIN_FD_RECORD_SIZE)
        struct.pack_into("<I", blob, _DARWIN_FD_DEV_OFFSET, self._identity[0])
        struct.pack_into("<Q", blob, _DARWIN_FD_INO_OFFSET, self._identity[1])
        encoded = self._target.encode("utf-8")
        blob[_DARWIN_FD_PATH_OFFSET:_DARWIN_FD_PATH_OFFSET + len(encoded)] = encoded
        ctypes.memmove(record, bytes(blob), len(blob))
        return 1


def _run_guarded(func, *args):
    """Run ``func`` so a hang fails the test instead of hanging the suite."""
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(func(*args)), daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(HANG_GUARD)
    return worker, time.monotonic() - started, outcome


def test_hung_vnode_scan_fails_open_within_budget(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: _HungLibproc())
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    if hasattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS"):
        monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", BUDGET)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        worker, elapsed, outcome = _run_guarded(
            _iter_darwin_sidecar_holders, tmp_path / "state.db"
        )
    assert not worker.is_alive(), "scan blocked past the hang guard: no deadline"
    assert elapsed < HANG_GUARD
    assert outcome == [[]], "a timed-out scan must fail open (no holders, no refusal)"
    assert any(
        "timed out" in record.getMessage() for record in caplog.records
    ), "deadline expiry must log a visible warning"


def test_healthy_scan_still_reports_holder(monkeypatch, tmp_path, caplog):
    """The timeout wrapper must not swallow a fast, successful scan."""
    watched = os.path.realpath(os.path.join(os.fspath(tmp_path), "state.db-wal"))
    # The watched path names nothing: stat fails, so the fd's identity is
    # judged a retired generation and the holder is reported.
    lib = _HealthyLibproc(watched, (0xDEAD, 0xBEEF))
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    if hasattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS"):
        monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", BUDGET)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        worker, elapsed, outcome = _run_guarded(
            _iter_darwin_sidecar_holders, tmp_path / "state.db"
        )
    assert not worker.is_alive()
    assert elapsed < BUDGET, "a fast scan must not wait out the budget"
    assert outcome == [[(4242, watched)]]
    assert not any(
        "timed out" in record.getMessage() for record in caplog.records
    ), "a successful scan must not log a timeout warning"


class _CountingLibproc(_HealthyLibproc):
    """Healthy, but every pass blocks until released and is counted."""

    def __init__(self, target, identity):
        super().__init__(target, identity)
        self.passes = 0
        self.release = threading.Event()

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        self.passes += 1
        self.release.wait()
        return super().proc_pidfdinfo(pid, fd, flavor, record, size)


def test_concurrent_callers_share_one_pass(monkeypatch, tmp_path):
    """A multiplexed gateway opens every profile's state.db at once: one pass serves them all."""
    watched = os.path.realpath(os.path.join(os.fspath(tmp_path), "state.db-wal"))
    lib = _CountingLibproc(watched, (0xDEAD, 0xBEEF))
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", BUDGET)
    scanning = lambda: sum(t.name == "hermes-wal-holder-scan" and t.is_alive() for t in threading.enumerate())
    before = scanning()  # an earlier test's deliberately hung pass may still be parked
    results: list = []
    callers = [
        threading.Thread(target=lambda: results.append(_iter_darwin_sidecar_holders(tmp_path / "state.db")), daemon=True)
        for _ in range(8)
    ]
    for c in callers:
        c.start()
    # Every caller is in before the pass may finish: it is still parked on the first descriptor.
    wait_until(lambda: lib.passes == 1 and sum(c.is_alive() for c in callers) == 8, timeout=HANG_GUARD)
    lib.release.set()
    for c in callers:
        c.join(HANG_GUARD)
    assert results == [[(4242, watched)]] * 8
    assert lib.passes == 1, "concurrent callers must share one libproc pass"
    deadline = time.monotonic() + HANG_GUARD
    while scanning() > before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert scanning() == before, "the shared pass must end once it has answered"


def test_overrun_pass_is_not_multiplied(monkeypatch, tmp_path, caplog):
    """After a pass overruns, later callers fail open at once instead of starting (or waiting on) more."""
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: _HungLibproc())
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", 0.3)
    before = sum(t.name == "hermes-wal-holder-scan" for t in threading.enumerate())
    assert _iter_darwin_sidecar_holders(tmp_path / "state.db") == []
    started = time.monotonic()
    for _ in range(20):
        assert _iter_darwin_sidecar_holders(tmp_path / "state.db") == []
    assert time.monotonic() - started < 0.3, "callers after an overrun must not wait out the budget again"
    after = sum(t.name == "hermes-wal-holder-scan" for t in threading.enumerate())
    assert after - before == 1, "only one abandoned pass, however many callers"


def test_finished_pass_is_not_reused(monkeypatch, tmp_path):
    """Results go stale once a pass ends: the next caller scans again (no cache across opens)."""
    watched = os.path.realpath(os.path.join(os.fspath(tmp_path), "state.db-wal"))
    lib = _CountingLibproc(watched, (0xDEAD, 0xBEEF))
    lib.release.set()
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_TIMEOUT_SECONDS", BUDGET)
    _iter_darwin_sidecar_holders(tmp_path / "state.db")
    _iter_darwin_sidecar_holders(tmp_path / "state.db")
    assert lib.passes == 2


def test_non_vnode_descriptors_are_skipped(monkeypatch, tmp_path):
    """Sockets and pipes have no pathname: no proc_pidfdinfo call for them."""

    class _SocketOnly(_HealthyLibproc):
        calls = 0

        def proc_pidinfo(self, pid, flavor, arg, buf, size):
            payload = struct.pack("<iI", 7, 2)  # PROX_FDTYPE_SOCKET
            ctypes.memmove(buf, payload, len(payload))
            return len(payload)

        def proc_pidfdinfo(self, *a):
            type(self).calls += 1
            return super().proc_pidfdinfo(*a)

    lib = _SocketOnly("/nowhere-wal", (1, 2))
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    assert _iter_darwin_sidecar_holders(tmp_path / "state.db") == []
    assert _SocketOnly.calls == 0
