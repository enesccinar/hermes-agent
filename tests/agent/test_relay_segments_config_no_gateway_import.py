"""Reading session-segment config must not import ``gateway.run`` in a non-gateway process.

``gateway.run``'s import-time bootstrap sets ``TERMINAL_CWD=$HOME`` (plus ``_HERMES_GATEWAY``
and ``HERMES_QUIET``); importing it from a CLI/one-shot turn re-homed the session out of its
launch directory, so scope guards saw ``~`` instead of the repo the user started in.
"""

from __future__ import annotations

import sys

from agent import relay_runtime


def test_segments_config_does_not_import_gateway_run(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(relay_runtime, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  telemetry:\n    session_segments:\n      on_compaction: true\n      max_turns: 7\n")

    cfg = relay_runtime._load_segments_config()

    assert cfg == {"on_compaction": True, "max_turns": 7}
    assert "gateway.run" not in sys.modules
    assert "TERMINAL_CWD" not in __import__("os").environ


def test_segments_config_defaults_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)
    monkeypatch.setattr(relay_runtime, "get_hermes_home", lambda: tmp_path)

    assert relay_runtime._load_segments_config() == {"on_compaction": False, "max_turns": 0}
    assert "gateway.run" not in sys.modules


def test_segments_config_reuses_loaded_gateway(monkeypatch):
    class _FakeGateway:
        @staticmethod
        def _load_gateway_config():
            return {"gateway": {"telemetry": {"session_segments": {"max_turns": 3}}}}

    monkeypatch.setitem(sys.modules, "gateway.run", _FakeGateway)

    assert relay_runtime._load_segments_config() == {"on_compaction": False, "max_turns": 3}
