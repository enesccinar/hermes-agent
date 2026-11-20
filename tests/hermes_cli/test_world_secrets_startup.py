"""A profile whose only secret layer is its World scope file (no ``<profile>/.env`` at all) through
the real startup paths, each in a fresh interpreter: the CLI entry point (``--profile`` switch, the
import-time dotenv load) and the multiplex gateway's routed turn (the shared gateway that runs every
CLI/TUI session). Both must resolve ``api_key: ${VAR}`` in config.yaml to the World value.

Plus the readers that used to stat ``<profile>/.env`` directly. Values are compared by hash only."""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_NAME = "WS_ROUTER_TEST_KEY"
SECRET = "ws-router-secret-never-printed"
SECRET_HASH = hashlib.sha256(SECRET.encode()).hexdigest()

_RESULT = """
import hashlib, json, os
from hermes_cli.runtime_provider import resolve_runtime_provider
r = resolve_runtime_provider()
key = r.get("api_key") or ""
print("RESULT " + json.dumps({"provider": r.get("provider"), "base_url": r.get("base_url"),
                              "key_sha": hashlib.sha256(key.encode()).hexdigest(),
                              "env_sha": hashlib.sha256(os.environ.get(%r, "").encode()).hexdigest()}))
""" % KEY_NAME


@pytest.fixture
def world_only_profile(tmp_path):
    """``<tmp>/home/.hermes`` with profile ``cinar`` whose secrets live only in World's store."""
    home = tmp_path / "home"
    root = home / ".hermes"
    profile = root / "profiles" / "cinar"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: base-model\n", encoding="utf-8")
    (profile / "config.yaml").write_text(textwrap.dedent(f"""\
        model:
          provider: custom
          default: world-test-model
          base_url: http://127.0.0.1:9/v1
          api_key: ${{{KEY_NAME}}}
        """), encoding="utf-8")
    world = tmp_path / "cinar-world"
    (world / "secrets").mkdir(parents=True)
    scope = world / "secrets" / "hermes-cinar.json"
    scope.write_text(json.dumps({KEY_NAME: SECRET}), encoding="utf-8")
    scope.chmod(0o600)
    assert not (profile / ".env").exists()
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_") and k != KEY_NAME}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(root), CINAR_WORLD_HOME=str(world),
               PYTHONPATH=str(REPO_ROOT), HERMES_TEST_ISOLATION=os.environ.get("HERMES_TEST_ISOLATION", "1"))
    return root, profile, world, env


def _run(code: str, env: dict, cwd: Path) -> dict:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=180, cwd=cwd, env=env, stdin=subprocess.DEVNULL, check=False)
    assert SECRET not in result.stdout and SECRET not in result.stderr
    assert result.returncode == 0, result.stderr[-4000:]
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT "):])


def test_cli_startup_with_only_world_scope(world_only_profile, tmp_path):
    """``hermes --profile cinar …``: the profile switch, then main's import-time dotenv load."""
    _root, _profile, _world, env = world_only_profile
    code = "import sys\nsys.argv = ['hermes', '--profile', 'cinar', 'chat']\nimport hermes_cli.main\n" + _RESULT
    out = _run(code, env, tmp_path)
    assert out["provider"] == "custom" and out["base_url"] == "http://127.0.0.1:9/v1"
    assert out["env_sha"] == SECRET_HASH
    assert out["key_sha"] == SECRET_HASH


def test_gateway_routed_turn_with_only_world_scope(world_only_profile, tmp_path):
    """The multiplex gateway (launch home = the root) serving a turn for the routed ``cinar``
    profile: credentials come from the profile's secret scope, never the gateway's own env."""
    _root, profile, _world, env = world_only_profile
    code = textwrap.dedent(f"""\
        from pathlib import Path
        import gateway.run as gr
        from agent.secret_scope import set_multiplex_active
        set_multiplex_active(True)
        with gr._profile_runtime_scope(Path({str(profile)!r})):
        """) + textwrap.indent(_RESULT, "    ")
    out = _run(code, env, tmp_path)
    assert out["provider"] == "custom" and out["base_url"] == "http://127.0.0.1:9/v1"
    assert out["env_sha"] != SECRET_HASH  # never published into the shared gateway process env
    assert out["key_sha"] == SECRET_HASH


# ── readers that used to stat <profile>/.env directly ─────────────────────────────────────────


@pytest.fixture
def world(tmp_path, monkeypatch):
    from agent.secret_scope import invalidate_env_file_cache
    from hermes_cli import world_secrets

    world_home = tmp_path / "cinar-world"
    (world_home / "secrets").mkdir(parents=True)
    monkeypatch.setenv("CINAR_WORLD_HOME", str(world_home))
    profile = tmp_path / ".hermes" / "profiles" / "cinar"
    profile.mkdir(parents=True)
    world_secrets._WARNED.clear()
    invalidate_env_file_cache()
    yield world_home / "secrets" / "hermes-cinar.json", profile
    invalidate_env_file_cache()


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_provider_key_check_reads_world_scope(world):
    from hermes_cli.main import _dotenv_has_provider_key

    scope, profile = world
    assert not _dotenv_has_provider_key(profile / ".env", {"OPENROUTER_API_KEY"})
    _write(scope, {"OPENROUTER_API_KEY": SECRET})
    assert _dotenv_has_provider_key(profile / ".env", {"OPENROUTER_API_KEY"})
    _write(scope, {"OPENROUTER_API_KEY": "  "})
    assert not _dotenv_has_provider_key(profile / ".env", {"OPENROUTER_API_KEY"})


def test_terminal_scope_reads_world_scope(world):
    from tools.terminal_scope import build_profile_terminal_scope

    scope, profile = world
    _write(scope, {"TERMINAL_CWD": "/srv/world-cwd"})
    assert build_profile_terminal_scope(profile)["TERMINAL_CWD"] == "/srv/world-cwd"


def test_serve_signature_follows_world_scope(world):
    from gateway.run_profile_reconcile import profile_serve_signature

    scope, profile = world
    _write(scope, {"DISCORD_BOT_TOKEN": "one"})
    before = profile_serve_signature(profile)
    _write(scope, {"DISCORD_BOT_TOKEN": "two-rotated"})
    assert profile_serve_signature(profile) != before


def test_channel_reads_and_env_backfill_respect_world_scope(world, monkeypatch):
    from hermes_cli import profile_channels, profiles

    scope, profile = world
    _write(scope, {"DISCORD_BOT_TOKEN": SECRET})
    assert profile_channels._env_values(profile / ".env") == {"DISCORD_BOT_TOKEN": SECRET}
    assert "discord" in profile_channels.channel_platforms_configured(profile)

    monkeypatch.setattr(profiles, "_iter_named_profile_dirs", lambda: [profile])
    assert profiles.backfill_profile_envs(quiet=True) == []
    assert not (profile / ".env").exists()  # no placeholder .env beside the World scope
