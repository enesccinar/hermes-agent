"""World's central secret store (``$CINAR_WORLD_HOME/secrets/hermes-<profile>.json``) as a profile's
``.env`` layer: load, fallback, write, and that values never reach logs or stderr."""

import json
import logging
import os
import stat

import pytest

from agent.secret_scope import invalidate_env_file_cache, load_env_file
from hermes_cli import world_secrets
from hermes_cli.env_loader import load_hermes_dotenv

SECRET = "s3cr3t-value-never-printed"
KEYS = ("WS_TOKEN", "WS_LITERAL", "WS_NUM", "WS_DOTENV_ONLY", "WS_NEW", "HERMES_ACP_AUTH_METHOD")


@pytest.fixture
def world(tmp_path, monkeypatch):
    world_home = tmp_path / "cinar-world"
    monkeypatch.setenv("CINAR_WORLD_HOME", str(world_home))
    profile = tmp_path / ".hermes" / "profiles" / "cinar"
    profile.mkdir(parents=True)
    saved = {k: os.environ.get(k) for k in KEYS}
    for k in KEYS:
        os.environ.pop(k, None)
    world_secrets._WARNED.clear()
    invalidate_env_file_cache()
    yield world_home, profile
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    invalidate_env_file_cache()


def _write_scope(world_home, data, name="hermes-cinar"):
    secrets = world_home / "secrets"
    secrets.mkdir(parents=True, exist_ok=True)
    path = secrets / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_world_file_replaces_profile_dotenv(world):
    world_home, profile = world
    (profile / ".env").write_text("WS_DOTENV_ONLY=from-dotenv\nWS_TOKEN=old\n", encoding="utf-8")
    _write_scope(world_home, {"WS_TOKEN": SECRET, "WS_LITERAL": "a${HOME}b", "WS_NUM": 5, "WS_NULL": None})

    load_hermes_dotenv(hermes_home=profile)

    assert os.environ["WS_TOKEN"] == SECRET
    assert os.environ["WS_LITERAL"] == "a${HOME}b"  # World values are literal, never ${VAR}-expanded
    assert "WS_DOTENV_ONLY" not in os.environ
    scoped = load_env_file(profile / ".env")
    assert scoped["WS_TOKEN"] == SECRET and scoped["WS_NUM"] == "5"
    assert "WS_NULL" not in scoped and "WS_DOTENV_ONLY" not in scoped


def test_falls_back_to_dotenv_without_scope_file(world, capsys):
    world_home, profile = world
    (profile / ".env").write_text(f"WS_TOKEN={SECRET}\n", encoding="utf-8")
    (world_home / "secrets").mkdir(parents=True)  # World is set up, this profile not imported yet
    _write_scope(world_home, {"WS_TOKEN": "other"}, name="hermes-other")

    load_hermes_dotenv(hermes_home=profile)

    assert os.environ["WS_TOKEN"] == SECRET
    err = capsys.readouterr().err
    assert "hermes-cinar.json missing" in err and SECRET not in err


def test_non_profile_dotenv_is_never_mapped(world, tmp_path):
    world_home, _profile = world
    _write_scope(world_home, {"WS_TOKEN": SECRET}, name="hermes-default")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("WS_TOKEN=project\n", encoding="utf-8")
    assert world_secrets.scope_for_env_path(project / ".env") is None
    assert load_env_file(project / ".env") == {"WS_TOKEN": "project"}


def test_clear_known_keys_missing_follows_world_file(world):
    world_home, profile = world
    _write_scope(world_home, {"WS_TOKEN": SECRET})
    os.environ["HERMES_ACP_AUTH_METHOD"] = "inherited"
    load_hermes_dotenv(hermes_home=profile)
    assert "HERMES_ACP_AUTH_METHOD" not in os.environ

    _write_scope(world_home, {"WS_TOKEN": SECRET, "HERMES_ACP_AUTH_METHOD": "kept"})
    load_hermes_dotenv(hermes_home=profile)
    assert os.environ["HERMES_ACP_AUTH_METHOD"] == "kept"


def test_save_and_remove_write_the_world_file(world, monkeypatch):
    from hermes_cli.config import load_env, remove_env_value, save_env_value

    world_home, profile = world
    monkeypatch.setenv("HERMES_HOME", str(profile))
    scope = _write_scope(world_home, {"WS_TOKEN": "old"})

    save_env_value("WS_NEW", SECRET)
    save_env_value("WS_TOKEN", "rotated")

    data = json.loads(scope.read_text(encoding="utf-8"))
    assert data == {"WS_NEW": SECRET, "WS_TOKEN": "rotated"}
    assert stat.S_IMODE(scope.stat().st_mode) == 0o600
    assert not (profile / ".env").exists()
    assert (profile / ".env.lock").exists()  # the existing .env write lock still serializes writers
    assert os.environ["WS_NEW"] == SECRET
    assert load_env()["WS_TOKEN"] == "rotated"
    assert [p.name for p in scope.parent.iterdir()] == ["hermes-cinar.json"]  # no temp file left

    assert remove_env_value("WS_NEW") is True
    assert remove_env_value("WS_NEW") is False
    assert "WS_NEW" not in json.loads(scope.read_text(encoding="utf-8"))
    assert "WS_NEW" not in os.environ


def test_save_uses_dotenv_when_no_scope_file(world, monkeypatch):
    from hermes_cli.config import save_env_value

    world_home, profile = world
    monkeypatch.setenv("HERMES_HOME", str(profile))
    save_env_value("WS_NEW", "plain")
    assert "WS_NEW=plain" in (profile / ".env").read_text(encoding="utf-8")
    assert not (world_home / "secrets").exists()


def test_values_never_logged(world, monkeypatch, caplog, capsys):
    from hermes_cli.config import save_env_value

    world_home, profile = world
    monkeypatch.setenv("HERMES_HOME", str(profile))
    caplog.set_level(logging.DEBUG)
    scope = _write_scope(world_home, {"WS_TOKEN": SECRET})
    load_hermes_dotenv(hermes_home=profile)
    save_env_value("WS_NEW", SECRET)
    scope.write_text(f'{{"WS_TOKEN": "{SECRET}"', encoding="utf-8")  # corrupt: warns, names no values
    assert load_env_file(profile / ".env") == {}

    out = capsys.readouterr()
    assert "not valid JSON" in out.err
    assert SECRET not in caplog.text and SECRET not in out.err and SECRET not in out.out
