"""World's central secret store as a Hermes profile's secret source.

A profile home ``<root>/profiles/<name>`` (or the root itself, profile ``default``) maps to the World
scope ``hermes-<name>``: ``$CINAR_WORLD_HOME/secrets/hermes-<name>.json`` (``~/.cinar-world`` when the
variable is unset), a flat JSON object of string values, mode 0600.

One rule decides both directions: when that scope file exists it IS the profile's ``.env`` layer --
every reader of ``<home>/.env`` sees its keys instead of the file's, and ``save_env_value`` /
``remove_env_value`` write it (under the same ``.env.lock``). When it does not exist, Hermes uses the
``.env`` exactly as before; if the World secrets directory exists while the profile still has a
``.env``, a one-line note on stderr says the scope has not been imported yet.

Only key names and paths are ever printed or logged, never values.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

_SCOPE_RE = re.compile(r"^[a-z0-9-]+$")
_CACHE: dict[str, tuple[bytes, dict[str, str]]] = {}
_CACHE_LOCK = threading.Lock()
_WARNED: set[str] = set()


def world_home() -> Path:
    raw = os.environ.get("CINAR_WORLD_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".cinar-world"


def secrets_dir() -> Path:
    return world_home() / "secrets"


def _profile_name(home: Path) -> str | None:
    """``<root>/profiles/<name>`` → name; the Hermes root itself → ``default``; anything else None."""
    from hermes_constants import _is_hermes_profiles_root, get_default_hermes_root

    try:
        if home.parent.name == "profiles" and _is_hermes_profiles_root(home.parent):
            return home.name
        if home.resolve(strict=False) == get_default_hermes_root().resolve(strict=False):
            return "default"
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def scope_for_env_path(env_path: str | os.PathLike) -> str | None:
    """The World scope a profile ``.env`` path stands for, or None for any other dotenv file."""
    path = Path(env_path)
    if path.name != ".env":
        return None
    name = _profile_name(path.parent)
    scope = f"hermes-{name}" if name else None
    return scope if scope and _SCOPE_RE.match(scope) else None


def secrets_file_for_env_path(env_path: str | os.PathLike) -> Path | None:
    """The World scope file that replaces *env_path*, when it exists; else None (use the ``.env``)."""
    scope = scope_for_env_path(env_path)
    if scope is None:
        return None
    target = secrets_dir() / f"{scope}.json"
    if target.is_file():
        return target
    if secrets_dir().is_dir() and Path(env_path).exists():
        _warn_once(f"fallback:{env_path}",
                   f"  World secrets: {target} missing; {env_path} still in use "
                   f"(world-secret import {scope} {env_path})")
    return None


def env_layer_exists(env_path: str | os.PathLike) -> bool:
    """The profile's ``.env`` layer exists: the World scope file that replaces it, or the file itself."""
    return secrets_file_for_env_path(env_path) is not None or Path(env_path).exists()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(message, file=sys.stderr)


def _as_string(value) -> str | None:
    if value is None or isinstance(value, list):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def _decode(raw: bytes, path: Path) -> dict[str, str]:
    try:
        data = json.loads(raw.decode("utf-8-sig") or "{}")
    except (UnicodeDecodeError, ValueError):
        _warn_once(f"bad:{path}", f"  World secrets: {path} is not valid JSON; loading no keys from it")
        return {}
    if not isinstance(data, dict):
        _warn_once(f"bad:{path}", f"  World secrets: {path} is not a JSON object; loading no keys from it")
        return {}
    out: dict[str, str] = {}
    for key, value in data.items():
        text = _as_string(value)
        if isinstance(key, str) and text is not None:
            out[key] = text
    return out


def read_file(path: Path) -> dict[str, str]:
    """The scope file's keys as strings (memoised on its bytes). Unreadable → ``{}``."""
    key = str(path)
    try:
        raw = path.read_bytes()
    except OSError:
        with _CACHE_LOCK:
            _CACHE.pop(key, None)
        return {}
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == raw:
            return dict(cached[1])
    values = _decode(raw, path)
    with _CACHE_LOCK:
        _CACHE[key] = (raw, dict(values))
    return values


def read_for_env_path(env_path: str | os.PathLike) -> dict[str, str] | None:
    """The World values standing in for *env_path*, or None when the ``.env`` is still the source."""
    target = secrets_file_for_env_path(env_path)
    return None if target is None else read_file(target)


def write_file(path: Path, updates: dict[str, str | None]) -> None:
    """Apply *updates* (None removes a key) to the scope file: temp file + fsync + rename, mode 0600.
    The caller holds the profile's ``.env.lock`` (``hermes_cli.config._env_write_lock``)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
    except FileNotFoundError:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object; refusing to overwrite it")
    for key, value in updates.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    from utils import atomic_replace, mkstemp_beside

    fd, tmp = mkstemp_beside(path, suffix=".tmp", prefix=f".{path.stem}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)
    with _CACHE_LOCK:
        _CACHE.pop(str(path), None)


def save_for_env_path(env_path: Path, key: str, value: str | None) -> bool | None:
    """``save_env_value`` / ``remove_env_value`` (value None) for a profile whose World scope file
    replaces its ``.env``: under the ``.env.lock``, then published like a ``.env`` write. None when
    the ``.env`` is still the store; else whether ``key`` was present before."""
    target = secrets_file_for_env_path(env_path)
    if target is None:
        return None
    from hermes_cli.config import _env_write_lock, _publish_env_value, invalidate_env_cache

    with _env_write_lock(Path(env_path)):
        found = key in read_file(target)
        write_file(target, {key: value})
    _publish_env_value(key, value)
    invalidate_env_cache()
    return found
