"""Regression tests for #101113 — a credentialless satellite profile under
``gateway.profile_routes`` delivers cron output through the PRIMARY adapter
for exactly the targets the primary routes to it, and fails closed otherwise.

The multiplex ticker hands such a profile a ``SharedRouteAdapters`` view over
the primary adapter map; ``_deliver_result`` resolves a transport from it per
target using the same ``ProfileRoute.matches`` predicate as inbound routing.
"""
import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

from cron.scheduler import _deliver_result
from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PRIMARY_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "fit", "platform": "discord", "chat_id": "1543065293755256852", "profile": "fitness"},
            {"name": "off", "platform": "discord", "chat_id": "999", "profile": "fitness", "enabled": False},
            {"name": "other", "platform": "discord", "chat_id": "777", "profile": "other"},
        ],
    }
}


def _job(chat_id: str) -> dict:
    return {"id": "a7ae1520356c", "name": "brief", "deliver": f"discord:{chat_id}"}


def _run(job, adapters):
    """Drive ``_deliver_result`` with a live loop and a real DeliveryRouter."""
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    standalone = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "DISCORD_BOT_TOKEN is not set"}

    config = MagicMock()
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, "hello", adapters=adapters, loop=loop)
    return error, standalone


def _primary_adapter():
    adapter = MagicMock()
    adapter.sent = []

    async def send(chat_id, content, metadata=None):
        adapter.sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    adapter.send = send
    return adapter


def test_satellite_routes_exact_target_through_primary_adapter(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fitness_home = root / "profiles" / "fitness"
    fitness_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(PRIMARY_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(fitness_home))
    try:
        shared = SharedRouteAdapters(
            {Platform.DISCORD: primary}, _primary_profile_routes_for_current_home()
        )
        # exact enabled route → primary adapter sends, no standalone attempt
        error, standalone = _run(_job("1543065293755256852"), shared)
        assert error is None, error
        assert primary.sent == ["1543065293755256852"]
        assert standalone == []

        # unmatched chat, disabled route, route for another profile → the
        # primary bot is NEVER used; delivery stays on the satellite's own
        # (credentialless) standalone path and reports its failure.
        for chat in ("424242", "999", "777"):
            primary.sent.clear()
            error, standalone = _run(_job(chat), shared)
            assert error is not None and "DISCORD_BOT_TOKEN" in error
            assert primary.sent == []
            assert standalone == [chat]
    finally:
        reset_hermes_home_override(token)


def test_shared_view_is_falsy_without_routes_or_primary_adapters():
    assert not SharedRouteAdapters({}, [])
    assert SharedRouteAdapters({Platform.DISCORD: object()}, []).get(Platform.DISCORD) is None


def test_guild_scoped_route_authorizes_cron_target_even_when_satellite_has_no_platform_block(
    tmp_path, monkeypatch,
):
    """The documented Discord route shape is ``guild_id + chat_id``. A cron target carries no guild
    anchor, so the route must be matched on its target-exact discriminators; and the satellite's
    missing/disabled ``platforms.discord`` block must not veto the PRIMARY's authorized transport
    (#89302 sibling) — before, both fell to standalone "DISCORD_BOT_TOKEN is not set"."""
    root = tmp_path / "root"
    sat_home = root / "profiles" / "fitness"
    sat_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"multiplex_profiles": True, "profile_routes": [
            {"platform": "discord", "guild_id": "G1", "chat_id": "C1", "profile": "fitness"}]},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    primary = _primary_adapter()

    token = set_hermes_home_override(str(sat_home))
    try:
        shared = SharedRouteAdapters({Platform.DISCORD: primary}, _primary_profile_routes_for_current_home())
        for satellite_platforms in ({}, {Platform.DISCORD: PlatformConfig(enabled=False)}):
            primary.sent.clear()
            config = MagicMock()
            config.platforms = satellite_platforms
            config.get_home_channel = lambda p: None
            with patch("gateway.config.load_gateway_config", return_value=config):
                error, standalone = _run(_job("C1"), shared)
            assert error is None, error
            assert primary.sent == ["C1"] and standalone == []
    finally:
        reset_hermes_home_override(token)


def test_live_native_adapter_without_platform_block_is_not_treated_as_disabled():
    """#89302: a live native adapter handed in by the gateway is the authorization; an absent
    ``platforms.<p>`` block in the firing profile means "no config", not "disabled"."""
    from cron.scheduler_delivery import _resolve_target_transport

    config = MagicMock()
    config.platforms = {}
    adapter = object()
    resolved, err = _resolve_target_transport(
        {"id": "j"}, Platform.DISCORD, "discord", {"platform": "discord", "chat_id": "C1"},
        {Platform.DISCORD: adapter}, config)
    assert err is None and resolved[2] is adapter and resolved[1].enabled
    # an explicitly disabled block still vetoes
    config.platforms = {Platform.DISCORD: PlatformConfig(enabled=False)}
    resolved, err = _resolve_target_transport(
        {"id": "j"}, Platform.DISCORD, "discord", {"platform": "discord", "chat_id": "C1"},
        {Platform.DISCORD: adapter}, config)
    assert resolved is None and "not configured/enabled" in err


def test_multiplex_tick_routes_desktop_served_satellite_with_only_local_adapter(
    tmp_path, monkeypatch,
):
    """#139825: a satellite profile also chatted with from Hermes Desktop holds
    ``{LOCAL: adapter}`` in ``_profile_adapters``. LOCAL carries no platform credential, so its
    cron tick must still hand it the routed primary view — before, the non-empty map skipped
    ``SharedRouteAdapters`` and delivery fell through to the profile's own credentialless config
    (``platform 'discord' not configured/enabled``)."""
    import threading

    from cron.scheduler_provider import InProcessCronScheduler
    from cron.scheduler_preflight import SharedRouteAdapters

    root = tmp_path / "root"
    sat_home = root / "profiles" / "sat"
    (sat_home / "cron").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"multiplex_profiles": True, "profile_routes": [
            {"platform": "discord", "guild_id": "G1", "chat_id": "C1", "profile": "sat"}]},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    primary = _primary_adapter()
    captured = {}
    stop = threading.Event()

    def _tick_and_stop(*_args, **kwargs):
        captured["adapters"] = kwargs["adapters"]
        stop.set()

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_tick_and_stop):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("sat", sat_home)],
                "adapters": {Platform.DISCORD: primary},
                "profile_adapters": {"sat": {Platform.LOCAL: object()}},
                "default_profile": "default",
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=10)
        stop.set()
        thread.join(timeout=10)

    assert not thread.is_alive()
    shared = captured["adapters"]
    assert isinstance(shared, SharedRouteAdapters)
    # The routed view releases the primary adapter for exactly the route-mapped target.
    assert shared.get(Platform.DISCORD, {"chat_id": "C1"}) is primary
    assert shared.get(Platform.DISCORD, {"chat_id": "unmapped"}) is None


# --- bot_profile routes: the satellite rides the NAMED secondary bot, not the primary ---------------

BOT_ROUTE_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "sat-ch", "platform": "discord", "bot_profile": "botowner", "guild_id": "G1",
             "chat_id": "C1", "profile": "sat"},
            {"name": "prim-ch", "platform": "discord", "guild_id": "G1", "chat_id": "C2", "profile": "sat"},
        ],
    }
}


def _bot_route_home(tmp_path, monkeypatch):
    root = tmp_path / "root"
    sat_home = root / "profiles" / "sat"
    (sat_home / "cron").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(BOT_ROUTE_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return sat_home


def test_bot_profile_route_resolves_the_named_secondary_bot(tmp_path, monkeypatch):
    """A route with ``bot_profile: botowner`` is served inbound by botowner's bot, so cron output for
    that exact target must go out through botowner's bot. Before, the view only knew the primary
    map: with no primary Discord bot every such cron send fell to standalone
    (``DISCORD_BOT_TOKEN is not set``)."""
    sat_home = _bot_route_home(tmp_path, monkeypatch)
    primary, bot = _primary_adapter(), _primary_adapter()
    token = set_hermes_home_override(str(sat_home))
    try:
        routes = _primary_profile_routes_for_current_home()
        shared = SharedRouteAdapters({}, routes, bot_adapters={"botowner": {Platform.DISCORD: bot}})
        assert shared
        assert shared.get(Platform.DISCORD, {"chat_id": "C1"}) is bot
        assert shared.get(Platform.DISCORD, {"chat_id": "C2"}) is None  # primary route, no primary bot
        assert shared.get(Platform.DISCORD, {"chat_id": "C9"}) is None
        error, standalone = _run(_job("C1"), shared)
        assert error is None, error
        assert bot.sent == ["C1"] and standalone == []

        # each route only releases ITS bot: the primary never serves the botowner route, nor botowner
        # the primary route.
        both = SharedRouteAdapters({Platform.DISCORD: primary}, routes,
                                   bot_adapters={"botowner": {Platform.DISCORD: bot}})
        assert both.get(Platform.DISCORD, {"chat_id": "C1"}) is bot
        assert both.get(Platform.DISCORD, {"chat_id": "C2"}) is primary
        # botowner's bot down: fail closed, never the primary
        down = SharedRouteAdapters({Platform.DISCORD: primary}, routes, bot_adapters={"botowner": {}})
        assert down.get(Platform.DISCORD, {"chat_id": "C1"}) is None
        assert not SharedRouteAdapters({}, routes, bot_adapters={"botowner": {}})
    finally:
        reset_hermes_home_override(token)


def test_multiplex_tick_hands_bot_profile_satellite_the_named_bot(tmp_path, monkeypatch):
    """The ticker must build the routed view even when the PRIMARY map has no bot at all (the only
    Discord bot belongs to a secondary profile) — before, ``and adapters`` skipped it."""
    import threading

    from cron.scheduler_provider import InProcessCronScheduler

    sat_home = _bot_route_home(tmp_path, monkeypatch)
    bot = _primary_adapter()
    captured = {}
    stop = threading.Event()

    def _tick_and_stop(*_args, **kwargs):
        captured["adapters"] = kwargs["adapters"]
        stop.set()

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_tick_and_stop):
        thread = threading.Thread(
            target=provider.start, args=(stop,),
            kwargs={
                "interval": 0, "profile_homes": [("sat", sat_home)], "adapters": {},
                "profile_adapters": {"sat": {}, "botowner": {Platform.DISCORD: bot}},
                "default_profile": "default",
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=10)
        stop.set()
        thread.join(timeout=10)

    shared = captured["adapters"]
    assert isinstance(shared, SharedRouteAdapters)
    assert shared.get(Platform.DISCORD, {"chat_id": "C1"}) is bot
    assert shared.get(Platform.DISCORD, {"chat_id": "C9"}) is None
