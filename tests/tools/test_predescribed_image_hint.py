"""Pre-described image notes must steer the model away from a full-image re-read.

The gateway, CLI and non-vision agent paths describe an attached image with the
auxiliary vision model before the main model sees it. The note they append is the
only hint the main model gets about what to do next. A bare "if you need a closer
look, use vision_analyze" made models immediately re-run vision_analyze on the
whole original (measured: 190 s on a local qwen, a second paid call on Gemini).
"""

import json
from unittest.mock import AsyncMock, patch

from tools.vision_tools import predescribed_image_followup_hint


def _assert_steers_to_crop(note: str, path: str) -> None:
    low = note.lower()
    assert path in note, "path must stay so a follow-up crop is possible"
    assert "already" in low, "must say the description already came from the vision model"
    assert "region" in low, "must point at a cropped follow-up (region)"
    assert "do not" in low and "whole image" in low, "must forbid a full-image re-read"
    assert "mac-ocr" in low, "must offer the free on-device OCR for exact text"


def test_hint_helper_text():
    _assert_steers_to_crop(predescribed_image_followup_hint("/tmp/x.png"), "/tmp/x.png")


def test_gateway_note_steers_to_crop():
    import asyncio
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A cat on a chair."}),
    ):
        result = asyncio.run(runner._enrich_message_with_vision(user_text="What is it?", image_paths=["/tmp/cat.png"]))
    assert "A cat on a chair." in result
    _assert_steers_to_crop(result, "/tmp/cat.png")


def test_agent_fallback_note_steers_to_crop(tmp_path):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = "custom"
    agent.model = "text-only"
    agent._anthropic_image_fallback_cache = {}
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A dashboard."}),
    ):
        note = agent._describe_image_for_anthropic_fallback(str(img), "user")
    assert "A dashboard." in note
    _assert_steers_to_crop(note, str(img))


def test_cli_note_steers_to_crop(tmp_path):
    from hermes_cli import cli_stream_mixin

    class _Stub(cli_stream_mixin.CLIStreamMixin):  # type: ignore[misc]
        pass

    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    stub = object.__new__(_Stub)

    async def _fake(**kwargs):
        return json.dumps({"success": True, "analysis": "A chart."})

    with patch("tools.vision_tools.vision_analyze_tool", side_effect=_fake):
        out = stub._preprocess_images_with_vision("explain", [img], announce=False)
    assert "A chart." in out
    _assert_steers_to_crop(out, str(img))
