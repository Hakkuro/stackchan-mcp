"""Tests for the Edge TTS engine (edge_tts_engine.py).

All tests mock the ``edge_tts`` library and ``av`` decoder so they run
without network access or native dependencies.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from stackchan_mcp.tts.edge_tts_engine import (
    DEFAULT_EDGE_TTS_RATE,
    DEFAULT_EDGE_TTS_VOICE,
    DEFAULT_EDGE_TTS_VOLUME,
    EdgeTTSEngine,
    _mp3_to_pcm16_mono,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fake MP3 bytes — content doesn't matter because we mock the decoder.
FAKE_MP3 = b"\xff\xfb\x90\x00" * 100

# 60 ms of silence at 16 kHz mono = 960 samples = 1920 bytes
FAKE_PCM = b"\x00\x00" * 960


def _make_fake_edge_tts_module(
    mp3_data: bytes = FAKE_MP3,
    *,
    empty: bool = False,
    capture_voices: list | None = None,
    capture_params: list | None = None,
):
    """Build a fake ``edge_tts`` module with a ``Communicate`` class."""

    class FakeCommunicate:
        def __init__(self, text, voice, *, rate="+0%", volume="+0%"):
            self.text = text
            self.voice = voice
            self.rate = rate
            self.volume = volume
            if capture_voices is not None:
                capture_voices.append(voice)
            if capture_params is not None:
                capture_params.append({"rate": rate, "volume": volume})

        async def stream(self):
            if not empty:
                yield {"type": "audio", "data": mp3_data}
            # edge-tts also yields metadata chunks; we must skip them
            yield {
                "type": "WordBoundary",
                "offset": 0,
                "duration": 100,
                "text": "hello",
            }

    module = SimpleNamespace(Communicate=FakeCommunicate)
    return module


# ---------------------------------------------------------------------------
# Defaults / Configuration
# ---------------------------------------------------------------------------


def test_engine_name_is_edge_tts():
    """Registry uses ``name`` to look up engines from the say tool's voice arg."""
    engine = EdgeTTSEngine()
    assert engine.name == "edge-tts"


def test_default_voice_constant():
    """Default voice is Japanese female — matches Stack-chan's personality."""
    assert DEFAULT_EDGE_TTS_VOICE == "ja-JP-NanamiNeural"


def test_default_rate_and_volume_constants():
    """Rate and volume defaults are neutral (no change)."""
    assert DEFAULT_EDGE_TTS_RATE == "+0%"
    assert DEFAULT_EDGE_TTS_VOLUME == "+0%"


def test_constructor_defaults():
    """Without arguments, engine uses env vars or defaults."""
    engine = EdgeTTSEngine()
    assert engine.voice == DEFAULT_EDGE_TTS_VOICE
    assert engine.rate == DEFAULT_EDGE_TTS_RATE
    assert engine.volume == DEFAULT_EDGE_TTS_VOLUME


def test_constructor_param_overrides():
    """Constructor arguments take precedence."""
    engine = EdgeTTSEngine(
        voice="en-US-AriaNeural",
        rate="+20%",
        volume="+50%",
    )
    assert engine.voice == "en-US-AriaNeural"
    assert engine.rate == "+20%"
    assert engine.volume == "+50%"


def test_constructor_env_overrides(monkeypatch):
    """Environment variables are used when constructor args are not given."""
    monkeypatch.setenv("STACKCHAN_EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    monkeypatch.setenv("STACKCHAN_EDGE_TTS_RATE", "+10%")
    monkeypatch.setenv("STACKCHAN_EDGE_TTS_VOLUME", "-20%")

    engine = EdgeTTSEngine()
    assert engine.voice == "zh-CN-XiaoxiaoNeural"
    assert engine.rate == "+10%"
    assert engine.volume == "-20%"


def test_constructor_param_wins_over_env(monkeypatch):
    """Constructor argument wins over environment variable."""
    monkeypatch.setenv("STACKCHAN_EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    engine = EdgeTTSEngine(voice="en-US-GuyNeural")
    assert engine.voice == "en-US-GuyNeural"


# ---------------------------------------------------------------------------
# synthesize() pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_calls_edge_tts_and_returns_pcm():
    """Happy path: text → edge-tts → MP3 → PCM decode → bytes returned."""
    engine = EdgeTTSEngine(voice="en-US-AriaNeural")
    fake_module = _make_fake_edge_tts_module()

    with (
        patch(
            "stackchan_mcp.tts.edge_tts_engine._import_edge_tts",
            return_value=fake_module,
        ),
        patch(
            "stackchan_mcp.tts.edge_tts_engine._mp3_to_pcm16_mono",
            return_value=FAKE_PCM,
        ) as mock_decode,
    ):
        pcm = await engine.synthesize("Hello world")

    assert isinstance(pcm, bytes)
    assert len(pcm) > 0
    # Verify decoder was called with the collected MP3 bytes
    mock_decode.assert_called_once_with(FAKE_MP3, 16000)


@pytest.mark.asyncio
async def test_synthesize_uses_speaker_id_as_voice_override():
    """speaker_id in opts overrides the engine's default voice."""
    engine = EdgeTTSEngine(voice="ja-JP-NanamiNeural")
    captured_voices: list[str] = []
    fake_module = _make_fake_edge_tts_module(capture_voices=captured_voices)

    with (
        patch(
            "stackchan_mcp.tts.edge_tts_engine._import_edge_tts",
            return_value=fake_module,
        ),
        patch(
            "stackchan_mcp.tts.edge_tts_engine._mp3_to_pcm16_mono",
            return_value=FAKE_PCM,
        ),
    ):
        await engine.synthesize("hello", speaker_id="en-US-GuyNeural")

    assert captured_voices == ["en-US-GuyNeural"]


@pytest.mark.asyncio
async def test_synthesize_passes_rate_and_volume():
    """Rate and volume modifiers are forwarded to edge-tts."""
    engine = EdgeTTSEngine(
        voice="en-US-AriaNeural",
        rate="+20%",
        volume="+50%",
    )
    captured_params: list[dict] = []
    fake_module = _make_fake_edge_tts_module(capture_params=captured_params)

    with (
        patch(
            "stackchan_mcp.tts.edge_tts_engine._import_edge_tts",
            return_value=fake_module,
        ),
        patch(
            "stackchan_mcp.tts.edge_tts_engine._mp3_to_pcm16_mono",
            return_value=FAKE_PCM,
        ),
    ):
        await engine.synthesize("hello")

    assert captured_params[0]["rate"] == "+20%"
    assert captured_params[0]["volume"] == "+50%"


@pytest.mark.asyncio
async def test_synthesize_per_call_rate_override():
    """Per-call rate/volume opts override engine defaults."""
    engine = EdgeTTSEngine(rate="+10%", volume="+10%")
    captured_params: list[dict] = []
    fake_module = _make_fake_edge_tts_module(capture_params=captured_params)

    with (
        patch(
            "stackchan_mcp.tts.edge_tts_engine._import_edge_tts",
            return_value=fake_module,
        ),
        patch(
            "stackchan_mcp.tts.edge_tts_engine._mp3_to_pcm16_mono",
            return_value=FAKE_PCM,
        ),
    ):
        await engine.synthesize("hello", rate="+30%", volume="-20%")

    assert captured_params[0]["rate"] == "+30%"
    assert captured_params[0]["volume"] == "-20%"


@pytest.mark.asyncio
async def test_synthesize_rejects_empty_text():
    """Empty/whitespace text fails fast before any network call."""
    engine = EdgeTTSEngine()

    with pytest.raises(ValueError, match="text"):
        await engine.synthesize("   ")


@pytest.mark.asyncio
async def test_synthesize_rejects_non_string_text():
    """Non-string text is a clean ValueError."""
    engine = EdgeTTSEngine()

    with pytest.raises(ValueError, match="text"):
        await engine.synthesize("")


@pytest.mark.asyncio
async def test_synthesize_raises_when_no_audio_returned():
    """Edge TTS returning no audio chunks surfaces as RuntimeError."""
    engine = EdgeTTSEngine()
    fake_module = _make_fake_edge_tts_module(empty=True)

    with patch(
        "stackchan_mcp.tts.edge_tts_engine._import_edge_tts",
        return_value=fake_module,
    ):
        with pytest.raises(RuntimeError, match="no audio data"):
            await engine.synthesize("hello")


@pytest.mark.asyncio
async def test_synthesize_defaults_voice_when_speaker_id_empty():
    """Empty or whitespace speaker_id falls back to engine default."""
    engine = EdgeTTSEngine(voice="ja-JP-NanamiNeural")
    captured_voices: list[str] = []
    fake_module = _make_fake_edge_tts_module(capture_voices=captured_voices)

    with (
        patch(
            "stackchan_mcp.tts.edge_tts_engine._import_edge_tts",
            return_value=fake_module,
        ),
        patch(
            "stackchan_mcp.tts.edge_tts_engine._mp3_to_pcm16_mono",
            return_value=FAKE_PCM,
        ),
    ):
        await engine.synthesize("hello", speaker_id="   ")

    assert captured_voices == ["ja-JP-NanamiNeural"]


# ---------------------------------------------------------------------------
# _mp3_to_pcm16_mono decoder
# ---------------------------------------------------------------------------


def test_mp3_to_pcm16_raises_without_av():
    """Missing av package gives a clear install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "av":
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="av.*not installed"):
            _mp3_to_pcm16_mono(b"fake", 16000)


def test_import_edge_tts_raises_without_package():
    """Missing edge-tts package gives a clear install hint."""
    from stackchan_mcp.tts.edge_tts_engine import _import_edge_tts
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "edge_tts":
            raise ImportError("No module named 'edge_tts'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="edge-tts is not installed"):
            _import_edge_tts()


# ---------------------------------------------------------------------------
# Integration with the orchestrator pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_tts_registers_in_framework():
    """EdgeTTSEngine can be registered and looked up in the engine registry."""
    from stackchan_mcp.tts import EngineRegistry

    reg = EngineRegistry()
    engine = EdgeTTSEngine()
    reg.register(engine)

    assert reg.get("edge-tts") is engine
    assert "edge-tts" in reg.names()


@pytest.mark.asyncio
async def test_full_pipeline_with_edge_tts(monkeypatch):
    """End-to-end: synthesize_and_send with EdgeTTSEngine produces frames.

    Uses fake opus encoding and fake edge-tts to test the full
    orchestrator → engine → encode → push pipeline.
    """
    from stackchan_mcp.tts import EngineRegistry, synthesize_and_send
    from stackchan_mcp.tts.audio_utils import (
        DEVICE_FRAME_DURATION_MS,
        DEVICE_SAMPLE_RATE,
    )

    # 60 ms of PCM → 1 Opus frame
    pcm_60ms = b"\x01\x00" * 960

    # Fake opus encoder
    def fake_encode(pcm: bytes, **kwargs):
        samples_per_frame = DEVICE_SAMPLE_RATE * DEVICE_FRAME_DURATION_MS // 1000
        bytes_per_frame = samples_per_frame * 2
        n_full = len(pcm) // bytes_per_frame
        n_partial = 1 if len(pcm) % bytes_per_frame else 0
        return iter(f"opus_frame_{i}".encode() for i in range(n_full + n_partial))

    import stackchan_mcp.tts.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "encode_opus_frames", fake_encode)

    # Fake ESP32 + Gateway
    class FakeESP32:
        device_connected = True

        def __init__(self):
            self.frames: list[bytes] = []
            self.tts_states: list[str] = []
            self.tts_lock = asyncio.Lock()

        async def send_audio_frame(self, frame: bytes) -> None:
            self.frames.append(frame)

        async def send_tts_state(self, state: str) -> None:
            self.tts_states.append(state)

    class FakeGateway:
        def __init__(self):
            self.esp32 = FakeESP32()

    gateway = FakeGateway()

    # Build engine with mocked synthesis
    engine = EdgeTTSEngine(voice="en-US-AriaNeural")

    # Patch the synthesize method to return known PCM
    async def fake_synthesize(text, **opts):
        return pcm_60ms

    monkeypatch.setattr(engine, "synthesize", fake_synthesize)

    reg = EngineRegistry()
    reg.register(engine)

    result = await synthesize_and_send(
        {"text": "hello", "voice": "edge-tts"},
        gateway=gateway,
        registry=reg,
    )

    assert result["engine"] == "edge-tts"
    assert result["frame_count"] == 1
    assert gateway.esp32.tts_states == ["start", "stop"]
    assert len(gateway.esp32.frames) == 1
