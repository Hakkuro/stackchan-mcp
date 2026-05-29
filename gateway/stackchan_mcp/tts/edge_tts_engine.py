"""Edge TTS engine — free cloud TTS via Microsoft Edge's speech service.

``edge-tts`` is an MIT-licensed Python library that uses Microsoft Edge's
online text-to-speech service.  It requires no API key and supports a
wide range of voices across many languages (English, Japanese, Chinese,
etc.).  The service returns MP3 audio, which we decode to 16 kHz mono
PCM using the ``av`` (PyAV) library — the same dependency already
pulled in transitively by ``faster-whisper``.

Configuration:

    ``STACKCHAN_EDGE_TTS_VOICE``
        Default voice identifier.  Default ``ja-JP-NanamiNeural``
        (Japanese female).  Override per-call via the say() tool's
        ``speaker_id`` argument.

    ``STACKCHAN_EDGE_TTS_RATE``
        Speech rate modifier string (e.g. ``"+10%"``, ``"-20%"``).
        Default ``"+0%"`` (no change).

    ``STACKCHAN_EDGE_TTS_VOLUME``
        Volume modifier string (e.g. ``"+50%"``, ``"-30%"``).
        Default ``"+0%"`` (no change).
"""

from __future__ import annotations

import asyncio
import logging
import os
from io import BytesIO
from typing import Any

from .audio_utils import DEVICE_SAMPLE_RATE
from .base import TTSEngine

logger = logging.getLogger(__name__)


#: Default voice.  ``ja-JP-NanamiNeural`` is a clear, expressive Japanese
#: female voice that works well for Stack-chan's personality.
DEFAULT_EDGE_TTS_VOICE = "ja-JP-NanamiNeural"

#: Default speech rate modifier.
DEFAULT_EDGE_TTS_RATE = "+0%"

#: Default volume modifier.
DEFAULT_EDGE_TTS_VOLUME = "+0%"


def _mp3_to_pcm16_mono(mp3_bytes: bytes, target_rate: int) -> bytes:
    """Decode MP3 bytes to signed-16-bit LE mono PCM at *target_rate*.

    Uses PyAV (``av``) for decoding and resampling in a single pass.
    PyAV is pulled in transitively by ``faster-whisper`` which most
    users of this gateway already have installed, so it adds no new
    dependency in the common case.
    """
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyAV (av) is not installed.  Install with "
            "'pip install av' or 'pip install stackchan-mcp[tts-edge]' "
            "to enable Edge TTS MP3 decoding."
        ) from exc

    container = av.open(BytesIO(mp3_bytes))
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_rate,
    )

    pcm_chunks: list[bytes] = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for rf in resampled:
            pcm_chunks.append(rf.to_ndarray().tobytes())

    container.close()
    return b"".join(pcm_chunks)


class EdgeTTSEngine(TTSEngine):
    """Synthesise text via Microsoft Edge's online speech service.

    Setup::

        pip install stackchan-mcp[tts-edge]

    No API key is needed.  The service is free and supports 300+
    voices across 70+ languages.

    Configuration:

        ``STACKCHAN_EDGE_TTS_VOICE``
            Voice identifier (e.g. ``"en-US-AriaNeural"``).
            Default ``"ja-JP-NanamiNeural"``.

        ``STACKCHAN_EDGE_TTS_RATE``
            Speed modifier (e.g. ``"+20%"``).  Default ``"+0%"``.

        ``STACKCHAN_EDGE_TTS_VOLUME``
            Volume modifier (e.g. ``"+50%"``).  Default ``"+0%"``.
    """

    name = "edge-tts"

    def __init__(
        self,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
    ) -> None:
        env_voice = os.getenv("STACKCHAN_EDGE_TTS_VOICE")
        env_rate = os.getenv("STACKCHAN_EDGE_TTS_RATE")
        env_volume = os.getenv("STACKCHAN_EDGE_TTS_VOLUME")

        self._voice = voice or env_voice or DEFAULT_EDGE_TTS_VOICE
        self._rate = rate or env_rate or DEFAULT_EDGE_TTS_RATE
        self._volume = volume or env_volume or DEFAULT_EDGE_TTS_VOLUME

    @property
    def voice(self) -> str:
        """Current default voice identifier."""
        return self._voice

    @property
    def rate(self) -> str:
        """Current rate modifier."""
        return self._rate

    @property
    def volume(self) -> str:
        """Current volume modifier."""
        return self._volume

    async def synthesize(self, text: str, **opts: Any) -> bytes:
        """Stream text through edge-tts and return 16 kHz mono PCM.

        Recognised opts:

            ``speaker_id``: str
                Voice identifier override (e.g. ``"en-US-GuyNeural"``).
                Falls back to :attr:`voice`.

            ``rate``: str
                Per-call speed modifier override.

            ``volume``: str
                Per-call volume modifier override.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Edge TTS synthesize: 'text' must be a non-empty string"
            )

        edge_tts = _import_edge_tts()

        # Resolve voice — speaker_id acts as a voice override
        voice_override = opts.get("speaker_id")
        if isinstance(voice_override, str) and voice_override.strip():
            voice = voice_override.strip()
        else:
            voice = self._voice

        rate = opts.get("rate", self._rate) or self._rate
        volume = opts.get("volume", self._volume) or self._volume

        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate,
            volume=volume,
        )

        # Collect all audio chunks from the streaming response
        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

        mp3_bytes = b"".join(mp3_chunks)
        if not mp3_bytes:
            raise RuntimeError(
                f"Edge TTS returned no audio data for voice={voice!r}, "
                f"text={text[:60]!r}"
            )

        # Decode MP3 → 16 kHz mono s16le PCM in a thread to avoid
        # blocking the event loop on CPU-intensive FFmpeg work.
        pcm = await asyncio.to_thread(
            _mp3_to_pcm16_mono, mp3_bytes, DEVICE_SAMPLE_RATE
        )

        logger.info(
            "Edge TTS synthesised %d bytes PCM (16 kHz mono) for "
            "voice=%s, text=%r",
            len(pcm),
            voice,
            text[:60],
        )
        return pcm


def _import_edge_tts():
    """Import and return the ``edge_tts`` module, or raise RuntimeError."""
    try:
        import edge_tts  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts is not installed.  Install with "
            "'pip install stackchan-mcp[tts-edge]' to enable "
            "Edge TTS support."
        ) from exc
    return edge_tts

