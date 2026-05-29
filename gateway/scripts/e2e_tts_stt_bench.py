"""End-to-end TTS → STT pipeline benchmark through real MCP production code.

Simulates the full Stack-chan conversation loop WITHOUT a physical ESP32:

  1. TTS (edge-tts)  → synthesize text to MP3 audio
  2. Audio decode     → convert MP3 to 16 kHz mono s16le PCM
                        (same format the gateway produces after Opus decode)
  3. STT (faster-whisper FasterWhisperEngine.transcribe())
                      → feed PCM into the REAL production engine code path

Measures latency at every stage and reports a summary.
"""

import asyncio
import os
import sys
import time
from io import BytesIO

# Ensure the package can be imported if run from the gateway directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mirror the production code's HF_ENDPOINT fallback
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ── Test cases ───────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "label": "English short",
        "text": "Hello! I am Stack-chan, your friendly robot assistant.",
        "voice": "en-US-AriaNeural",
        "language": "en",
    },
    {
        "label": "Japanese short",
        "text": "こんにちは！僕はスタックちゃんだよ。よろしくね！",
        "voice": "ja-JP-NanamiNeural",
        "language": "ja",
    },
    {
        "label": "English long",
        "text": (
            "Today we are testing the speech to text pipeline. "
            "The weather is sunny, and I hope you are having a wonderful day. "
            "Let me know if you need any help with anything at all."
        ),
        "voice": "en-US-AriaNeural",
        "language": "en",
    },
    {
        "label": "Chinese short",
        "text": "你好！我是Stack-chan，一个可爱的桌面机器人。很高兴认识你！",
        "voice": "zh-CN-XiaoxiaoNeural",
        "language": "zh",
    },
]


# ── Stage 1: TTS synthesis via edge-tts ──────────────────────────────────
async def tts_synthesize(text: str, voice: str) -> tuple[bytes, float]:
    """Synthesize text to MP3 bytes via edge-tts. Returns (mp3_bytes, elapsed)."""
    import edge_tts

    start = time.perf_counter()
    communicate = edge_tts.Communicate(text, voice)

    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])

    elapsed = time.perf_counter() - start
    return b"".join(chunks), elapsed


# ── Stage 2: MP3 → 16 kHz mono s16le PCM ────────────────────────────────
def mp3_to_pcm_16k(mp3_bytes: bytes) -> tuple[bytes, float, float]:
    """Convert MP3 bytes to 16 kHz mono s16le PCM using av (bundled with faster-whisper).

    Returns (pcm_bytes, audio_duration_seconds, conversion_elapsed).
    """
    import av

    start = time.perf_counter()

    container = av.open(BytesIO(mp3_bytes))
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=16000,
    )

    pcm_chunks: list[bytes] = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for rf in resampled:
            pcm_chunks.append(rf.to_ndarray().tobytes())

    container.close()
    pcm = b"".join(pcm_chunks)
    elapsed = time.perf_counter() - start

    # Calculate audio duration from PCM length (16 kHz, 16-bit mono = 2 bytes/sample)
    audio_duration = len(pcm) / (16000 * 2)
    return pcm, audio_duration, elapsed


# ── Stage 3: STT via the REAL production FasterWhisperEngine ─────────────
async def stt_transcribe(pcm: bytes, language: str, model: str = "tiny") -> tuple[dict, float]:
    """Feed PCM into the production FasterWhisperEngine.transcribe().

    Returns (result_dict, elapsed).
    """
    # Import the REAL production engine
    from stackchan_mcp.stt.faster_whisper import FasterWhisperEngine

    engine = FasterWhisperEngine(model=model)

    start = time.perf_counter()
    result = await engine.transcribe(pcm, language=language)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ── Main benchmark ───────────────────────────────────────────────────────
async def main():
    print("=" * 70)
    print("   Stack-chan MCP  ·  End-to-End TTS → STT Pipeline Benchmark")
    print("=" * 70)
    print()

    import importlib.util
    if importlib.util.find_spec("edge_tts") is None or importlib.util.find_spec("av") is None:
        print("Error: Required dependencies (edge-tts, av) are missing.")
        print("Please run: pip install stackchan-mcp[tts-edge,stt] first.")
        return

    results = []

    for case in TEST_CASES:
        label = case["label"]
        text = case["text"]
        voice = case["voice"]
        lang = case["language"]

        print(f"── [{label}] ──")
        print(f"   Original text : {text}")
        print(f"   Voice / Lang  : {voice} / {lang}")
        print()

        # Stage 1: TTS
        print("   [1/3] TTS (edge-tts) synthesizing...", end="", flush=True)
        try:
            mp3_bytes, tts_time = await tts_synthesize(text, voice)
            print(f" done. ({tts_time:.3f}s, {len(mp3_bytes):,} bytes MP3)")
        except Exception as e:
            print(f" failed: {e}")
            continue

        # Stage 2: Decode MP3 → PCM
        print("   [2/3] Audio decode (MP3 → 16kHz PCM)...", end="", flush=True)
        try:
            pcm, audio_dur, decode_time = mp3_to_pcm_16k(mp3_bytes)
            print(f" done. ({decode_time:.3f}s, {len(pcm):,} bytes PCM, {audio_dur:.2f}s audio)")
        except Exception as e:
            print(f" failed: {e}")
            continue

        # Stage 3: STT (production code path)
        print("   [3/3] STT (FasterWhisperEngine.transcribe)...", end="", flush=True)
        try:
            result, stt_time = await stt_transcribe(pcm, language=lang)
            recognized = result.get("text", "")
            det_lang = result.get("language", "")
            print(f" done. ({stt_time:.3f}s)")
        except Exception as e:
            print(f" failed: {e}")
            continue

        total = tts_time + decode_time + stt_time
        print()
        print(f"   Recognized    : {recognized}")
        print(f"   Detected lang : {det_lang}")
        print("   ────────────────────────────────────")
        print(f"   TTS time      : {tts_time:.3f}s")
        print(f"   Decode time   : {decode_time:.3f}s")
        print(f"   STT time      : {stt_time:.3f}s  (Real-time factor: {stt_time/audio_dur:.2f}x)")
        print(f"   Total pipeline: {total:.3f}s")
        print()

        results.append({
            "label": label,
            "original": text,
            "recognized": recognized,
            "audio_dur": audio_dur,
            "tts_time": tts_time,
            "decode_time": decode_time,
            "stt_time": stt_time,
            "total": total,
            "rtf": stt_time / audio_dur if audio_dur else 0,
        })

    if not results:
        print("No test cases completed successfully.")
        return

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("                        BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Test Case':<18} {'Audio':>6} {'TTS':>7} {'Decode':>7} {'STT':>7} {'Total':>7} {'RTF':>5}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['label']:<18} "
            f"{r['audio_dur']:>5.2f}s "
            f"{r['tts_time']:>6.3f}s "
            f"{r['decode_time']:>6.3f}s "
            f"{r['stt_time']:>6.3f}s "
            f"{r['total']:>6.3f}s "
            f"{r['rtf']:>4.2f}x"
        )
    print("-" * 70)
    print()
    print("RTF = Real-Time Factor for STT (< 1.0 means faster than real-time)")
    print("All timings are wall-clock, measured with time.perf_counter().")
    print()

    # Print text comparison
    print("── Text Accuracy ──")
    for r in results:
        print(f"  [{r['label']}]")
        print(f"    Original  : {r['original']}")
        print(f"    Recognized: {r['recognized']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
