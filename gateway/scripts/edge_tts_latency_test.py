import asyncio
import time
import re
import os

# Ensure the package can be imported if run from the gateway directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define a paragraph with a mix of long and very short sentences
TEST_TEXT = (
    "Hello! Yes. Let us begin. This is a real-world latency test for the edge-tts speech synthesis engine. "
    "We are comparing the default all-at-once method against our optimized pipeline. "
    "By splitting, merging short sentences, and starting parallel synthesis for S1 and S2, "
    "the robot starts speaking almost instantly, and subsequent sentences are prefetched. "
    "It is amazing! Yes. Really."
)

VOICE = "en-US-AriaNeural"

def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r'([。！？\n\?\!]|\.(?=\s|$))', text)
    sentences = []
    current_sentence = ""
    for part in parts:
        if part is None:
            continue
        current_sentence += part
        if part in ("。", "！", "？", "\n", "?", "!", "."):
            stripped = current_sentence.strip()
            if stripped:
                sentences.append(stripped)
            current_sentence = ""
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    if not sentences and text.strip():
        sentences.append(text.strip())
    return sentences

def split_and_merge_sentences(text: str, min_length: int = 15) -> list[str]:
    raw_sentences = split_into_sentences(text)
    merged_sentences = []
    current = ""
    for s in raw_sentences:
        if current:
            current += " " + s
        else:
            current = s
        if len(current) >= min_length:
            merged_sentences.append(current)
            current = ""
    if current:
        merged_sentences.append(current)
    return merged_sentences

async def run_synthesis(text: str, voice: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

async def run_scenario_a():
    print("\n--- [Scenario A: All-at-once Synthesis] ---")
    start = time.time()
    data = await run_synthesis(TEST_TEXT, VOICE)
    elapsed = time.time() - start
    print(f"Scenario A complete in {elapsed:.3f}s. Received {len(data)} bytes.")
    return elapsed

async def run_scenario_b():
    print("\n--- [Scenario B: Basic Sentence-by-Sentence (No Merging, No Parallel Pre-trigger)] ---")
    sentences = split_into_sentences(TEST_TEXT)
    print(f"Text split into {len(sentences)} sentences.")
    
    start = time.time()
    timeline = []
    
    # Process S1
    await run_synthesis(sentences[0], VOICE)
    ttfs = time.time() - start
    print(f"  S1 complete. TTFS: {ttfs:.3f}s")
    timeline.append(ttfs)
    
    # Process rest sequentially
    for i in range(1, len(sentences)):
        s_start = time.time()
        await run_synthesis(sentences[i], VOICE)
        s_elapsed = time.time() - s_start
        cumulative = time.time() - start
        print(f"  S{i+1} complete. Synthesized in {s_elapsed:.3f}s (Timeline: {cumulative:.3f}s)")
        timeline.append(cumulative)
        
    total_time = time.time() - start
    print(f"Scenario B complete. Total elapsed time: {total_time:.3f}s.")
    return ttfs, total_time, len(sentences)

async def run_scenario_c():
    print("\n--- [Scenario C: Optimized Sentence-by-Sentence (With Merging & Parallel S1+S2 Pre-trigger)] ---")
    sentences = split_and_merge_sentences(TEST_TEXT, min_length=15)
    print(f"Text split & merged into {len(sentences)} sentences:")
    for idx, s in enumerate(sentences):
        print(f"  {idx+1}: {s}")
        
    start = time.time()
    timeline = []
    
    # Parallel trigger S1 & S2 at time = 0
    s1_task = asyncio.create_task(run_synthesis(sentences[0], VOICE))
    s2_task = None
    if len(sentences) > 1:
        s2_task = asyncio.create_task(run_synthesis(sentences[1], VOICE))
        
    # Await S1 (TTFS)
    await s1_task
    ttfs = time.time() - start
    print(f"  S1 (TTFS) ready at {ttfs:.3f}s")
    timeline.append(ttfs)
    
    current_task = s2_task
    # We iterate and prefetch the next sentences
    for i in range(1, len(sentences)):
        # While S_i is being "played", we pre-trigger S_i+1 in parallel
        next_task = None
        if i + 1 < len(sentences):
            next_task = asyncio.create_task(run_synthesis(sentences[i + 1], VOICE))
            
        # Await the current sentence synthesis to complete
        s_start = time.time()
        if current_task:
            await current_task
        s_elapsed = time.time() - s_start
        cumulative = time.time() - start
        print(f"  S{i+1} ready at {cumulative:.3f}s (Wait time during playback loop: {s_elapsed:.3f}s)")
        timeline.append(cumulative)
        
        current_task = next_task
        
    total_time = time.time() - start
    print(f"Scenario C complete. Total elapsed time: {total_time:.3f}s.")
    return ttfs, total_time, len(sentences)

async def main():
    print("==================================================")
    print("      EDGE TTS LATENCY & BENCHMARK TEST")
    print("==================================================")
    print(f"Test Input: {TEST_TEXT}\n")
    
    import importlib.util
    if importlib.util.find_spec("edge_tts") is None:
        print("Error: edge-tts package is not installed. Run 'pip install stackchan-mcp[tts-edge]' first.")
        return
        
    time_a = await run_scenario_a()
    ttfs_b, total_b, count_b = await run_scenario_b()
    ttfs_c, total_c, count_c = await run_scenario_c()
    
    print("\n==================================================")
    print("                 BENCHMARK SUMMARY")
    print("==================================================")
    print("Scenario A (All-at-once):")
    print(f"  - Startup Latency (TTFS): {time_a:.3f}s")
    print(f"  - Total Synthesis Time:  {time_a:.3f}s")
    
    print(f"\nScenario B (Basic Sentence Split - {count_b} sentences):")
    print(f"  - Startup Latency (TTFS): {ttfs_b:.3f}s ({(time_a - ttfs_b)*1000:.1f}ms / {(1 - ttfs_b/time_a)*100:.1f}% faster than A)")
    print(f"  - Total Sequential Time: {total_b:.3f}s")
    
    print(f"\nScenario C (Optimized Merge & Parallel Prefetch - {count_c} sentences):")
    print(f"  - Startup Latency (TTFS): {ttfs_c:.3f}s ({(time_a - ttfs_c)*1000:.1f}ms / {(1 - ttfs_c/time_a)*100:.1f}% faster than A)")
    print(f"  - Total Pipelined Time:  {total_c:.3f}s")
    
    # Calculate performance improvements
    speedup_ttfs = (time_a - ttfs_c) / time_a * 100
    overlap_efficiency = (total_b - total_c) / total_b * 100
    print("--------------------------------------------------")
    print(f"TTFS Improvement (A vs C): {speedup_ttfs:.1f}% FASTER to start speaking!")
    print(f"Pipeline Synthesis Efficiency (B vs C): {overlap_efficiency:.1f}% time saved due to parallel overlap!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
