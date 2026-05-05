"""
transcribe_fast.py — Whisper large-v3 on NVIDIA CUDA
=====================================================
Transcribe audio using faster-whisper with the large-v3 model on GPU.

Features:
  • large-v3 (best accuracy ~95-97%)
  • CUDA float16 precision with auto-fallback (int8_float16 → int8 → CPU)
  • Word-level timestamps (every word has its own start/end)
  • VAD filter (Voice Activity Detection — skips silence automatically)
  • Hallucination detection & deduplication
  • Chunk-based processing with resume support (crash-safe)

Install:
    pip install faster-whisper
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # if needed

Usage:
    python transcribe_fast.py audio.wav
    python transcribe_fast.py audio.wav --language vi
    python transcribe_fast.py audio.wav --model large-v3 --compute-type int8_float16
    python transcribe_fast.py audio.wav --output segments.json --clean
    python transcribe_fast.py audio.wav --cpu
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Optional

import numpy as np
import soundfile as sf


DEFAULT_MODEL         = "large-v3"
DEFAULT_CHUNK_MINUTES = 10
SAMPLE_RATE           = 16000


# ══════════════════════════════════════════════════════════════════════════════
# Audio loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_audio(wav_path: str) -> np.ndarray:
    """Load WAV file into a float32 mono numpy array at 16 kHz."""
    print(f"[audio] Loading: {wav_path}")
    audio, sr = sf.read(wav_path, dtype="float32")

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != SAMPLE_RATE:
        print(f"[audio] Resampling {sr} Hz → {SAMPLE_RATE} Hz...")
        try:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="kaiser_best"
            )
        except ImportError:
            try:
                from scipy.signal import resample_poly
                g = math.gcd(SAMPLE_RATE, sr)
                audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
            except ImportError:
                new_len = int(math.ceil(len(audio) * SAMPLE_RATE / sr))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

    duration_min = len(audio) / SAMPLE_RATE / 60
    print(f"[audio] Duration: {duration_min:.1f} min")
    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Chunk path helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_dir(wav_path: str) -> str:
    return os.path.splitext(wav_path)[0] + "_chunks"


def _chunk_json_path(chunk_dir: str, idx: int) -> str:
    return os.path.join(chunk_dir, f"chunk_{idx:04d}.json")


# ══════════════════════════════════════════════════════════════════════════════
# Hallucination filter
# ══════════════════════════════════════════════════════════════════════════════

_HALL_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^[\s\W]+$"),
    re.compile(r"(?i)thank you for watching"),
    re.compile(r"(?i)like and subscribe"),
    re.compile(r"(?i)^\s*\.\s*$"),
    re.compile(r"(?i)^\s*,\s*$"),
    re.compile(r"(?i)subtitles? by"),
    re.compile(r"(?i)www\.\w+\.\w+"),
]


def _is_hallucination(text: str, duration: float) -> bool:
    """Return True if segment looks like a Whisper hallucination."""
    if not text.strip():
        return True
    for p in _HALL_PATTERNS:
        if p.search(text):
            return True
    # Speech rate > 10 words/sec = impossible → hallucination
    if duration > 0 and len(text.split()) / duration > 10:
        return True
    return False


def _deduplicate(segments: list) -> list:
    """Remove consecutive identical segments (Whisper repeat bug)."""
    if not segments:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        if seg["text"].strip().lower() != out[-1]["text"].strip().lower():
            out.append(seg)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Single-chunk transcription
# ══════════════════════════════════════════════════════════════════════════════

def _transcribe_chunk(
    model,
    audio_chunk: np.ndarray,
    time_offset: float,
    language: Optional[str],
    word_timestamps: bool = True,
) -> list:
    """
    Transcribe one audio chunk, return list of segment dicts:
      {start, end, text, words: [{word, start, end, probability}]}
    All timestamps are offset to match the full-audio timeline.
    """
    segments_gen, _info = model.transcribe(
        audio_chunk,
        language=language,

        # Accuracy
        beam_size=5,
        best_of=5,
        patience=1.0,
        condition_on_previous_text=True,

        # Word timestamps
        word_timestamps=word_timestamps,

        # Hallucination filters
        no_speech_threshold=0.4,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,

        # Temperature fallback schedule
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],

        # VAD — built-in silence detection
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=400,
            max_speech_duration_s=float("inf"),
        ),
    )

    segments = []
    for seg in segments_gen:
        text     = seg.text.strip()
        duration = seg.end - seg.start

        if _is_hallucination(text, duration):
            continue

        entry = {
            "start": round(seg.start + time_offset, 3),
            "end":   round(seg.end   + time_offset, 3),
            "text":  text,
        }

        if word_timestamps and seg.words:
            entry["words"] = [
                {
                    "word":        w.word,
                    "start":       round(w.start + time_offset, 3),
                    "end":         round(w.end   + time_offset, 3),
                    "probability": round(w.probability, 4),
                }
                for w in seg.words
            ]

        segments.append(entry)

    return _deduplicate(segments)


# ══════════════════════════════════════════════════════════════════════════════
# Model loader with VRAM fallback chain
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(WhisperModel, model_name: str, device: str, compute_type: str):
    """Try loading WhisperModel; fall back through precision tiers if VRAM fails."""
    fallback_chain = [
        (device,  compute_type),
        ("cuda",  "int8_float16"),
        ("cuda",  "int8"),
        ("cpu",   "int8"),
    ]
    seen, chain = set(), []
    for d, ct in fallback_chain:
        if (d, ct) not in seen:
            seen.add((d, ct))
            chain.append((d, ct))

    for dev, ct in chain:
        try:
            m = WhisperModel(model_name, device=dev, compute_type=ct)
            if (dev, ct) != (device, compute_type):
                print(f"  ⚠️  Fallback: device={dev}  compute_type={ct}")
            return m
        except Exception as e:
            print(f"  ⚠️  {dev.upper()} {ct} failed: {str(e)[:100]}")

    raise RuntimeError("Could not load model on any device/precision tier.")


# ══════════════════════════════════════════════════════════════════════════════
# Main: chunked transcription with resume
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_chunked(
    wav_path: str,
    model_name: str            = DEFAULT_MODEL,
    language: Optional[str]    = None,
    chunk_minutes: int         = DEFAULT_CHUNK_MINUTES,
    output_path: Optional[str] = None,
    clean_chunks: bool         = False,
    compute_type: str          = "float16",
    device: str                = "cuda",
    word_timestamps: bool      = True,
) -> list:
    """
    Transcribe a WAV file in resumable chunks using faster-whisper.

    Args:
        wav_path:        Path to the 16 kHz mono WAV file.
        model_name:      Whisper model (default: large-v3).
        language:        ISO code e.g. "en", "vi". None = auto-detect.
        chunk_minutes:   Minutes per chunk (default: 10).
        output_path:     Final segments JSON path.
        clean_chunks:    Delete temp chunk JSONs after merging.
        compute_type:    "float16" = max accuracy (~10 GB VRAM)
                         "int8_float16" = fast, tiny loss (~6 GB VRAM)
                         "int8" = lowest VRAM (~4 GB)
        device:          "cuda" for NVIDIA GPU, "cpu" as fallback.
        word_timestamps: Save per-word start/end times in output JSON.

    Returns:
        List of segment dicts.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper is not installed.", file=sys.stderr)
        print("   pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    # Chunk temp directory
    chunk_dir = _chunk_dir(wav_path)
    os.makedirs(chunk_dir, exist_ok=True)

    # Load audio
    audio         = _load_audio(wav_path)
    total_samples = len(audio)
    chunk_samples = chunk_minutes * 60 * SAMPLE_RATE
    total_chunks  = math.ceil(total_samples / chunk_samples)
    total_dur_min = total_samples / SAMPLE_RATE / 60

    print(f"\n{'═'*62}")
    print(f"  Whisper {model_name}  |  GPU-accelerated transcription")
    print(f"{'─'*62}")
    print(f"  Device        : {device.upper()}")
    print(f"  Precision     : {compute_type}")
    print(f"  Language      : {language or 'auto-detect'}")
    print(f"  Audio         : {total_dur_min:.1f} min  ({total_chunks} chunks × {chunk_minutes} min)")
    print(f"  Word timestamps: {'yes' if word_timestamps else 'no'}")
    print(f"  Resume        : yes (already-done chunks skipped)")
    print(f"{'═'*62}")

    # Load model
    print(f"\n[model] Loading '{model_name}' on {device.upper()} ({compute_type})...")
    t0    = time.time()
    model = _load_model(WhisperModel, model_name, device, compute_type)
    print(f"[model] ✅ Ready ({time.time() - t0:.1f}s)\n")

    # Process chunks
    total_start = time.time()

    for i in range(total_chunks):
        chunk_json = _chunk_json_path(chunk_dir, i)

        if os.path.isfile(chunk_json):
            print(f"[chunk {i+1:3d}/{total_chunks}] ✅ Already done — skipping (resume)")
            continue

        start_sample = i * chunk_samples
        end_sample   = min(start_sample + chunk_samples, total_samples)
        time_offset  = start_sample / SAMPLE_RATE
        chunk_audio  = audio[start_sample:end_sample]
        chunk_dur    = len(chunk_audio) / SAMPLE_RATE / 60

        print(
            f"[chunk {i+1:3d}/{total_chunks}]  "
            f"{_mmss(time_offset)} → {_mmss(end_sample / SAMPLE_RATE)}  "
            f"({chunk_dur:.1f} min)  transcribing...",
            end="", flush=True,
        )

        t_chunk  = time.time()
        segments = _transcribe_chunk(model, chunk_audio, time_offset, language, word_timestamps)
        elapsed  = time.time() - t_chunk

        with open(chunk_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        chunks_done = i + 1
        avg_time    = (time.time() - total_start) / chunks_done
        eta         = avg_time * (total_chunks - chunks_done)
        print(f"  ✅  {elapsed:.0f}s  |  {len(segments)} segs  |  ETA: {_dur(eta)}")

    # Merge all chunks
    print(f"\n[merge] Merging {total_chunks} chunks...")
    all_segments = []
    global_id    = 1

    for i in range(total_chunks):
        chunk_json = _chunk_json_path(chunk_dir, i)
        if not os.path.isfile(chunk_json):
            print(f"  ⚠️  Chunk {i} missing — skipped")
            continue
        with open(chunk_json, "r", encoding="utf-8") as f:
            segs = json.load(f)
        for seg in segs:
            seg["id"] = global_id
            global_id += 1
        all_segments.extend(segs)

    total_elapsed = time.time() - total_start

    # Save final JSON
    if output_path is None:
        base        = os.path.splitext(wav_path)[0]
        output_path = base + "_segments.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'═'*62}")
    print(f"  ✅  Transcription complete!")
    print(f"  Segments       : {len(all_segments)}")
    print(f"  Total time     : {_dur(total_elapsed)}")
    print(f"  Output         : {output_path}")
    print(f"  Word timestamps: {'yes' if word_timestamps else 'no'}")
    print(f"{'═'*62}")

    print("\n── Preview (first 5 segments) " + "─"*32)
    for seg in all_segments[:5]:
        print(f"  [{_mmss(seg['start'])} → {_mmss(seg['end'])}]  {seg['text']}")
    print("─" * 62 + "\n")

    # Clean temp chunks
    if clean_chunks:
        import shutil
        shutil.rmtree(chunk_dir)
        print("[clean] Temp chunks deleted.")
    else:
        print(f"[clean] Temp chunks kept at: {chunk_dir}")
        print("[clean] Use --clean to remove them on next run.\n")

    return all_segments


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.1f} hr"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe WAV with faster-whisper large-v3 on NVIDIA CUDA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_fast.py audio.wav
  python transcribe_fast.py audio.wav --language vi
  python transcribe_fast.py audio.wav --compute-type int8_float16
  python transcribe_fast.py audio.wav --output segments.json --clean
  python transcribe_fast.py audio.wav --cpu
        """,
    )
    parser.add_argument("wav",
        help="Path to the 16 kHz mono WAV file.")
    parser.add_argument("--model", "-m",
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Whisper model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--language", "-l",
        default=None,
        help="Language ISO code e.g. 'en', 'vi', 'ja'. Omit = auto-detect.")
    parser.add_argument("--chunk-minutes", "-c",
        type=int, default=DEFAULT_CHUNK_MINUTES,
        help=f"Minutes per chunk (default: {DEFAULT_CHUNK_MINUTES}).")
    parser.add_argument("--output", "-o",
        default=None,
        help="Output JSON path (default: <wav>_segments.json).")
    parser.add_argument("--clean",
        action="store_true",
        help="Delete temp chunk files after merging.")
    parser.add_argument("--compute-type",
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        help="GPU precision (default: float16 = max accuracy, ~10 GB VRAM).")
    parser.add_argument("--cpu",
        action="store_true",
        help="Force CPU mode (no GPU required, slower).")
    parser.add_argument("--no-word-timestamps",
        action="store_true",
        help="Disable word-level timestamps (smaller output JSON).")

    args   = parser.parse_args()
    device = "cpu" if args.cpu else "cuda"
    ct     = "int8" if args.cpu else args.compute_type

    try:
        transcribe_chunked(
            wav_path        = args.wav,
            model_name      = args.model,
            language        = args.language,
            chunk_minutes   = args.chunk_minutes,
            output_path     = args.output,
            clean_chunks    = args.clean,
            compute_type    = ct,
            device          = device,
            word_timestamps = not args.no_word_timestamps,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()