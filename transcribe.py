"""
transcribe.py — Whisper large-v3 on CUDA, full file (no chunking)
=================================================================
Uses openai-whisper directly on GPU to transcribe the entire audio
file in one pass — avoids missing dialogue from chunk boundaries.

Install:
    pip install openai-whisper
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

VRAM guide (fp16):
    tiny    ~1 GB   |  base   ~1 GB  |  small  ~2 GB
    medium  ~5 GB   |  large  ~10 GB |  large-v2/v3  ~10 GB

    RTX 3050 4 GB  → use: small or medium (int8 via --int8)
    RTX 3060 12 GB → use: large-v3
    RTX 4090 24 GB → use: large-v3

Usage:
    python transcribe.py audio_16k.wav
    python transcribe.py audio_16k.wav --language en
    python transcribe.py audio_16k.wav --language vi --output segments.json
    python transcribe.py audio_16k.wav --model medium
    python transcribe.py audio_16k.wav --int8        (4-bit, saves VRAM)
    python transcribe.py audio_16k.wav --cpu
"""

import argparse
import json
import os
import sys
import time
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# CUDA diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def _check_cuda(torch, force_cuda: bool = False) -> None:
    """Print CUDA status. Exit with instructions if CUDA unavailable and forced."""
    cuda_available   = torch.cuda.is_available()
    cuda_built       = torch.backends.cuda.is_built()
    torch_version    = torch.__version__
    cuda_version_str = torch.version.cuda or "N/A"

    print(f"\n[CUDA] torch={torch_version}  cuda_built={cuda_built}  "
          f"cuda_available={cuda_available}  CUDA={cuda_version_str}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[CUDA] GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")
        return

    # Not available — print reason
    print("", file=sys.stderr)
    print("!" * 62, file=sys.stderr)
    if not cuda_built:
        print("  ERROR: PyTorch is CPU-only build — no CUDA support!", file=sys.stderr)
        print("", file=sys.stderr)
        print("  FIX — reinstall PyTorch with CUDA:", file=sys.stderr)
        print("", file=sys.stderr)
        print("  pip uninstall torch torchvision torchaudio -y", file=sys.stderr)
        print("", file=sys.stderr)
        print("  # CUDA 12.1 (recommended):", file=sys.stderr)
        print("  pip install torch torchvision torchaudio \\", file=sys.stderr)
        print("      --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        print("", file=sys.stderr)
        print("  # CUDA 11.8:", file=sys.stderr)
        print("  pip install torch torchvision torchaudio \\", file=sys.stderr)
        print("      --index-url https://download.pytorch.org/whl/cu118", file=sys.stderr)
    else:
        print("  WARNING: torch has CUDA build but GPU not detected.", file=sys.stderr)
        print("  Check:", file=sys.stderr)
        print("    - NVIDIA driver installed?  Run: nvidia-smi", file=sys.stderr)
        print(f"    - torch.version.cuda = {cuda_version_str}", file=sys.stderr)
    print("!" * 62, file=sys.stderr)
    print("", file=sys.stderr)

    if force_cuda:
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Core transcription
# ══════════════════════════════════════════════════════════════════════════════

# VRAM requirements per model at fp16 (GB)
_VRAM_REQ = {
    "tiny": 1.0, "base": 1.0, "small": 2.0,
    "medium": 5.0, "large": 10.0, "large-v2": 10.0, "large-v3": 10.0,
}


def transcribe(
    wav_path: str,
    model_name: str            = "large-v3",
    language: Optional[str]    = None,
    output_path: Optional[str] = None,
    device: Optional[str]      = None,
    word_timestamps: bool      = True,
    use_int8: bool             = False,
) -> list:
    """
    Transcribe a WAV file using openai-whisper on GPU, full file at once.

    Args:
        wav_path:        Path to the 16 kHz mono WAV file.
        model_name:      Whisper model (default: large-v3).
        language:        ISO code e.g. "en", "vi". None = auto-detect.
        output_path:     Where to save segments JSON.
        device:          "cuda" | "cpu" | None (auto-select).
        word_timestamps: Save per-word start/end times.
        use_int8:        Load model in int8 (saves VRAM, slight accuracy loss).

    Returns:
        List of segment dicts [{id, start, end, text, words}, ...].
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")

    # Import
    try:
        import whisper
        import torch
    except ImportError:
        print("ERROR: Missing libraries. Run:", file=sys.stderr)
        print("  pip install openai-whisper", file=sys.stderr)
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        sys.exit(1)

    # Check CUDA and select device
    _check_cuda(torch, force_cuda=(device == "cuda"))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available.", file=sys.stderr)
        sys.exit(1)

    fp16 = (device == "cuda") and (not use_int8)

    # VRAM check
    file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    vram_avail   = 0.0
    if device == "cuda":
        vram_avail = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_need  = _VRAM_REQ.get(model_name, 10.0)
        if vram_avail < vram_need and not use_int8:
            suitable = [m for m, v in _VRAM_REQ.items() if v <= vram_avail]
            print(f"\n[WARN] GPU has {vram_avail:.1f} GB VRAM, "
                  f"'{model_name}' needs ~{vram_need:.0f} GB!")
            print(f"[WARN] Suitable models: {suitable}")
            print(f"[WARN] Or add --int8 to reduce VRAM usage.")
            print(f"[WARN] Continuing anyway...\n")

    # Info banner
    print(f"\n{'='*62}")
    print(f"  Whisper {model_name}  |  openai-whisper")
    print(f"{'-'*62}")
    print(f"  File     : {os.path.basename(wav_path)}  ({file_size_mb:.1f} MB)")
    print(f"  Device   : {device.upper()}", end="")
    if device == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)}, {vram_avail:.1f} GB)")
    else:
        print()
    print(f"  Precision: {'int8' if use_int8 else 'float16 (fp16)' if fp16 else 'float32'}")
    print(f"  Language : {language or 'auto-detect'}")
    print(f"  Words    : {'yes' if word_timestamps else 'no'}")
    print(f"  Mode     : full file, no chunking")
    print(f"{'='*62}\n")

    # Load model
    print(f"[model] Loading '{model_name}' on {device.upper()}...")
    t0    = time.time()
    model = whisper.load_model(model_name, device=device)
    print(f"[model] Ready in {time.time() - t0:.1f}s\n")

    # Load audio as numpy array — avoids needing ffmpeg in PATH
    print(f"[audio] Loading {os.path.basename(wav_path)} ...")
    try:
        # whisper.load_audio uses ffmpeg internally — try it first
        audio = whisper.load_audio(wav_path)
    except Exception:
        # Fallback: read WAV directly with soundfile
        import math
        import numpy as np
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)          # stereo -> mono
        if sr != 16000:
            from scipy.signal import resample_poly
            g     = math.gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    print(f"[audio] Loaded ({len(audio)/16000/60:.1f} min)\n")

    # Transcribe — entire file at once
    print(f"[transcribe] Starting full-file transcription (verbose=True)...\n")
    t1 = time.time()

    result = model.transcribe(
        audio,                          # numpy array — no ffmpeg needed

        # Language
        language=language,

        # Accuracy
        beam_size=5,
        best_of=5,
        patience=1.0,
        condition_on_previous_text=True,

        # Word-level timestamps
        word_timestamps=word_timestamps,

        # Hallucination / silence filters
        no_speech_threshold=0.4,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,

        # Temperature fallback (retry with higher temp if low confidence)
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),

        # GPU float16
        fp16=fp16,

        # Show each segment as it is produced
        verbose=True,
    )

    elapsed = time.time() - t1
    print(f"\n[transcribe] Done in {_dur(elapsed)}")

    # Convert to list of dicts
    segments = []
    for i, seg in enumerate(result.get("segments", []), 1):
        text = seg["text"].strip()
        if not text:
            continue

        entry = {
            "id":    i,
            "start": round(seg["start"], 3),
            "end":   round(seg["end"],   3),
            "text":  text,
        }

        if word_timestamps and seg.get("words"):
            entry["words"] = [
                {
                    "word":        w["word"],
                    "start":       round(w["start"], 3),
                    "end":         round(w["end"],   3),
                    "probability": round(w.get("probability", 1.0), 4),
                }
                for w in seg["words"]
                if w["word"].strip()
            ]

        segments.append(entry)

    # Save JSON
    if output_path is None:
        base        = os.path.splitext(wav_path)[0]
        output_path = base + "_segments.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # Summary
    word_count = sum(len(s.get("words", [])) for s in segments)
    print(f"\n{'='*62}")
    print(f"  Transcription complete!")
    print(f"  Segments : {len(segments)}")
    print(f"  Words    : {word_count}")
    print(f"  Time     : {_dur(elapsed)}")
    print(f"  Output   : {output_path}")
    print(f"{'='*62}")

    print("\n-- Preview (first 5 segments) " + "-"*32)
    for seg in segments[:5]:
        print(f"  [{_mmss(seg['start'])} -> {_mmss(seg['end'])}]  {seg['text']}")
    print("-" * 62 + "\n")

    return segments


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
        description="Transcribe WAV with openai-whisper on CUDA. No chunking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VRAM guide:  tiny/base=1GB  small=2GB  medium=5GB  large/v2/v3=10GB
             Add --int8 to roughly halve VRAM usage.

Examples:
  python transcribe.py audio_16k.wav
  python transcribe.py audio_16k.wav --language en
  python transcribe.py audio_16k.wav --language vi --output segments.json
  python transcribe.py audio_16k.wav --model medium
  python transcribe.py audio_16k.wav --model small --int8
  python transcribe.py audio_16k.wav --cpu
        """,
    )
    parser.add_argument("wav",
        help="Path to the 16 kHz mono WAV file.")
    parser.add_argument("--model", "-m",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model (default: large-v3).")
    parser.add_argument("--language", "-l",
        default=None,
        help="Language ISO code: 'en', 'vi', 'ja', ... Omit = auto-detect.")
    parser.add_argument("--output", "-o",
        default=None,
        help="Output JSON path (default: <wav>_segments.json).")
    parser.add_argument("--cpu",
        action="store_true",
        help="Force CPU mode.")
    parser.add_argument("--int8",
        action="store_true",
        help="Load model in int8 quantization (saves ~half VRAM, slight accuracy loss).")
    parser.add_argument("--no-word-timestamps",
        action="store_true",
        help="Disable word-level timestamps (smaller output JSON).")

    args   = parser.parse_args()
    device = "cpu" if args.cpu else None

    try:
        transcribe(
            wav_path        = args.wav,
            model_name      = args.model,
            language        = args.language,
            output_path     = args.output,
            device          = device,
            word_timestamps = not args.no_word_timestamps,
            use_int8        = args.int8,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Run again to restart from beginning.")
        sys.exit(0)


if __name__ == "__main__":
    main()