"""
pipeline.py  –  One-command subtitle generator
================================================
Runs the full pipeline end-to-end:
    MP4  →  WAV  →  Whisper segments  →  [translate]  →  SRT/SUB/VTT/ASS

Usage examples:

  # English movie → English subtitles (no translation)
  python pipeline.py movie.mp4

  # English movie → Vietnamese subtitles
  python pipeline.py movie.mp4 --target vi

  # French movie → English subtitles
  python pipeline.py movie.mp4 --source fr --target en

  # Choose model size (tiny/base/small/medium/large)
  python pipeline.py movie.mp4 --target vi --model medium

  # Output .sub instead of .srt
  python pipeline.py movie.mp4 --target vi --format sub

  # Keep intermediate WAV and JSON files
  python pipeline.py movie.mp4 --target vi --keep-temp
"""

import argparse
import os
import sys
import time

# Force UTF-8 output on Windows to avoid codec errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from extract_audio    import extract_audio
from transcribe       import transcribe
from translate        import translate_segments
from generate_subtitle import generate_subtitle, FORMAT_MAP


def run_pipeline(
    video_path: str,
    source_lang: str = "en",
    target_lang: str | None = None,
    model_name: str = "small",
    subtitle_format: str = "srt",
    output_path: str | None = None,
    fps: float = 23.976,
    keep_temp: bool = False,
) -> str:
    """
    Full pipeline: video → subtitle file.

    Args:
        video_path:       Input video file (MP4, MKV, AVI, …).
        source_lang:      ISO-639-1 code of the spoken language  (default: "en").
        target_lang:      ISO-639-1 code to translate into.
                          Pass None to skip translation (keep source language).
        model_name:       Whisper model size (tiny/base/small/medium/large).
        subtitle_format:  "srt", "vtt", "sub", or "ass".
        output_path:      Where to save the subtitle file.
                          Defaults to <video_name>.<ext> in the same directory.
        fps:              Frames-per-second (for .sub format only).
        keep_temp:        If True, keep intermediate WAV and JSON files.

    Returns:
        Path to the generated subtitle file.
    """
    pipeline_start = time.time()

    # ── Derive paths ──────────────────────────────────────────────────────────
    base, _ = os.path.splitext(video_path)
    wav_path  = base + "_audio.wav"
    json_path = base + "_segments.json"
    ext, _    = FORMAT_MAP[subtitle_format]

    if output_path is None:
        lang_tag = f".{target_lang}" if target_lang else ""
        output_path = base + lang_tag + ext

    print("=" * 60)
    print("  [PIPELINE]  Subtitle Generator")
    print("=" * 60)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {model_name}")
    print(f"  Source  : {source_lang}")
    print(f"  Target  : {target_lang or '(same as source, no translation)'}")
    print(f"  Format  : {subtitle_format.upper()}")
    print(f"  Output  : {output_path}")
    print("=" * 60)

    # ── Step 1: Extract audio ─────────────────────────────────────────────────
    print("\n[Step 1/4]  Extracting audio…")
    extract_audio(video_path, wav_path)

    # ── Step 2: Transcribe ────────────────────────────────────────────────────
    print("\n[Step 2/4]  Transcribing with Whisper…")
    segments = transcribe(
        wav_path=wav_path,
        model_name=model_name,
        language=source_lang,
        output_path=json_path,
    )
    print(f"           → {len(segments)} segments")

    # ── Step 3: Translate (optional) ──────────────────────────────────────────
    if target_lang and target_lang != source_lang:
        print(f"\n[Step 3/4]  Translating {source_lang}→{target_lang} with ArgosTranslate…")
        segments = translate_segments(segments, source_lang, target_lang)
    else:
        print("\n[Step 3/4]  Skipping translation (source == target or no target set).")

    # ── Step 4: Generate subtitle file ────────────────────────────────────────
    print(f"\n[Step 4/4]  Writing {subtitle_format.upper()} file…")
    generate_subtitle(segments, output_path, subtitle_format, fps)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not keep_temp:
        for tmp in [wav_path, json_path]:
            if os.path.isfile(tmp):
                os.remove(tmp)
                print(f"[pipeline]  Removed temp file: {tmp}")

    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  [DONE]  Finished in {elapsed:.1f}s")
    print(f"  [FILE]  Subtitle -> {output_path}")
    print(f"{'=' * 60}")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "All-in-one subtitle generator: MP4 → SRT/SUB/VTT/ASS\n"
            "Uses local Whisper (transcription) + ArgosTranslate (translation).\n"
            "No API key or internet connection needed after first install."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to the video file (MP4, MKV, AVI, …).")
    parser.add_argument(
        "--source", "-s",
        default="en",
        help="Language spoken in the video (ISO code, default: en).",
    )
    parser.add_argument(
        "--target", "-t",
        default=None,
        help=(
            "Translate subtitles to this language (ISO code, e.g. vi, fr, de, zh, ja, ko, es).\n"
            "Omit to keep the original language."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help=(
            "Whisper model size (default: small).\n"
            "  tiny   – fastest, lowest accuracy (~1 GB RAM)\n"
            "  base   – fast, decent accuracy\n"
            "  small  – good balance ← recommended\n"
            "  medium – high accuracy (~5 GB RAM)\n"
            "  large  – best accuracy (~10 GB RAM)"
        ),
    )
    parser.add_argument(
        "--format", "-f",
        default="srt",
        choices=list(FORMAT_MAP),
        help="Output subtitle format (default: srt).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output subtitle file path (auto-generated if omitted).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=23.976,
        help="Frames per second for .sub format (default: 23.976).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate WAV and JSON files.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(
            video_path=args.video,
            source_lang=args.source,
            target_lang=args.target,
            model_name=args.model,
            subtitle_format=args.format,
            output_path=args.output,
            fps=args.fps,
            keep_temp=args.keep_temp,
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
