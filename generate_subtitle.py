"""
generate_subtitle.py — Convert segments JSON → subtitle files
=============================================================
Supported formats:
  • .srt  — SubRip (universal: VLC, MPV, Windows Media Player)
  • .vtt  — WebVTT (HTML5 video, browsers)
  • .sub  — MicroDVD (frame-based)
  • .ass  — Advanced SubStation Alpha with:
              - Word-level karaoke highlight ({\kf} tags)
              - Per-frame precision using word timestamps
              - Clean cinematic style (1920×1080)

Input JSON format (from transcribe_fast.py):
    [
      {
        "id": 1,
        "start": 12.34,
        "end": 15.67,
        "text": "Hello world",
        "words": [                          ← optional, from --word-timestamps
          {"word": "Hello", "start": 12.34, "end": 13.10, "probability": 0.99},
          {"word": "world", "start": 13.15, "end": 15.67, "probability": 0.97}
        ]
      },
      ...
    ]

Usage:
    python generate_subtitle.py segments.json
    python generate_subtitle.py segments.json --format srt
    python generate_subtitle.py segments.json --format ass --output movie.ass
    python generate_subtitle.py segments.json --format ass --word-highlight
    python generate_subtitle.py segments.json --format srt --word-level
"""

import argparse
import json
import os
import sys
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Timestamp helpers
# ══════════════════════════════════════════════════════════════════════════════

def _srt_ts(seconds: float) -> str:
    """Float seconds → SRT timestamp  HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    """Float seconds → VTT timestamp  HH:MM:SS.mmm"""
    return _srt_ts(seconds).replace(",", ".")


def _ass_ts(seconds: float) -> str:
    """Float seconds → ASS timestamp  H:MM:SS.cc  (centiseconds)"""
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _frames(seconds: float, fps: float) -> int:
    """Float seconds → frame number for MicroDVD .sub"""
    return int(round(seconds * fps))


# ══════════════════════════════════════════════════════════════════════════════
# SRT
# ══════════════════════════════════════════════════════════════════════════════

def to_srt(segments: list, word_level: bool = False) -> str:
    """
    Generate SRT content.

    word_level=True → create one entry per word (ultra-precise per-frame sync).
    word_level=False → one entry per segment (standard).
    """
    lines = []
    idx   = 1

    if word_level:
        for seg in segments:
            for w in seg.get("words", []):
                word = w["word"].strip()
                if not word:
                    continue
                lines.append(str(idx))
                lines.append(f"{_srt_ts(w['start'])} --> {_srt_ts(w['end'])}")
                lines.append(word)
                lines.append("")
                idx += 1
    else:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            lines.append(str(idx))
            lines.append(f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}")
            lines.append(text)
            lines.append("")
            idx += 1

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# VTT
# ══════════════════════════════════════════════════════════════════════════════

def to_vtt(segments: list, word_level: bool = False) -> str:
    """Generate WebVTT content."""
    lines = ["WEBVTT", ""]
    idx   = 1

    if word_level:
        for seg in segments:
            for w in seg.get("words", []):
                word = w["word"].strip()
                if not word:
                    continue
                lines.append(str(idx))
                lines.append(f"{_vtt_ts(w['start'])} --> {_vtt_ts(w['end'])}")
                lines.append(word)
                lines.append("")
                idx += 1
    else:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            lines.append(str(idx))
            lines.append(f"{_vtt_ts(seg['start'])} --> {_vtt_ts(seg['end'])}")
            lines.append(text)
            lines.append("")
            idx += 1

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MicroDVD .sub
# ══════════════════════════════════════════════════════════════════════════════

def to_sub(segments: list, fps: float = 23.976, **_) -> str:
    """Generate MicroDVD .sub content (frame-based)."""
    lines = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        sf_ = _frames(seg["start"], fps)
        ef  = _frames(seg["end"],   fps)
        lines.append(f"{{{sf_}}}{{{ef}}}{text}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ASS — Advanced SubStation Alpha with word highlight & per-frame precision
# ══════════════════════════════════════════════════════════════════════════════

_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,0.5,0,1,2.5,1,2,80,80,55,1
Style: Highlight,Arial,52,&H0000FFFF,&H000000FF,&H00000000,&HAA000000,-1,0,0,0,100,100,0.5,0,1,2.5,1,2,80,80,55,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_karaoke_line(seg: dict) -> Optional[str]:
    """
    Build one ASS Dialogue line with {\kf} karaoke tags so each word
    lights up in a highlight colour exactly when it is spoken.

    Requires the segment to have a 'words' list with per-word timestamps.
    Falls back to a plain line if no word data is available.
    """
    words = seg.get("words", [])

    if not words:
        # Fallback: plain line, no karaoke
        text  = seg["text"].strip()
        start = _ass_ts(seg["start"])
        end   = _ass_ts(seg["end"])
        return f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"

    # Build karaoke tags {\kfN} where N = duration in centiseconds
    # {\kf} = karaoke fill — word sweeps from left to right over duration N*10ms
    parts = []
    prev_end = seg["start"]

    for w in words:
        word      = w["word"]
        w_start   = w["start"]
        w_end     = w["end"]

        # Gap between previous word and this one → silent {\kf} tag
        gap_cs = max(0, int(round((w_start - prev_end) * 100)))
        if gap_cs > 0:
            parts.append(f"{{\\kf{gap_cs}}}")

        # Word duration in centiseconds
        dur_cs = max(1, int(round((w_end - w_start) * 100)))
        parts.append(f"{{\\kf{dur_cs}}}{word}")
        prev_end = w_end

    karaoke_text = "".join(parts)
    start = _ass_ts(seg["start"])
    end   = _ass_ts(seg["end"])
    return f"Dialogue: 0,{start},{end},Default,,0,0,0,,{karaoke_text}"


def to_ass(
    segments: list,
    word_highlight: bool = True,
    word_level: bool     = False,
    **_,
) -> str:
    """
    Generate Advanced SubStation Alpha (.ass) content.

    word_highlight=True → use {\\kf} karaoke tags (word-by-word highlight).
    word_level=True     → one dialogue entry per word (max per-frame precision).
    """
    dialogue_lines = []

    if word_level:
        # One ASS entry per word — absolute per-frame sync
        for seg in segments:
            for w in seg.get("words", []):
                word = w["word"].strip()
                if not word:
                    continue
                start = _ass_ts(w["start"])
                end   = _ass_ts(w["end"])
                dialogue_lines.append(
                    f"Dialogue: 0,{start},{end},Default,,0,0,0,,{word}"
                )
    elif word_highlight:
        for seg in segments:
            line = _ass_karaoke_line(seg)
            if line:
                dialogue_lines.append(line)
    else:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            start = _ass_ts(seg["start"])
            end   = _ass_ts(seg["end"])
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            )

    return _ASS_HEADER + "\n".join(dialogue_lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

FORMAT_MAP = {
    "srt": (".srt", to_srt),
    "vtt": (".vtt", to_vtt),
    "sub": (".sub", to_sub),
    "ass": (".ass", to_ass),
}


def generate_subtitle(
    segments: list,
    output_path: str,
    fmt: str             = "srt",
    fps: float           = 23.976,
    word_level: bool     = False,
    word_highlight: bool = True,
) -> str:
    """
    Write a subtitle file from timed segments.

    Args:
        segments:        List of dicts from transcribe_fast (with optional 'words').
        output_path:     Where to write the subtitle file.
        fmt:             "srt" | "vtt" | "sub" | "ass".
        fps:             Frames per second (only for .sub).
        word_level:      One entry per word (ultra-precise per-frame sync).
        word_highlight:  ASS only — karaoke-style word highlight (default: True).

    Returns:
        Path to the written file.
    """
    fmt = fmt.lower()
    if fmt not in FORMAT_MAP:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {list(FORMAT_MAP)}")

    _, generator = FORMAT_MAP[fmt]

    if fmt == "sub":
        content = generator(segments, fps=fps)
    elif fmt == "ass":
        content = generator(segments, word_highlight=word_highlight, word_level=word_level)
    else:
        content = generator(segments, word_level=word_level)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        # utf-8-sig adds BOM — helps media players detect UTF-8 correctly
        f.write(content)

    word_count = sum(len(seg.get("words", [])) for seg in segments)
    has_words  = word_count > 0
    print(f"[generate_subtitle] Saved {fmt.upper()} → {output_path}")
    print(f"[generate_subtitle] {len(segments)} segments | "
          f"{word_count} word timestamps | "
          f"word-level mode: {'yes' if word_level else 'no'}")
    return output_path


def generate_from_json(
    json_path: str,
    fmt: str                 = "srt",
    output_path: Optional[str] = None,
    fps: float               = 23.976,
    word_level: bool         = False,
    word_highlight: bool     = True,
) -> str:
    """Load segments from a JSON file and write a subtitle file."""
    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)

    if output_path is None:
        base, _  = os.path.splitext(json_path)
        ext, _   = FORMAT_MAP[fmt]
        output_path = base + ext

    return generate_subtitle(
        segments, output_path, fmt=fmt, fps=fps,
        word_level=word_level, word_highlight=word_highlight,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert segments JSON → subtitle file (.srt/.vtt/.sub/.ass).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_subtitle.py segments.json
  python generate_subtitle.py segments.json --format srt
  python generate_subtitle.py segments.json --format ass --output movie.ass
  python generate_subtitle.py segments.json --format ass --word-highlight
  python generate_subtitle.py segments.json --format srt --word-level
        """,
    )
    parser.add_argument("input",
        help="Path to segments JSON file (from transcribe_fast.py).")
    parser.add_argument("--format", "-f",
        default="srt",
        choices=list(FORMAT_MAP),
        help="Output subtitle format (default: srt).")
    parser.add_argument("--output", "-o",
        default=None,
        help="Output subtitle path (default: <input>.<ext>).")
    parser.add_argument("--fps",
        type=float, default=23.976,
        help="FPS for .sub format (default: 23.976).")
    parser.add_argument("--word-level",
        action="store_true",
        help="One subtitle entry per word (ultra-precise per-frame sync). "
             "Requires word timestamps in JSON.")
    parser.add_argument("--word-highlight",
        action="store_true",
        help="ASS only — karaoke-style word highlight using {\\kf} tags.")
    parser.add_argument("--no-word-highlight",
        action="store_true",
        help="ASS only — disable karaoke highlight, use plain text.")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    word_highlight = not args.no_word_highlight  # default True unless disabled

    out = generate_from_json(
        json_path       = args.input,
        fmt             = args.format,
        output_path     = args.output,
        fps             = args.fps,
        word_level      = args.word_level,
        word_highlight  = word_highlight,
    )
    print(f"\n✅ Subtitle written → {out}")


if __name__ == "__main__":
    main()
