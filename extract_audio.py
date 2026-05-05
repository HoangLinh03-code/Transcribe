"""
Step 1: extract_audio.py  (improved)
=====================================
Extract audio from an MP4 file.

Two output modes:
  • 16 kHz mono WAV  → for Whisper transcription  (always created)
  • High-quality WAV → for human listening / QC    (optional, --hq flag)

Why two files?
  16 kHz mono sounds like a telephone — that's normal and intentional.
  Whisper was trained on 16 kHz audio, so that setting gives the best
  transcription accuracy. But if you want to listen and check the audio
  yourself, use --hq to get a full-quality copy as well.

Usage:
    python extract_audio.py movie.mp4
    python extract_audio.py movie.mp4 --hq
    python extract_audio.py movie.mp4 --output audio_16k.wav --hq
"""

import argparse
import os
import sys
import ffmpeg
import imageio_ffmpeg


def extract_audio(
    input_path: str,
    output_path: str | None = None,
    also_extract_hq: bool = False,
) -> dict[str, str]:
    """
    Extract audio from a video file.

    Args:
        input_path:       Path to the input video file (mp4, mkv, avi, …).
        output_path:      Path for the 16 kHz mono WAV (Whisper input).
                          Defaults to <input_name>_16k.wav
        also_extract_hq:  If True, also save a high-quality stereo WAV
                          at 44.1 kHz so you can listen and check audio.
                          Saved as <input_name>_hq.wav

    Returns:
        Dict with keys:
            "whisper_wav" → path to the 16 kHz mono WAV  (always present)
            "hq_wav"      → path to high-quality WAV      (present if also_extract_hq=True)
    """
    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base, _ = os.path.splitext(input_path)

    # ── Output paths ──────────────────────────────────────────────────────────
    whisper_wav = output_path if output_path else base + "_16k.wav"
    hq_wav      = base + "_hq.wav"

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[extract_audio] Input      : {input_path}")
    print(f"[extract_audio] ffmpeg bin : {ffmpeg_bin}")

    # ── Extract 16 kHz mono WAV (for Whisper) ─────────────────────────────────
    # Settings explained:
    #   ac=1            → mono  (Whisper was trained on mono audio)
    #   ar=16000        → 16 kHz sample rate (Whisper requirement)
    #   acodec=pcm_s16le→ uncompressed 16-bit PCM, the standard WAV format
    #   af=loudnorm     → normalize loudness so quiet speech is boosted,
    #                     this significantly helps Whisper catch quiet words
    print(f"\n[extract_audio] Extracting 16 kHz WAV for Whisper → {whisper_wav}")
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                whisper_wav,
                ac=1,
                ar=16000,
                vn=None,
                acodec="pcm_s16le",
                af="loudnorm",          # ← KEY FIX: normalize volume for better transcription
            )
            .overwrite_output()
            .run(quiet=True, cmd=ffmpeg_bin)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        print(f"[extract_audio] ffmpeg error:\n{stderr}", file=sys.stderr)
        raise

    size_mb = os.path.getsize(whisper_wav) / (1024 * 1024)
    print(f"[extract_audio] ✅ Whisper WAV saved ({size_mb:.1f} MB) → {whisper_wav}")

    result = {"whisper_wav": whisper_wav}

    # ── Extract high-quality WAV for human listening (optional) ───────────────
    # Settings explained:
    #   ac=2            → stereo (natural sounding)
    #   ar=44100        → 44.1 kHz (CD quality, full frequency range)
    #   acodec=pcm_s16le→ uncompressed WAV
    #   (no loudnorm)   → keep the original mix, not altered
    if also_extract_hq:
        print(f"\n[extract_audio] Extracting high-quality WAV for listening → {hq_wav}")
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    hq_wav,
                    ac=2,
                    ar=44100,
                    vn=None,
                    acodec="pcm_s16le",
                )
                .overwrite_output()
                .run(quiet=True, cmd=ffmpeg_bin)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            print(f"[extract_audio] ffmpeg error (hq):\n{stderr}", file=sys.stderr)
            raise

        size_mb = os.path.getsize(hq_wav) / (1024 * 1024)
        print(f"[extract_audio] ✅ HQ WAV saved ({size_mb:.1f} MB) → {hq_wav}")
        result["hq_wav"] = hq_wav

    print("\n[extract_audio] Note: The 16k WAV will sound like telephone quality.")
    print("[extract_audio] That is NORMAL — Whisper is trained on 16 kHz audio.")
    print("[extract_audio] Use the _hq.wav file if you want to listen and verify.\n")

    return result


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from a video file → WAV (16 kHz for Whisper)."
    )
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the 16 kHz WAV. Defaults to <input_name>_16k.wav",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="Also extract a full-quality 44.1 kHz stereo WAV for human listening.",
    )
    args = parser.parse_args()

    try:
        paths = extract_audio(args.input, args.output, also_extract_hq=args.hq)
        print("✅ Done!")
        print(f"   Whisper WAV : {paths['whisper_wav']}")
        if "hq_wav" in paths:
            print(f"   HQ WAV      : {paths['hq_wav']}  ← listen to this one")
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except ffmpeg.Error:
        sys.exit(1)


if __name__ == "__main__":
    main()