"""
Step 3: translate.py
====================
Translate subtitle segments to another language using ArgosTranslate
(100% local, no API key, no internet needed after first install).

Supported language pairs depend on installed ArgosTranslate packages.
The script automatically downloads the required package on first use.

Usage:
    python translate.py segments.json --target vi
    python translate.py segments.json --source en --target vi
    python translate.py segments.json --target vi --output translated.json
"""

import argparse
import json
import os
import sys
from typing import Optional


def _ensure_argos_package(source_lang: str, target_lang: str):
    """Download and install the ArgosTranslate package if not already present."""
    import argostranslate.package
    import argostranslate.translate

    # Check if the pair is already installed
    installed = argostranslate.translate.get_installed_languages()
    installed_codes = [lang.code for lang in installed]

    pair_ready = False
    for lang in installed:
        if lang.code == source_lang:
            for t in lang.translations_to:
                if t.code == target_lang:
                    pair_ready = True
                    break

    if pair_ready:
        print(f"[translate] ArgosTranslate package {source_lang}→{target_lang} already installed.")
        return

    # Need to download
    print(f"[translate] Downloading ArgosTranslate package for {source_lang}→{target_lang}…")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    pkg = next(
        (p for p in available
         if p.from_code == source_lang and p.to_code == target_lang),
        None,
    )

    if pkg is None:
        codes = sorted({f"{p.from_code}→{p.to_code}" for p in available})
        raise ValueError(
            f"No ArgosTranslate package found for {source_lang}→{target_lang}.\n"
            f"Available pairs:\n  " + "\n  ".join(codes)
        )

    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)
    print(f"[translate] Package installed: {source_lang}→{target_lang}")


def translate_segments(
    segments: list[dict],
    source_lang: str = "en",
    target_lang: str = "vi",
) -> list[dict]:
    """
    Translate the 'text' field of each segment.

    Args:
        segments:    List of segment dicts with at least {"id", "start", "end", "text"}.
        source_lang: ISO-639-1 code of the source language (default: "en").
        target_lang: ISO-639-1 code of the target language (default: "vi").

    Returns:
        New list of segment dicts with translated 'text' and original 'original_text'.
    """
    import argostranslate.translate

    _ensure_argos_package(source_lang, target_lang)

    # Get the translator object
    installed = argostranslate.translate.get_installed_languages()
    src_lang_obj = next((l for l in installed if l.code == source_lang), None)
    if src_lang_obj is None:
        raise RuntimeError(f"Source language '{source_lang}' not found after install.")

    translator = src_lang_obj.get_translation(
        next(l for l in installed if l.code == target_lang)
    )

    translated = []
    total = len(segments)
    for i, seg in enumerate(segments, 1):
        original_text = seg["text"]
        # Skip empty segments
        if not original_text.strip():
            translated.append({**seg, "original_text": original_text})
            continue

        translated_text = translator.translate(original_text)
        translated.append({
            **seg,
            "original_text": original_text,
            "text": translated_text,
        })

        # Progress indicator every 10 segments
        if i % 10 == 0 or i == total:
            print(f"[translate] {i}/{total} segments translated…")

    return translated


def translate_file(
    input_path: str,
    source_lang: str = "en",
    target_lang: str = "vi",
    output_path: Optional[str] = None,
) -> list[dict]:
    """Load segments from JSON, translate, save."""
    with open(input_path, encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[translate] {len(segments)} segments loaded from {input_path}")
    translated = translate_segments(segments, source_lang, target_lang)

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_translated_{target_lang}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print(f"[translate] Saved → {output_path}")
    return translated


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Translate subtitle segments (JSON) using local ArgosTranslate."
    )
    parser.add_argument(
        "input",
        help="Path to the segments JSON file (output of transcribe.py).",
    )
    parser.add_argument(
        "--source", "-s",
        default="en",
        help="Source language ISO code (default: en).",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target language ISO code, e.g. vi, fr, de, ja, zh, es, ko…",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path. Defaults to <input>_translated_<target>.json",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        segments = translate_file(args.input, args.source, args.target, args.output)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ {len(segments)} segments translated ({args.source}→{args.target}).")
    print("\n── Preview (first 5) ───────────────────────────────────────")
    for seg in segments[:5]:
        ts = f"[{seg['start']:>8.3f}s → {seg['end']:>8.3f}s]"
        print(f"  {ts}  {seg['text']}")


if __name__ == "__main__":
    main()
