# 🎬 Subtitle Generator – Free & Local

Generate subtitles for any movie, **no API key, no billing, no internet after setup**.

Uses:
- **[Whisper](https://github.com/openai/whisper)** – speech-to-text (by OpenAI, runs on your PC)
- **[ArgosTranslate](https://github.com/argosopentech/argos-translate)** – translation (fully local)
- **ffmpeg** – audio extraction

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `extract_audio.py` | Step 1 — Extract WAV audio from video |
| `transcribe.py` | Step 2 — Transcribe audio → timestamped text |
| `translate.py` | Step 3 — Translate segments to another language |
| `generate_subtitle.py` | Step 4 — Write .srt / .sub / .vtt / .ass file |
| `pipeline.py` | **All-in-one** — runs all 4 steps with one command |

---

## ⚙️ Setup (One-Time)

### 1. Install ffmpeg

Download from https://ffmpeg.org/download.html  
→ Windows builds: https://www.gyan.dev/ffmpeg/builds/

After downloading, add `ffmpeg/bin` to your system PATH.  
Verify: open CMD and run `ffmpeg -version`

### 2. Create a Python virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

> **GPU (NVIDIA)?** Run this for much faster transcription:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. First run — model download

On the **first** run, Whisper will download the model (~240 MB for `small`).  
On the **first** translation, ArgosTranslate downloads the language pack (~100 MB per pair).  
Both are cached automatically — no re-download needed.

---

## 🚀 Usage

### Quick start — one command does everything

```bash
# English movie → English subtitle (.srt)
python pipeline.py movie.mp4

# English movie → Vietnamese subtitle
python pipeline.py movie.mp4 --target vi

# Japanese movie → English subtitle
python pipeline.py movie.mp4 --source ja --target en

# French movie → Vietnamese subtitle
python pipeline.py movie.mp4 --source fr --target vi

# Use a more accurate model (needs more RAM/time)
python pipeline.py movie.mp4 --target vi --model medium

# Output as .sub instead of .srt
python pipeline.py movie.mp4 --target vi --format sub
```

### Run each step separately

```bash
# Step 1: Extract audio
python extract_audio.py movie.mp4

# Step 2: Transcribe (creates segments JSON)
python transcribe.py movie.wav --model small --language en

# Step 3: Translate segments (optional)
python translate.py movie_segments.json --target vi

# Step 4: Generate subtitle file
python generate_subtitle.py movie_segments_translated_vi.json --format srt
```

---

## 🌐 Supported Languages

**Whisper** understands 90+ languages automatically.  
**ArgosTranslate** supports common pairs such as:

| Code | Language |
|------|----------|
| `vi` | Vietnamese |
| `en` | English |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `ar` | Arabic |
| `pt` | Portuguese |
| `ru` | Russian |
| `it` | Italian |

> Not all pairs are available. If a pair is missing,
> the script tells you what's available.

---

## 📺 Using the Subtitle in Windows Media Player / VLC

### Windows Media Player / Movies & TV

Place the `.srt` file in the **same folder** as the video,  
and give it the **same name**:

```
movie.mp4
movie.srt      ← subtitle file
```

Open the video → right-click → **Subtitles** → select the file.

### VLC

Open VLC → **Subtitle** menu → **Add Subtitle File** → pick your `.srt`.

Or: place `.srt` next to `.mp4` with the same name — VLC loads it automatically.

---

## 🔧 Model Size Guide

| Model | Size | Speed | Accuracy | RAM needed |
|-------|------|-------|----------|-----------|
| `tiny` | 39 MB | ⚡⚡⚡ | ★★☆☆☆ | ~1 GB |
| `base` | 74 MB | ⚡⚡ | ★★★☆☆ | ~1 GB |
| `small` | 244 MB | ⚡ | ★★★★☆ | ~2 GB |
| `medium` | 769 MB | 🐢 | ★★★★★ | ~5 GB |
| `large` | 1.5 GB | 🐢🐢 | ★★★★★ | ~10 GB |

**Recommendation**: Start with `small`. If accuracy is not enough, try `medium`.

---

## ❓ FAQ

**Q: Do I need an internet connection to use this?**  
A: Only for the first download of models and language packs. After that — fully offline.

**Q: How long does it take?**  
A: ~1–3× real-time on CPU. A 2-hour movie takes ~2–6 hours on CPU.  
With an NVIDIA GPU it's ~5–10× faster (20–40 minutes).

**Q: The subtitles are slightly off-sync. What do I do?**  
A: Try the `medium` or `large` model for better timing accuracy.  
In VLC you can also press `G` / `H` to shift subtitles forward/backward.

**Q: The translation looks a bit rough. Why?**  
A: ArgosTranslate is a lightweight local model, so quality is good but not perfect.
For a student with no budget this is the best free option. For critical use,
consider DeepL Free (500 000 chars/month free with account) or LibreTranslate.
