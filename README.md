# LECTA

**Text-to-speech for Russian, English and Hebrew.**

> *KATAV writes, LECTA reads.* — [KATAV](https://github.com/Flexible78/KATAV) is the companion speech-to-text tool.

LECTA turns books, articles and pasted text into natural-sounding audiobooks.
It routes each sentence to the best available engine based on its language,
mixing local neural TTS with cloud voices in a single output file.

---

## Acknowledgements

LECTA is built on [fb2tts](https://gitverse.ru/diger/fb2tts) by diger.
This fork adds multilingual routing for Russian, English and Hebrew,
Edge cloud voices, a vocabulary tab and batch synthesis.

Third-party engines and models:

| Component | Source |
|-----------|--------|
| [Vosk TTS](https://github.com/alphacep/vosk-tts) | Alpha Cephei |
| [Silero](https://github.com/snakers4/silero-models) | Silero |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | SWivid |
| F5-TTS model weights — [Misha24-10](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) | Misha24-10 |
| F5-TTS model weights — [ESpeech](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2) | ESpeech |
| [RuAccent](https://gitverse.ru/Den4ikAI/ruaccent) | Den4ikAI |
| [Silero Stress](https://github.com/snakers4/silero-stress) | Silero |
| [Gradio](https://gradio.app) | Gradio |

See [NOTICE.md](NOTICE.md) for full attribution and licensing notes.

---

## Features

- **Multilingual routing** — automatically detects Russian, English and Hebrew in the same text and sends each segment to the appropriate engine.
- **Edge cloud voices** — instant generation via Microsoft Edge TTS for all three languages (requires internet; recommended for Hebrew, which has no local backend).
- **Local backends** — Vosk TTS (56 Russian voices), Silero (Russian + English), F5-TTS (high-quality neural Russian via Misha24-10 and ESpeech models).
- **LECTA-branded favicon** — an open book with a sound wave, generated programmatically via `tools/make_favicon.py` (Pillow). Multi-layer ICO (16–256 px) plus 512 px PNG.
- **FB2 and document parsing** — load FB2, PDF, DOCX, EPUB, RTF, HTML, TXT, Markdown, JSON and CSV. Articles can be fetched directly from a URL.
- **Vocabulary tab** — extract unique words from a text, translate them via Google Translate, and build a pronunciation dictionary.
- **Batch synthesis** — process all projects in one run with a two-level progress bar, per-project download and a ZIP of all results.
- **Stress placement** — RuAccent or Silero Stress for correct Russian word stress.
- **Model management** — delete voice models with on-disk size display and free-space reporting. Active models are protected from accidental deletion.
- **Find & remove fragments** — multi-replace panel with clipboard integration, regex support, whole-line mode, and undo stack (up to 10 steps).
- **Custom sounds** — insert event sounds, pauses and background music at specific text markers.

---

## Quick start

```bash
git clone https://github.com/Flexible78/LECTA.git
cd LECTA
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:7860** in your browser.

Voice models are downloaded from the **System & Cleanup** tab inside the app — no manual download needed for the standard set.

---

## Requirements

- **OS:** Windows 10/11 (primary), Linux (Ubuntu/Debian)
- **Python:** 3.11 or newer
- **ffmpeg:** required by `pydub` / `librosa` for audio processing
- **GPU (optional):** CUDA-capable GPU with ≥ 6 GB VRAM for local neural TTS. Falls back to CPU automatically.

---

## Troubleshooting

<details>
<summary><b>Model folder not found</b></summary>

If the app reports that voice models are missing:

1. Set `LECTA_MODELS_DIR` to the folder containing your models:
   ```bash
   export LECTA_MODELS_DIR=/path/to/models
   ```
2. Or check the settings file (`user_settings.json`) for a custom path.
3. Or place a `models/` folder next to the LECTA application.

You can also download models from the **System & Cleanup → Update / download voice models** tab.
</details>

<details>
<summary><b>Port already in use</b></summary>

If the Gradio UI cannot start because port 7860 is occupied:

```bash
export LECTA_PORT=7861
python app.py
```

For the Gemini proxy (`start.py`), use `LECTA_PROXY_PORT` instead:

```bash
export LECTA_PROXY_PORT=8081
python start.py
```
</details>

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LECTA_PORT` | `7860` | Port for the Gradio UI (`app.py`) |
| `LECTA_PROXY_PORT` | `8080` | Port for the Gemini API proxy (`start.py`) |
| `LECTA_MODELS_DIR` | `models` | Directory containing voice model files |
| `LECTA_TTS_WORKERS` | `8` | Number of parallel TTS worker threads |

---

## Tested on

- Windows 11, Python 3.11, CUDA 12.x, ffmpeg N-122401
- Ubuntu 22.04, Python 3.11, CPU-only

---

## Documentation

- [Setup guide](docs/SETUP.md) — clean-clone installation, model placement, verification
- [Usage guide](docs/USAGE.md) — end-user guide to all tabs and features
- [NOTICE.md](NOTICE.md) — attribution and licensing notes

## Author

Alexander Tsyrkin (Flexible78)
