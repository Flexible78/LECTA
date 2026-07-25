# LECTA — Setup Guide

This guide walks through a clean-clone installation from scratch.

---

## 1. Prerequisites

### Python

Install **Python 3.11 or newer**.

- **Windows:** download from [python.org](https://www.python.org/downloads/) or use `winget install Python.Python.3.11`
- **Linux (Ubuntu/Debian):** `sudo apt install python3 python3-pip python3-venv`

Verify:

```bash
python --version
```

### ffmpeg

LECTA uses `pydub` and `librosa`, both of which require `ffmpeg` on the system PATH.

- **Windows:** download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to PATH, or use `winget install Gyan.FFmpeg`
- **Linux:** `sudo apt install ffmpeg`

Verify:

```bash
ffmpeg -version
```

### Git

```bash
git --version
```

If missing: `winget install Git.Git` (Windows) or `sudo apt install git` (Linux).

---

## 2. Clone and set up a virtual environment

```bash
git clone https://github.com/Flexible78/LECTA.git
cd LECTA
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Optional document formats:** PDF, DOCX, EPUB and RTF support require extra
> packages that are not in `requirements.txt` by default. Install them only if
> you need those formats:
>
> ```bash
> pip install beautifulsoup4 striprtf PyPDF2 EbookLib python-docx
> ```
>
> `beautifulsoup4` is also needed for the web scraper (URL → article).

---

## 4. Voice models

LECTA does **not** ship model files — they are too large for Git.

### Option A: Download from inside the app (recommended)

1. Launch the app (see step 5 below).
2. Go to the **System & Cleanup** tab.
3. Click **⬇️⬇️ Update ALL models**.
4. Wait for the download to complete. Models are placed in `models/` automatically.

### Option B: Download manually

Models live in a `models/` folder next to the app. The expected layout is:

```
models/
├── vosk-model-tts-ru-0.10-multi/   # Vosk TTS (56 Russian voices)
├── silero/
│   ├── v5_5_ru.pt                    # Silero Russian (5 voices)
│   ├── v5_cis_base_nostress.pt       # Silero CIS (60 voices)
│   └── v3_en.pt                      # Silero English
├── F5TTS_v1_Base_v4_winter/
│   └── model_212000.safetensors      # F5-TTS (Misha24-10)
├── ESpeech-TTS-1_RL-V2/
│   └── espeech_tts_rlv2.pt           # F5-TTS (ESpeech)
├── vocos-mel-24khz/
│   ├── pytorch_model.bin             # Vocoder for F5-TTS
│   └── config.yaml
└── silero_stress/
    └── accentor.pt                   # Silero Stress (word stress)
```

Download URLs are listed in the **System & Cleanup** tab, or see the
`VOICE_MODELS_REGISTRY` in `libs/system_tools.py`.

### Custom models directory

If your models are stored elsewhere, set the `LECTA_MODELS_DIR` environment
variable before launching:

```bash
export LECTA_MODELS_DIR=/path/to/your/models
python app.py
```

---

## 5. Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LECTA_PORT` | `7860` | Port for the Gradio UI |
| `LECTA_PROXY_PORT` | `8080` | Port for the Gemini API proxy (`start.py`) |
| `LECTA_MODELS_DIR` | `models` | Directory containing voice model files |
| `LECTA_TTS_WORKERS` | `8` | Number of parallel TTS worker threads |

All variables are optional. The defaults work for a standard local setup.

---

## 6. Verification (without launching the GUI)

You can verify the installation without starting the web UI:

```bash
# Verify all Python files parse correctly
python -c "import ast,pathlib;[ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py') if 'venv' not in str(p)];print('AST OK')"

# Verify core modules import without errors
python -c "from config import AppConfig, config, TTS_WORKERS; print('config OK, TTS_WORKERS =', TTS_WORKERS)"
python -c "from libs.utils import data_path, models_path; print('data_path:', data_path, '| models_path:', models_path)"
python -c "from libs.ui_assets import custom_css, custom_head; print('ui_assets OK')"
```

If all three print without errors, the installation is correct.

---

## 7. Launch

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

The Gemini API proxy (`start.py`) is only needed if you use the built-in AI
assistant features. It requires a Gemini CLI OAuth token at `~/.gemini/oauth_creds.json`.

```bash
python start.py     # runs on port 8080 by default
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'xxx'`** — run `pip install -r requirements.txt` again inside the activated venv.
- **Model folder not found** — set `LECTA_MODELS_DIR` or download models from the System tab.
- **Port already in use** — set `LECTA_PORT` to a different port.
- **CUDA out of memory** — switch the compute device to `cpu` in the sidebar, or reduce `LECTA_TTS_WORKERS`.
