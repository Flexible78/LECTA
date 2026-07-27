# LECTA — Usage Guide

This guide covers the end-user workflow: loading text, choosing voices,
synthesising speech, and using the vocabulary and batch features.

---

## Sidebar

The sidebar is always visible on the left.

| Control | Description |
|---------|-------------|
| **Select TTS model** | Choose the speech engine: Vosk 0.10, Silero v5_5, Silero v5_cis, Misha24-10 (F5-TTS), ESpeech-TTS (F5-TTS), Silero English v3 |
| **Stress placement** | Choose the stress model: RuAccent or Silero Stress (Russian only) |
| **Compute device** | `auto` = GPU if available (fast, ≥ 6 GB VRAM), `cpu` = RAM only (slower) |
| **⚡ Russian via cloud** | Use Microsoft Edge TTS for Russian (instant, needs internet, voice: Dmitry) |
| **⚡ English via cloud** | Use Microsoft Edge TTS for English |
| **⚡ Hebrew via cloud** | Use Microsoft Edge TTS for Hebrew (recommended — no local Hebrew backend exists) |
| **📚 Dictionary mode** | Insert a 1-second pause between language switches |
| **🔄 Restart** | Restart the server |
| **🚪 Quit** | Stop the server |

> Cloud voices are generated in parallel and cached to disk (`data/tts_cache/`).
> The first request for a phrase is slower; subsequent requests are instant.

---

## Tab 1 — 📁 Project Manager

This is the main tab. A **project** is a folder under `data/` containing an FB2
file, parsed XML sections, a cover image, and generated audio.

### Loading text

Three ways to create a project:

1. **Upload file(s)** — drag FB2, PDF, DOCX, EPUB, RTF, HTML, TXT, Markdown,
   JSON or CSV files into the upload area, or paste file paths in the text box.
   Click **⬇️ Upload file(s)**. Each file is converted to FB2 and auto-parsed.

2. **Download article** — paste a web URL and click **⬇️ Download article**.
   The page text is extracted and saved as an FB2 project.

3. **From clipboard** — click **📋 From clipboard** to paste file paths or a URL
   from the clipboard.

A live progress bar shows parsing status per project.

### Managing projects

- **Select project** dropdown — choose which project to work with.
- **✏️ Rename** — rename the project folder.
- **❌ Delete** — remove a single project.
- **💣 Delete ALL** — remove all projects (irreversible).

### Inner tabs

When a project is selected, four inner tabs appear:

| Tab | Purpose |
|-----|---------|
| **Cover** | View and edit the audiobook cover image |
| **🔍 ANALYZE** | Parse/re-parse the FB2 file into XML sections, edit parsed text |
| **📚 Vocabulary Parser** | Extract and translate unique words |
| **🎧 TTS** | Synthesise speech from the parsed sections |

---

## Tab 2 — 📝 Create FB2 / Editor

Create a new project from raw pasted text, or edit an existing one.

- **Project name** — used as the folder and file name.
- **Text** — paste text here. Each line becomes a separate paragraph in the FB2.
  You can also paste SRT subtitles and clean them with the **✂️ Clean SRT** button.
- **✨ Save (Ctrl+Enter)** — create the project.
- **🔄 Overwrite** — replace the content of an existing project.
- **🗑 Delete project** — remove the project.

---

## Tab 3 — 🎙️ Demo TTS

Quick test of a single phrase without creating a project.

- **Select voice** — choose from the loaded model's voice list.
- **Set speed / Noise level / Pitch** — adjust synthesis parameters.
- **Text** — type any text. The placeholder shows supported languages:
  `English | עברית | Русский`.
- **Add stress marks** — insert stress markers into Russian text.
- **Convert to speech (Ctrl+Enter)** — generate audio.

> For F5-TTS models (Misha24-10, ESpeech), a **voice sample** section appears.
> Upload a reference audio clip and enter the text spoken in it for voice cloning.

---

## Tab 4 — 🛠️ System & Cleanup

### Clean tmp folder

Removes temporary files from `tmp/`.

### Delete model

Remove a model folder from `models/`. Each model shows its on-disk size next to
its name (e.g. `F5TTS_v1_Base_v4_winter — 1.26 GB`).

**Where to find it:**
- Top of the **🛠️ System & Cleanup** tab — next to the tmp folder cleaner.
- Also duplicated inside the **Update / download voice models** section so you
  can delete a model right where you download it.

**Before deleting:**
- Check the **Disk space** field at the top of the tab — it shows total model
  size and free disk space. It refreshes after every deletion.
- Tick the **I understand this deletes files from disk** checkbox. Without it,
  the delete button returns a warning and does nothing. No popup dialogs.

**Protection:** you cannot delete a model that is currently loaded in the
sidebar (TTS model or stress model). The app will ask you to switch to
another model first.

**After deletion:** the dropdowns update, the disk space line recalculates, and
the status confirms `✅ Deleted: <name> — freed X.XX GB`. The model can be
downloaded again from the update section below.

### Update / download voice models

- **Select model to update** — dropdown showing all models with ✅ (installed)
  or ❌ (missing) and on-disk size.
- **⬇️ Update selected** — download or refresh one model.
- **⬇️⬇️ Update ALL models** — download all missing models with a live
  progress log.
- **🔍 Check for updates** — check which models are installed vs missing
  (no download, just status).
- **🛑 Abort** — stop the current download after the current file.

Models and their download URLs:

| Model ID | Description |
|----------|-------------|
| `vosk_010` | Vosk 0.10 (56 Russian voices) |
| `silero_ru` | Silero v5_5 (Russian, 5 voices) |
| `silero_cis` | Silero v5_cis (60 voices) |
| `silero_en` | Silero English v3 |
| `f5_misha` | Misha24-10 (F5-TTS, high-quality Russian) |
| `f5_espeech` | ESpeech-TTS (F5-TTS) |
| `vocos` | Vocoder Vocos-mel-24khz (required by F5-TTS) |
| `silero_stress` | Silero Stress (word stress placement) |

---

## Tab 5 — ⚙️ Settings

### Background music

Upload `.wav` files to `sound/back/` and select one to mix under the narration.

### Exceptions dictionary

Add words with manual stress marks (e.g. `за́мок` vs `замо́к`) that override
the automatic stress model.

### Abbreviations

Add abbreviations that should be spelled out or pronounced in a specific way.

### Event sounds

Map text markers to `.wav` sound files in `sound/events/`. When the parser
encounters a marker in the text, the corresponding sound is inserted into the
audio at that position.

### Voice samples

Record or upload reference voice clips for F5-TTS voice cloning. Each sample
has a name and the text spoken in it.

### Storage paths

Customise the `data/` and `models/` directories. Changes are saved to
`user_settings.json`.

### 🔔 Completion sound

Choose a `.wav` file to play when TTS finishes.

### Find & remove repeated fragments

> Located in the **🔍 ANALYZE** tab, right below the Manual save / Auto-Clean
> buttons. Open the **🧹 Find & remove repeated fragments** accordion.

This tool removes or replaces repeated garbage fragments from the parsed XML
before you synthesise it. It works like a multi-replace panel without saving
anything to disk until you press **📝 Manual save (Ctrl+S)**.

**Step-by-step workflow:**

1. Select a fragment in the **File content** textarea with your mouse.
2. Click **📋 Copy selection** — the fragment is copied to the clipboard
   *and* added as a new line in the **Patterns** box.
3. Repeat for other fragments. You can also type or paste patterns manually.
4. Click **🔎 Count** to see how many times each fragment appears (no changes made).
5. Click **🧹 Remove all** to delete or replace all occurrences at once.
6. If you don't like the result, press **↩ Undo** (up to 10 steps).
7. When satisfied, press the existing **📝 Manual save (Ctrl+S)** to write
   the cleaned text back to disk.

**Pattern syntax (one per line):**

| Pattern | Meaning |
|---------|---------|
| `<p>отвэзетэ,</p>` | Delete every occurrence of this exact fragment |
| `hello => hi` | Replace `hello` with `hi` everywhere |

**Options:**

- **Whole line** — when checked, removes entire lines that contain the pattern
  (including the line break), instead of just the pattern itself.
- **Regex** — when checked, patterns are treated as regular expressions.
  If a pattern is not a valid regex, the app reports `❌ Bad regex` and
  skips it without changing the text.

**Undo:** stores up to 10 full copies of the text. Clicking **↩ Undo** restores
the previous version. The stack is lost when you switch files or reload.

**Important:** nothing is written to disk until you click **📝 Manual save**.
The cleanup report always shows `Not saved yet` as a reminder.

---

## 🎧 TTS — synthesis workflow

1. **Select a project** in the Project Manager tab.
2. Switch to the inner **🎧 TTS** tab.
3. The parsed XML sections are listed. Select which ones to synthesise.
4. Choose a **voice** from the dropdown.
5. Adjust **speed**, **noise level** and **pitch** if needed.
6. Click **Convert to speech**.

A progress bar shows:
- **Percentage** — overall completion
- **Elapsed** — time since start
- **Remaining** — estimated time left
- **Speed** — processing speed (proj/s or lines/s)

The output MP3 file appears in the project folder and is playable/downloadable
from the UI.

### How multilingual routing works

When you synthesise text that contains multiple languages, LECTA:

1. Splits the text into segments by detecting the script of each character
   (Cyrillic → Russian, Latin → English, Hebrew → Hebrew).
2. Routes each segment to the appropriate engine:
   - Hebrew → Edge cloud (if **⚡ Hebrew via cloud** is on, which is the default)
   - English → Edge cloud (if enabled) or Silero English v3 (local)
   - Russian → Edge cloud (if enabled) or the selected local model (Vosk/Silero/F5)
3. Synthesises all segments in parallel (cloud) or sequentially (local GPU).
4. Concatenates the audio with a short pause (0.3 s) between language switches.

You do not need to tag languages manually — the router detects them automatically.

### Batch mode

Click **⬇️ Batch TTS all projects** to process every project in `data/` in one run.

- A two-level progress bar shows overall batch progress and per-project progress.
- Each project's output MP3 is listed with a per-row download button.
- Select multiple rows to delete or download as a ZIP.
- The batch runs through the same parallel edge synth + shared cache as single TTS.

### Output location

Audio files are saved to `data/<project>/` as MP3. Partial results from an
interrupted run are saved as `<name>_PARTIAL.mp3`.

---

## 📚 Vocabulary Parser

The vocabulary tab extracts unique words from the current project's text and
translates them.

1. The project's parsed text is loaded automatically.
2. **SOURCE languages** — select which languages to extract words from
   (English, Hebrew, Russian).
3. **TARGET languages** — select which languages to translate into.
4. Click **Start** to begin extraction and translation.
5. A progress bar shows elapsed/remaining time and translation speed.
6. Results appear in a table: word → translations for each target language.
7. Click **Save vocabulary** to export the results.

> Translation uses Google Translate via the `deep-translator` library.
> Requires an internet connection.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+Enter** | Trigger the active tab's main button (Parse, TTS, Save FB2, Demo TTS) |
| **Ctrl+S** | Save XML (in the Analyze tab) |
| **Esc** | Stop / Abort current operation |
| **Enter** (in an input field) | Rename the selected project |
| **Delete** | Delete the selected file |
| **F2** | Rename the selected file |

---

## Known limitations

- **Interface language** is English only. The app does not bundle translations.
  If you prefer a different language, you can enable the browser's built-in
  page translation (right-click → "Translate to …") — the app does not block it.
  Note: with browser auto-translation enabled, Gradio's button labels may be
  mangled and extra quotation marks may appear. This is a side effect of the
  browser translator, not a bug in the application.
- **Hebrew** has no local TTS backend — only Edge cloud voices are available.
- **F5-TTS** is GPU-intensive. On CPU, generation is significantly slower.
- **Stress placement** (RuAccent / Silero Stress) applies to Russian text only.
- **Translation** in the vocabulary tab uses Google Translate and may produce
  imperfect results for idioms or context-dependent meanings.
- The Gemini API proxy (`start.py`) requires a Gemini CLI OAuth token and is
  not needed for core TTS functionality.
- **Undo** in the Find & remove fragments panel is limited to 10 steps and
  resets when you switch to a different file or reload the page.

---

## UI rollback

If the redesigned UI (commits starting with `feat(ui):`) does not suit your
workflow, you can revert to the legacy layout.

### Preview the old UI

```bash
git switch --detach ui-legacy-2026-07
```

To return to the current branch:

```bash
git switch -
```

### Permanently revert the redesign

Find the commit range of the redesign commits:

```bash
git log --oneline --grep="feat(ui):"
```

Then revert:

```bash
git revert --no-commit <first_commit>..<last_commit>
git commit -m "revert(ui): back to the legacy layout"
```

Replace `<first_commit>` and `<last_commit>` with the actual hashes from the
`git log` output.

> Note: The `ui-legacy-2026-07` tag and `ui-legacy` branch were created before
> the redesign and point to the last pre-redesign commit. They are not pushed
> to the remote.
