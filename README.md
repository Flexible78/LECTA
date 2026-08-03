# LECTA - Russian audiobook studio (FB2 to speech)

> **Attribution.** This project builds on the open-source [fb2tts](https://gitverse.ru/diger/fb2tts)
> project by *diger*, which provides the original Gradio interface and the FB2 parsing and TTS
> pipeline. Everything listed under "What this repository adds" below was written by me on top of
> that base. Please respect the upstream author when reusing this code.

## What this repository adds

- **Batch synthesis** routed through a parallel edge-TTS path with a shared cache, so a whole book
  is voiced in one run instead of chapter by chapter.
- **A file index** that resolves paths across projects, which is what makes cross-project batch
  runs possible at all.
- **Interrupted runs survive**: a partial file keeps the model it used and the time it spent, and a
  missing XML is parsed automatically instead of failing the run.
- **Honest result packaging**: the ZIP is built by scanning the output directory after completion,
  so every produced MP3 ends up inside it.
- **UI work**: per-row download, multiselect with bulk delete, checkbox state that survives a
  refresh, and a completion sound that does not block the player from loading the final MP3.
- **Robustness fixes**: the audio player is guarded against empty and directory paths that raised
  PermissionError, and `is_file()` is checked before a download is offered.

## Stack

Python 3.11, Gradio 6, PyTorch 2.6 with torchaudio, Vosk TTS / Silero / F5-TTS backends,
ONNX Runtime, RUAccent for stress placement, lxml and pymorphy3 for text processing,
FastAPI and uvicorn for the local provider proxy, SQLite for project state.
The UI binds to 127.0.0.1 by default and no public tunnel is opened unless `--share` is passed.

---

- [ ] fb2tts
Для преобразования текста в речь использутся [Vosk TTS](https://github.com/alphacep/vosk-tts), [Silero](https://github.com/snakers4/silero-models) и [F5-TTS](https://github.com/SWivid/F5-TTS). Для F5-TTS используются модели от [Misha24-10](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/tree/main/F5TTS_v1_Base_v4_winter) и [ESpeech](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2/tree/main)
Для расстановки ударений можно использовать [Ruaccent](https://gitverse.ru/Den4ikAI/ruaccent) или [Silero Stress](https://github.com/snakers4/silero-stress)

### Установка и использование

<details>
<summary><b>Для Windows</b></summary>

#### ✅ Требования

- Windows 10 или 11 (с установленным `winget`)
- Интернет-подключение

> `winget` входит в состав Windows 10/11 по умолчанию (обновления 21H1 и новее)

#### 1. Установка необходимых компонентов

Откройте **командную строку (CMD) или PowerShell от имени администратора** и выполните следующую команду:

```cmd
winget install Git.Git Python.Python.3.11
```

После установки перезапустите командную строку, чтобы обновить PATH.  
Проверьте установку:

```cmd
python -V
git --version
```

#### 2. Скачайте проект

```cmd
git clone https://gitverse.ru/diger/fb2tts.git
cd fb2tts
```

#### 3. Установите зависимости Python

```cmd
pip install -r requirements.txt
```

#### 4. Запустите программу

```cmd
python app.py
```

Перейдите в браузере по адресу: http://localhost:7860
</details>
<details>
<summary><b>Для Linux (Ubuntu/Debian)</b></summary>

#### 1. Обновите пакеты и установите зависимости:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip git ffmpeg -y
```

#### 2. Клонируйте репозиторий:

```bash
git clone https://github.com/diger/fb2tts.git
cd fb2tts
```

#### 3. Создайте виртуальное окружение и установите зависимости:  

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Запустите сервер:  

```bash
python3 app.py
```

Перейдите в браузере по адресу: http://localhost:7860

</details>

### Слушаем примеры преобразования  
На стартовой странице можно прослушать примеры генерации голоса.  
![DemoTTS](https://raw.githubusercontent.com/diger/diger.github.io/refs/heads/master/screenshots/demotts.png)

### Задаём обложку аудиокниги  
По-умолчанию выбирается обложка из fb2 файла. Можно установить свою картинку и создать подпись на ней.  
![Back](https://raw.githubusercontent.com/diger/diger.github.io/refs/heads/master/screenshots/back.png)

### Парсим fb2 файл  
Исходный fb2 файл разбивается либо по главам, либо по размеру. Также есть возможность задать произвольный тег.
Можно поправить обработанные файлы, например изменить ударение в словах.  
![Parse](https://raw.githubusercontent.com/diger/diger.github.io/refs/heads/master/screenshots/parse.png)

### Примеры озвученных глав из fb2  
[![Здесь примеры озвученных глав из fb2](https://github.com/diger/fb2tts/blob/main/libs/cover.jpg?raw=true&s=128)](https://samply.app/p/TqhqdbpCC1M30MzkzYmI?si=LF45p07JbyPSMXugaq4ShAI3hg92)