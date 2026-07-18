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