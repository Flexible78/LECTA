import os
import re
import json
import uuid
import asyncio
import logging
import httpx
from fastapi import FastAPI, Request
import uvicorn
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest

app = FastAPI()

# ── Настройки ──────────────────────────────────────────────────────────────────
# Все пути определяются автоматически из папки Gemini CLI текущего пользователя.
# Менять вручную не нужно.

GEMINI_DIR = os.path.expanduser("~/.gemini")          # ~/.gemini (работает на Win/Mac/Linux)
TOKEN_FILE_PATH = os.path.join(GEMINI_DIR, "oauth_creds.json")
GEMINI_CLI_VERSION = "0.40.0"                          # версия для User-Agent
HOST = "127.0.0.1"
PORT = int(os.getenv("LECTA_PROXY_PORT", "8080"))

# OAuth client ID/Secret Gemini CLI (одинаковые для всех пользователей,
# зашиты в бандле Gemini CLI — извлечены из chunk-3OSQ5US4.js)
def _load_gemini_cli_oauth():
    """Try to read CLIENT_ID/SECRET from Gemini CLI bundle automatically."""
    try:
        import glob as _glob
        patterns = [
            # Windows (npm global)
            os.path.join(os.environ.get("APPDATA", ""), "npm/node_modules/@google/gemini-cli/bundle/chunk-3OSQ5US4.js"),
            os.path.join(os.environ.get("APPDATA", ""), r"npm\node_modules\@google\gemini-cli\bundle\chunk-3OSQ5US4.js"),
            # macOS/Linux (npm global)
            "/usr/local/lib/node_modules/@google/gemini-cli/bundle/chunk-3OSQ5US4.js",
            "/usr/lib/node_modules/@google/gemini-cli/bundle/chunk-3OSQ5US4.js",
            # nvm
            os.path.expanduser("~/.nvm/versions/node/*/lib/node_modules/@google/gemini-cli/bundle/chunk-3OSQ5US4.js"),
        ]
        for pattern in patterns:
            for path in _glob.glob(pattern):
                content = open(path, encoding="utf-8", errors="ignore").read()
                cid = re.search(r'OAUTH_CLIENT_ID\s*=\s*"([^"]+)"', content)
                sec = re.search(r'OAUTH_CLIENT_SECRET\s*=\s*"([^"]+)"', content)
                if cid and sec:
                    return cid.group(1), sec.group(1)
    except Exception:
        pass
    return None, None

_auto_id, _auto_secret = _load_gemini_cli_oauth()
if not _auto_id or not _auto_secret:
    raise RuntimeError(
        "Не удалось найти CLIENT_ID/SECRET в бандле Gemini CLI. "
        "Убедитесь что Gemini CLI установлен (npm install -g @google/gemini-cli)."
    )
CLIENT_ID     = _auto_id
CLIENT_SECRET = _auto_secret

CODE_ASSIST_BASE = "https://cloudcode-pa.googleapis.com/v1internal"
# ──────────────────────────────────────────────────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(_dir, "proxy.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("proxy")

project_id_cache = None
project_id_lock = asyncio.Lock()
result_cache: dict = {}  # dedup cache
thought_sig_store: dict = {}  # tool_use_id -> thoughtSignature (in-memory)
THOUGHT_SIG_FILE = os.path.join(_dir, "thought_sigs.json")

def load_thought_sigs():
    global thought_sig_store
    try:
        with open(THOUGHT_SIG_FILE, "r", encoding="utf-8") as f:
            thought_sig_store = json.load(f)
    except Exception:
        thought_sig_store = {}

def save_thought_sigs():
    with open(THOUGHT_SIG_FILE, "w", encoding="utf-8") as f:
        json.dump(thought_sig_store, f)

def get_credentials():
    with open(TOKEN_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    creds = Credentials(
        token=data.get("access_token"),
        refresh_token=data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=data.get("scope", "").split(),
    )
    if not creds.valid:
        log.info("Обновляем токен...")
        creds.refresh(GoogleRequest())
        data["access_token"] = creds.token
        with open(TOKEN_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Токен обновлён")
    return creds

async def get_project_id(token: str) -> str:
    global project_id_cache
    if project_id_cache:
        return project_id_cache
    async with project_id_lock:
        if project_id_cache:
            return project_id_cache
        log.info("Получаем projectId через loadCodeAssist...")
        for attempt in range(2):
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{CODE_ASSIST_BASE}:loadCodeAssist",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"metadata": {"ideType": "IDE_UNSPECIFIED", "platform": "PLATFORM_UNSPECIFIED", "pluginType": "GEMINI"}},
                    timeout=30.0,
                )
                log.debug("loadCodeAssist status=%s body=%s", resp.status_code, resp.text[:500])
                if resp.status_code == 401 and attempt == 0:
                    log.warning("401 в loadCodeAssist, обновляем токен...")
                    creds = get_credentials()
                    creds.refresh(GoogleRequest())
                    token = creds.token
                    with open(TOKEN_FILE_PATH, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    d["access_token"] = token
                    with open(TOKEN_FILE_PATH, "w", encoding="utf-8") as f:
                        json.dump(d, f, indent=2)
                    continue
                resp.raise_for_status()
                result = resp.json()
                project_id_cache = result.get("cloudaicompanionProject") or result.get("projectId") or ""
                log.info("projectId: %s", project_id_cache)
                return project_id_cache
        raise Exception("Не удалось получить projectId после обновления токена")

# ── Format converters ──────────────────────────────────────────────────────────

ALLOWED_SCHEMA_KEYS = {"type", "properties", "required", "description", "items", "enum"}

def clean_schema(schema: dict) -> dict:
    """Recursively remove JSON Schema fields unsupported by Gemini."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    for k, v in schema.items():
        if k not in ALLOWED_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            result[k] = {pk: clean_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            result[k] = clean_schema(v)
        else:
            result[k] = v
    return result

def anthropic_tools_to_gemini(tools: list) -> list:
    """Convert Anthropic tool definitions to Gemini functionDeclarations."""
    result = []
    for t in tools:
        schema = clean_schema(t.get("input_schema", {}))
        result.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": schema or {"type": "object", "properties": {}},
        })
    return result

def anthropic_messages_to_gemini(messages: list) -> list:
    """Convert Anthropic messages array to Gemini contents array."""
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        content = msg.get("content", "")

        if isinstance(content, str):
            # Extract text from system-reminder tags if present
            text = re.sub(r"<system-reminder>.*?</system-reminder>", "", content, flags=re.DOTALL).strip()
            if text:
                contents.append({"role": role, "parts": [{"text": text}]})
            continue

        parts = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text = re.sub(r"<system-reminder>.*?</system-reminder>", "", block.get("text", ""), flags=re.DOTALL).strip()
                if text:
                    parts.append({"text": text})
            elif btype == "tool_use":
                # Anthropic tool_use → Gemini functionCall
                sig = thought_sig_store.get(block.get("id", ""))
                log.debug("tool_use '%s' id=%s sig=%s", block.get("name"), block.get("id"), "YES" if sig else "NO")
                if sig:
                    parts.append({"thoughtSignature": sig, "functionCall": {"name": block["name"], "args": block.get("input", {})}})
                else:
                    # No signature - represent as text to avoid 400 error
                    parts.append({"text": f"[Called tool: {block['name']}]"})
            elif btype == "tool_result":
                # Anthropic tool_result → Gemini functionResponse
                tool_id = block.get("tool_use_id", "")
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    tool_content = " ".join(b.get("text", "") for b in tool_content if b.get("type") == "text")
                if thought_sig_store.get(tool_id):
                    parts.append({
                        "functionResponse": {
                            "name": tool_id,
                            "response": {"output": tool_content},
                        }
                    })
                else:
                    # No signature for this tool - represent as text
                    parts.append({"text": f"[Tool result: {tool_content}]"})

        if parts:
            contents.append({"role": role, "parts": parts})

    return contents

def gemini_response_to_anthropic(chunks: list) -> dict:
    """
    Convert Gemini streamGenerateContent response chunks to Anthropic message format.
    Returns dict with keys: content (list), stop_reason, _thought_signatures (internal)
    """
    text_parts = []
    tool_calls = []
    finish_reason = "end_turn"
    thought_signatures = {}  # tool_name -> thoughtSignature

    # Collect thoughtSignature from parts before functionCall
    last_thought_sig = None

    for chunk in chunks:
        inner = chunk.get("response", chunk)
        candidates = inner.get("candidates", [])
        if not candidates:
            continue
        candidate = candidates[0]
        finish = candidate.get("finishReason", "")

        if finish == "MALFORMED_FUNCTION_CALL":
            msg = candidate.get("finishMessage", "")
            q = re.search(r'question:"([^"]+)"', msg)
            if q:
                text_parts.append(q.group(1))
            elif msg:
                text_parts.append(msg)
            # Don't continue - still process parts if any

        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_id = f"toolu_{uuid.uuid4().hex[:16]}"
                tool_calls.append({
                    "id": tool_id,
                    "name": fc.get("name", ""),
                    "args": fc.get("args", {}),
                    "thought_signature": part.get("thoughtSignature"),
                })
                finish_reason = "tool_use"
            elif "text" in part:
                text_parts.append(part["text"])

    content = []
    if text_parts:
        content.append({"type": "text", "text": "".join(text_parts)})
    for tc in tool_calls:
        content.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": tc["name"],
            "input": tc["args"],
        })

    return {
        "content": content,
        "stop_reason": finish_reason,
        "_thought_signatures": {tc["id"]: tc["thought_signature"] for tc in tool_calls if tc["thought_signature"]},
    }

# ── HTTP helpers ───────────────────────────────────────────────────────────────

class ModelCapacityError(Exception):
    pass

FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"]

async def gemini_post(token: str, project_id: str, gemini_model: str, payload: dict, fail_fast: bool = False) -> list:
    """Send request to Gemini and return parsed chunks list. Retries on 429."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"GeminiCLI/{GEMINI_CLI_VERSION}/{gemini_model} (win32; x64; cli)",
    }
    async with httpx.AsyncClient() as client:
        for attempt in range(5):
            log.debug("Отправляем запрос model=%s (попытка %d/5)...", gemini_model, attempt + 1)
            response = await client.post(
                f"{CODE_ASSIST_BASE}:streamGenerateContent",
                headers=headers, json=payload, timeout=120.0
            )
            log.debug("<<< status=%d body=%s", response.status_code, response.text[:500])
            if response.status_code == 401:
                log.warning("401, принудительно обновляем токен...")
                with open(TOKEN_FILE_PATH, "r", encoding="utf-8") as f:
                    cred_data = json.load(f)
                creds = Credentials(
                    token=None,  # force refresh
                    refresh_token=cred_data.get("refresh_token"),
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=CLIENT_ID,
                    client_secret=CLIENT_SECRET,
                )
                creds.refresh(GoogleRequest())
                cred_data["access_token"] = creds.token
                with open(TOKEN_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cred_data, f, indent=2)
                headers["Authorization"] = f"Bearer {creds.token}"
                log.info("Токен обновлён, повторяем запрос...")
                continue
            if response.status_code == 429:
                if "MODEL_CAPACITY_EXHAUSTED" in response.text:
                    raise Exception(f"MODEL_CAPACITY_EXHAUSTED for {gemini_model}")
                if fail_fast:
                    raise Exception(f"429 for {gemini_model}")
                wait = 60
                m = re.search(r"reset after (\d+)s", response.text)
                if m:
                    wait = int(m.group(1)) + 1
                log.warning("429, ждём %dс...", wait)
                await asyncio.sleep(wait)
                continue
            log.debug("<<< FULL BODY: %s", response.text)
            response.raise_for_status()
            chunks = json.loads(response.text)
            if not isinstance(chunks, list):
                chunks = [chunks]
            return chunks
    raise Exception("Превышено число попыток после 429")

# ── Models ─────────────────────────────────────────────────────────────────────

GEMINI_MODELS = {
    "auto":                   ("gemini-3-flash-preview",        "Авто",                  True),
    "gemini-2.5-pro":         ("gemini-2.5-pro",                "Gemini 2.5 Pro",         True),
    "gemini-2.5-flash":       ("gemini-2.5-flash",              "Gemini 2.5 Flash",       True),
    "gemini-2.5-flash-lite":  ("gemini-2.5-flash-lite",         "Gemini 2.5 Flash Lite",  True),
    "gemini-3-pro-preview":   ("gemini-3-pro-preview",          "Gemini 3 Pro Preview",   True),
    "gemini-3.1-pro-preview": ("gemini-3.1-pro-preview",        "Gemini 3.1 Pro Preview", True),
    "gemini-3-flash-preview": ("gemini-3-flash-preview",        "Gemini 3 Flash Preview", True),
    "gemini-3.1-flash-lite":  ("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite",  True),
}

AUTO_FALLBACK = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"]

@app.get("/v1/models")
async def list_models():
    models = []
    for alias, (_, display, supports1m) in GEMINI_MODELS.items():
        entry = {"type": "model", "id": alias, "display_name": display, "created_at": "2025-01-01T00:00:00Z"}
        if supports1m:
            entry["supports1m"] = True
        models.append(entry)
    return {"data": models, "has_more": False, "first_id": models[0]["id"], "last_id": models[-1]["id"]}

# ── Main handler ───────────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def handle_claude_request(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    tools = data.get("tools", [])
    log.debug(">>> model=%s messages=%d tools=%d", data.get("model"), len(messages), len(tools))
    log.debug(">>> MESSAGES: %s", json.dumps(messages, ensure_ascii=False)[:8000])
    # log.debug(">>> TOOLS: %s", json.dumps(tools, ensure_ascii=False)[:2000])
    # log.debug(">>> SYSTEM FULL: %s", json.dumps(data.get("system", ""), ensure_ascii=False)[:3000])

    # Build system prompt
    system = data.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
    # Remove billing header noise to reduce token count
    system = re.sub(r"x-anthropic-billing-header:[^\n]+\n?", "", system).strip()
    system_text = (system + "\nВсегда отвечай на русском языке.").strip()

    creds = get_credentials()
    token = creds.token
    project_id = await get_project_id(token)

    requested_model = data.get("model", "gemini-2.5-flash")
    gemini_model, _, _ = GEMINI_MODELS.get(requested_model, ("gemini-2.5-flash", "", False))
    log.info("Модель: %s -> %s", requested_model, gemini_model)

    # Convert messages history - keep only last 20 messages to limit token count
    # (same approach as Gemini CLI's truncateHistoryToBudget)
    recent_messages = messages[-20:] if len(messages) > 20 else messages
    contents = anthropic_messages_to_gemini(recent_messages)

    # Build payload
    payload: dict = {
        "model": gemini_model,
        "project": project_id,
        "request": {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
        }
    }

    # Add tools if present
    if tools:
        gemini_tools = anthropic_tools_to_gemini(tools)
        payload["request"]["tools"] = [{"functionDeclarations": gemini_tools}]
        log.debug("Передаём %d инструментов в Gemini", len(gemini_tools))

    # Dedup cache key based on last message
    last_msg_key = json.dumps(messages[-1] if messages else {}, ensure_ascii=False, sort_keys=True)
    dedup_key = f"{gemini_model}:{last_msg_key}"

    if dedup_key in result_cache:
        cached, ts = result_cache[dedup_key]
        if asyncio.get_event_loop().time() - ts < 5:
            log.info("Возвращаем кэшированный ответ")
            return cached
        del result_cache[dedup_key]

    try:
        if requested_model == "auto":
            chunks = None
            for try_model in AUTO_FALLBACK:
                payload["model"] = try_model
                try:
                    chunks = await gemini_post(token, project_id, try_model, payload, fail_fast=True)
                    break
                except Exception as e:
                    if "MODEL_CAPACITY_EXHAUSTED" in str(e) or "429" in str(e):
                        log.warning("429/capacity для %s, пробуем следующую...", try_model)
                        continue
                    raise
            if chunks is None:
                raise Exception("Все модели недоступны")
        else:
            chunks = await gemini_post(token, project_id, gemini_model, payload)
        result = gemini_response_to_anthropic(chunks)
        # Save thought signatures to file for persistence across restarts
        for tool_id, sig in result.get("_thought_signatures", {}).items():
            thought_sig_store[tool_id] = sig
        save_thought_sigs()
        log.info("stop_reason=%s content_blocks=%d", result["stop_reason"], len(result["content"]))
        # Claude app requires at least one content block
        if not result["content"]:
            log.debug("Пустой ответ от Gemini, добавляем placeholder")
            result["content"] = [{"type": "text", "text": "✓"}]
        if result["content"]:
            first = result["content"][0]
            if first["type"] == "text":
                log.info("Ответ: %s", first["text"][:200])
            else:
                log.info("tool_use: %s args=%s", first.get("name"), str(first.get("input", {}))[:200])
    except Exception as e:
        log.error("Ошибка запроса: %s", e)
        result = {"content": [{"type": "text", "text": f"Ошибка: {e}"}], "stop_reason": "end_turn"}

    response_body = {
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": result["content"],
        "stop_reason": result["stop_reason"],
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }
    result_cache[dedup_key] = (response_body, asyncio.get_event_loop().time())
    return response_body

if __name__ == "__main__":
    load_thought_sigs()
    log.info("Токен: %s", TOKEN_FILE_PATH)
    log.info("Лог: %s", LOG_FILE)
    log.info("CLIENT_ID: %s", CLIENT_ID[:30] + "...")
    log.info("Запущен на http://%s:%d", HOST, PORT)
    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except OSError as e:
        raise RuntimeError(
            f"Failed to bind port {PORT}. The port may already be in use. "
            f"Set LECTA_PROXY_PORT to a free port (e.g. LECTA_PROXY_PORT=8081) and restart. "
            f"Original error: {e}"
        ) from e
