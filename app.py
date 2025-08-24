# app.py
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from agent import run_once, esmeralda

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    """
    API endpoint na zavolanie agenta.
    Očakáva JSON:
    {
        "session_id": "uuid",
        "name": "Matej",
        "prompt": "Otázka používateľa"
    }
    """
    data = await request.json()
    session_id = data.get("session_id")
    name = data.get("name", "User")
    prompt = data.get("prompt")

    if not session_id or not prompt:
        return JSONResponse({"error": "session_id a prompt sú povinné"}, status_code=400)

    # Spusti agenta (stream -> výsledok zbiera run_once)
    await run_once(session_id, name, prompt)

    return {"status": "ok", "session_id": session_id}

# lokálny run (Render použije gunicorn/uvicorn)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
# app.py — FastAPI server s plnou funkcionalitou (streamovanie, účtovanie tokenov, tooly)
from typing import AsyncGenerator, Any, Optional, Dict, List
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Importujeme len helpery a objekty, agent.py NEMENÍME
from agent import (
    esmeralda,
    Runner,
    fetch_memory,
    save_message,
    save_token_usage,
    usage_counters,
)

# Niektoré verzie Agents SDK posielajú tento typ v stream udalostiach
try:
    from openai.types.responses import ResponseTextDeltaEvent  # type: ignore
except Exception:  # fallback, aby import nepovolil pád
    class ResponseTextDeltaEvent:  # type: ignore
        delta: str

app = FastAPI()

# ----------------- Pomocné funkcie (čisto lokálne v app.py) -----------------
def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def _get_field(obj: Any, *names: str) -> Optional[Any]:
    """Skús prečítať atribút alebo dict kľúč podľa viacerých možných názvov."""
    for n in names:
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
        try:
            if isinstance(obj, dict) and n in obj:
                return obj[n]
        except Exception:
            pass
    return None

def extract_usage_tokens(usage_obj: Any) -> tuple[int, int, int]:
    """
    Vráti (input_tokens, output_tokens, total_tokens) z rôznych tvarov usage.
    Podporuje:
      - input_tokens / output_tokens / total_tokens
      - prompt_tokens / completion_tokens / total_tokens
    """
    if usage_obj is None:
        return 0, 0, 0
    in_tok = _get_field(usage_obj, "input_tokens", "prompt_tokens")
    out_tok = _get_field(usage_obj, "output_tokens", "completion_tokens")
    tot_tok = _get_field(usage_obj, "total_tokens")
    in_tok_i = _as_int(in_tok, 0)
    out_tok_i = _as_int(out_tok, 0)
    tot_tok_i = _as_int(tot_tok, 0)
    if tot_tok_i == 0 and (in_tok_i or out_tok_i):
        tot_tok_i = in_tok_i + out_tok_i
    return in_tok_i, out_tok_i, tot_tok_i

def _enrich_input(session_id: str, prompt: str) -> str:
    """Zostaví prompt obohatený o pamäť konverzácie (rovnako ako v run_once)."""
    mem_rows = fetch_memory(session_id, limit=10)
    if mem_rows:
        memory_block = "\n".join([f"{r['role']}: {r['content']}" for r in mem_rows])
        return f"[MEMORY]\n{memory_block}\n[/MEMORY]\n\n[USER QUESTION]\n{prompt}"
    return prompt

# ----------------- Streamovací generátor (SSE) -----------------
async def sse_chat_stream(session_id: str, name: str, prompt: str) -> AsyncGenerator[bytes, None]:
    """
    Funkčne ekvivalentné k agent.run_once, len ako SSE stream pre FastAPI.
    - používa Runner.run_streamed (tooly a agent logika ostáva v agent.py),
    - streamuje tokeny po častiach,
    - na konci účtuje LLM aj embedding tokeny do Supabase.
    """
    # Reset per‑run embedding usage (agent.embed túto metriku zvyšuje)
    usage_counters["embedding_tokens"] = 0
    usage_counters["embedding_model"] = usage_counters.get("embedding_model", "text-embedding-3-small")

    # Ulož prichádzajúcu user správu
    save_message(session_id, "user", prompt)

    # Priprav „pamäťou“ obohatený vstup (rovnako ako v run_once)
    enriched_input = _enrich_input(session_id, prompt)

    # Spusti agenta v stream režime
    result = Runner.run_streamed(esmeralda, input=enriched_input)

    # Buffrovací string a usage placeholder
    buf: List[str] = []
    usage = None

    # Pošli otvárací signál (užitočné pri debugovaní)
    yield b'data: {"ready": true}\n\n'

    # Čítaj udalosti a streamuj textové delty
    async for ev in result.stream_events():
        if ev.type == "raw_response_event" and isinstance(getattr(ev, "data", None), ResponseTextDeltaEvent):
            piece = getattr(ev.data, "delta", "") or ""
            if piece:
                buf.append(piece)
                # SSE: každý kus musí byť ukončený prázdnym riadkom
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n".encode("utf-8")
        elif ev.type == "response.completed":
            # pokus získať usage priamo z eventu
            try:
                data_obj = getattr(ev, "data", None)
                if data_obj and hasattr(data_obj, "response"):
                    resp = data_obj.response
                    if getattr(resp, "output", None):
                        usage = resp.output[0].usage
            except Exception:
                usage = None

    # Zlož finálny text a ulož asistenta
    full_text = "".join(buf).strip()
    if full_text:
        save_message(session_id, "assistant", full_text)

    # Ak usage nie je, získaj z finálnej odpovede
    if usage is None:
        try:
            final = await result.get_final_response()
            if final and getattr(final, "output", None):
                out0 = final.output[0]
                if hasattr(out0, "usage") and out0.usage:
                    usage = out0.usage
        except Exception:
            usage = None

    # Zapíš usage (LLM + embedding) — vždy zapisujeme aj embedder
    try:
        in_tok, out_tok, _ = extract_usage_tokens(usage)
        save_token_usage(
            session_id,
            esmeralda.model,
            in_tok,
            out_tok,
            embedding_model=usage_counters.get("embedding_model"),
            embedding_input_tokens=_as_int(usage_counters.get("embedding_tokens", 0), 0),
        )
    except Exception as e:
        # Pošli debug udalosť do SSE, nech vieme, prečo sa usage nepodarilo uložiť
        yield f"data: {json.dumps({'usage_error': str(e)})}\n\n".encode("utf-8")

    # Ukončovací marker
    yield b'data: {"done": true}\n\n'

# ----------------- FastAPI endpointy -----------------
@app.post("/chat")
async def chat(request: Request):
    """
    SSE endpoint, ktorý vykoná to isté čo agent.run_once, ale ako HTTP stream.
    Telo:
    {
        "session_id": "uuid",
        "name": "Matej",
        "prompt": "Otázka používateľa"
    }
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    session_id = data.get("session_id")
    name = data.get("name", "User")
    prompt = data.get("prompt")

    if not session_id or not prompt:
        return JSONResponse({"error": "session_id a prompt sú povinné"}, status_code=400)

    return StreamingResponse(
        sse_chat_stream(session_id, name, prompt),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )

@app.post("/chat_json")
async def chat_json(request: Request):
    """
    Ne‑streamujúci JSON endpoint pre klientov, ktorí nevedia SSE (n8n HTTP Request).
    Funkčne to isté: tooly, agent, token accounting; len vrátime celé telo naraz.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    session_id = data.get("session_id")
    name = data.get("name", "User")
    prompt = data.get("prompt")

    if not session_id or not prompt:
        return JSONResponse({"error": "session_id a prompt sú povinné"}, status_code=400)

    # Reset embed usage pre tento beh (agent.embed bude zvyšovať counter)
    usage_counters["embedding_tokens"] = 0
    usage_counters["embedding_model"] = usage_counters.get("embedding_model", "text-embedding-3-small")

    save_message(session_id, "user", prompt)
    enriched_input = _enrich_input(session_id, prompt)

    result = Runner.run_streamed(esmeralda, input=enriched_input)
    buf: List[str] = []
    usage = None

    async for ev in result.stream_events():
        if ev.type == "raw_response_event" and isinstance(getattr(ev, "data", None), ResponseTextDeltaEvent):
            piece = getattr(ev.data, "delta", "") or ""
            if piece:
                buf.append(piece)
        elif ev.type == "response.completed":
            try:
                data_obj = getattr(ev, "data", None)
                if data_obj and hasattr(data_obj, "response"):
                    resp = data_obj.response
                    if getattr(resp, "output", None):
                        usage = resp.output[0].usage
            except Exception:
                usage = None

    full_text = "".join(buf).strip()
    if full_text:
        save_message(session_id, "assistant", full_text)

    # fallback usage z final response
    if usage is None:
        try:
            final = await result.get_final_response()
            if final and getattr(final, "output", None):
                out0 = final.output[0]
                if hasattr(out0, "usage") and out0.usage:
                    usage = out0.usage
        except Exception:
            usage = None

    # Zapíš usage (LLM + embedding) — aj keď LLM usage chýba, embedder sa uloží
    in_tok, out_tok, _ = extract_usage_tokens(usage)
    try:
        save_token_usage(
            session_id,
            esmeralda.model,
            in_tok,
            out_tok,
            embedding_model=usage_counters.get("embedding_model"),
            embedding_input_tokens=_as_int(usage_counters.get("embedding_tokens", 0), 0),
        )
    except Exception:
        pass

    return JSONResponse({
        "session_id": session_id,
        "text": full_text,
        "done": True,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "embedding_model": usage_counters.get("embedding_model"),
            "embedding_input_tokens": _as_int(usage_counters.get("embedding_tokens", 0), 0),
        }
    })

@app.get("/health")
def health():
    return {"who": "esmeralda-fastapi", "sse": True}

# Lokálne spustenie (v produkcii beží cez gunicorn/uvicorn worker)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)