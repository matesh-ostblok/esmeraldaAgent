# app.py
import asyncio
import re
import json
from contextlib import redirect_stdout
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from agent import run_once, esmeralda

app = FastAPI()

def _tokenize_small(s: str, size: int = 3):
    """Yield small token-like chunks of size `size` codepoints from string `s`.
    Preserves order; does not drop characters. Simple and UTF-8 safe.
    """
    i = 0
    n = len(s)
    while i < n:
        yield s[i:i+size]
        i += size

class _QueueWriter:
    """File-like writer that pushes writes into an asyncio.Queue."""
    def __init__(self, q: asyncio.Queue[str]):
        self.q = q
    def write(self, s: str) -> int:
        if s:
            try:
                self.q.put_nowait(s)
            except Exception:
                pass
        return len(s)
    def flush(self) -> None:
        return None

async def sse_chat_stream(session_id: str, name: str, prompt: str):
    """
    Run agent.run_once(session_id, name, prompt) and stream its stdout as SSE.
    All token accounting (LLM + embeddings) is done inside agent.run_once.
    """
    q: asyncio.Queue[str] = asyncio.Queue()

    first_print_skipped = False  # drop the first session/user log line

    async def _runner():
        writer = _QueueWriter(q)
        with redirect_stdout(writer):
            await run_once(session_id, name, prompt)
        await q.put("__RUN_DONE__")

    task = asyncio.create_task(_runner())

    # Tell clients we're ready
    yield b'data: {"ready": true}\n\n'

    buffer = ""
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=20)
            except asyncio.TimeoutError:
                # keep-alive comment for proxies during idle
                yield b": keepalive\n\n"
                continue

            if chunk == "__RUN_DONE__":
                if buffer:
                    # If the last buffered text is the session/user line, drop it.
                    if not first_print_skipped and re.match(r"^\s*\[Session:.*\]\s*\[User:.*\]\s*->", buffer):
                        first_print_skipped = True
                    else:
                        for tok in _tokenize_small(buffer):
                            yield ("data: " + json.dumps({"delta": tok}, ensure_ascii=False) + "\n\n").encode("utf-8")
                    buffer = ""
                break

            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line:
                    continue
                # Drop the first log line that looks like: [Session: ...] [User: ...] -> ...
                if not first_print_skipped and re.match(r"^\s*\[Session:.*\]\s*\[User:.*\]\s*->", line):
                    first_print_skipped = True
                    continue
                # Emit in small token-like chunks (3 codepoints)
                for tok in _tokenize_small(line):
                    yield ("data: " + json.dumps({"delta": tok}, ensure_ascii=False) + "\n\n").encode("utf-8")

    finally:
        try:
            await asyncio.wait_for(task, timeout=5)
        except Exception:
            task.cancel()

    yield b'data: {"done": true}\n\n'

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

@app.get("/health")
def health():
    return {"who": "esmeralda-fastapi", "sse": True}

# Lokálne spustenie (v produkcii beží cez gunicorn/uvicorn worker)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)