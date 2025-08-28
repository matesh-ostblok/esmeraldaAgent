# app.py
import asyncio
import re
import json
from contextlib import redirect_stdout
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from agent import run_once

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

    # We drop the very first session/user log line printed by run_once.
    header_done = False
    header_buffer = ""

    async def _runner():
        writer = _QueueWriter(q)
        with redirect_stdout(writer):
            await run_once(session_id, name, prompt)
        await q.put("__RUN_DONE__")

    task = asyncio.create_task(_runner())

    # Tell clients we're ready
    yield b'data: {"ready": true}\n\n'

    # Emit the user prompt once at the beginning
    if prompt:
        yield ("data: " + json.dumps({"delta": prompt}, ensure_ascii=False) + "\n\n").encode("utf-8")

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=20)
            except asyncio.TimeoutError:
                # keep-alive comment for proxies during idle
                yield b": keepalive\n\n"
                continue

            if chunk == "__RUN_DONE__":
                # Flush any remaining header buffer on completion
                if not header_done and header_buffer:
                    # If the buffered text is the session/user line, drop it.
                    if re.match(r"^\s*\[Session:.*\]\s*\[User:.*\]\s*->", header_buffer):
                        header_done = True
                    else:
                        header_done = True
                        for tok in _tokenize_small(header_buffer):
                            yield ("data: " + json.dumps({"delta": tok}, ensure_ascii=False) + "\n\n").encode("utf-8")
                break

            if not header_done:
                header_buffer += chunk
                if "\n" in header_buffer:
                    first_line, rest = header_buffer.split("\n", 1)
                    if re.match(r"^\s*\[Session:.*\]\s*\[User:.*\]\s*->", first_line):
                        header_done = True
                        # Emit the remainder immediately in small token-like chunks
                        if rest:
                            for tok in _tokenize_small(rest):
                                yield ("data: " + json.dumps({"delta": tok}, ensure_ascii=False) + "\n\n").encode("utf-8")
                        header_buffer = ""
                    else:
                        # Not a header line; emit it and continue with rest
                        header_done = True
                        for tok in _tokenize_small(first_line + ("\n" + rest if rest else "")):
                            yield ("data: " + json.dumps({"delta": tok}, ensure_ascii=False) + "\n\n").encode("utf-8")
                        header_buffer = ""
                continue

            # Header already handled: emit chunk immediately in small token-like chunks
            if chunk:
                for tok in _tokenize_small(chunk):
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
