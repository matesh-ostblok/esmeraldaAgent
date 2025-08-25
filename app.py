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