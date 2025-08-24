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