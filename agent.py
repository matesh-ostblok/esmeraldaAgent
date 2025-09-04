# agent.py
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
import sys
from pathlib import Path
# Force-load .env from this folder, overriding any preset env vars
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

from agents import Agent, Runner  # Agents SDK
from supabase import create_client, Client

from tools.searchLaw import searchLaw, usage_counters

# Supabase client was used for token accounting previously.
# Website now owns accounting; keep client unused to avoid breaking imports.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
sb: Client | None = None

## tools.searchLaw provides searchLaw tool and per-run embedding usage tracking

## Website is now responsible for persisting chat messages.

def _build_usage_record(
    uid: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    embedding_model: str | None = None,
    embedding_input_tokens: int = 0,
) -> Dict[str, Any]:
    """Prepare a structured usage record (no DB writes here)."""
    return {
        "uid": uid,
        "model": model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "embedding_model": embedding_model,
        "embedding_input_tokens": int(embedding_input_tokens or 0),
    }

## Topic generation moved to Website; no agent-side topic extraction.

## Memory is now passed in from the website (already trimmed/ordered).

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_TEMPLATE = (PROMPTS_DIR / "esmeralda_system_actual.md").read_text(encoding="utf-8").strip()

# --- Agent: používa len tool výstupy ---
esmeralda = Agent(
    name="Esmeralda",
    model="gpt-5-mini",
    instructions=SYSTEM_PROMPT_TEMPLATE,
    tools=[searchLaw],
)

# --- Optional: Agents SDK session-backed memory (server-side) ---
# Enabled by default. Set USE_AGENTS_SESSION=0 to disable.
USE_AGENTS_SESSION = os.environ.get("USE_AGENTS_SESSION", "1").strip() not in ("", "0", "false", "False")
AGENTS_SQLITE_PATH = os.environ.get("AGENTS_SQLITE_PATH", str((Path(__file__).parent / "agents_memory.sqlite3").resolve()))

_SESSION_AVAILABLE = False
try:
    # Import lazily; if not available, we fall back to website-provided memory
    from agents.memory.sqlite_session import SQLiteSession  # type: ignore
    _SESSION_AVAILABLE = True
except Exception:
    _SESSION_AVAILABLE = False

def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _get_session(uid: str):
    """Create or open a SQLite-backed session for given uid, if enabled/available."""
    if not (USE_AGENTS_SESSION and _SESSION_AVAILABLE and uid):
        return None
    db_path = Path(AGENTS_SQLITE_PATH)
    _ensure_parent(db_path)
    try:
        # session_id groups items per user/session; SQLiteSession handles schema internally
        return SQLiteSession(db_path=str(db_path), session_id=uid)
    except Exception as e:
        print(f"[Session] Failed to create SQLiteSession: {e}", file=sys.stderr)
        return None

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(uid: str, name: str, prompt: str, memory: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # Log session header to stderr so it isn't streamed
    print(f"[UID: {uid}] [User: {name}] -> {prompt}", file=sys.stderr)
    # Reset per-run embedding usage counters
    usage_counters["embedding_tokens"] = 0
    usage_counters["embedding_model"] = "text-embedding-3-small"
    # Determine memory strategy: Agents Session (server-side) vs website-supplied history
    session = _get_session(uid)
    use_session = session is not None

    if use_session:
        # With session, pass only the new user prompt; the runner will read/write history
        enriched_input = prompt
    else:
        # Použi pamäť dodanú z webu (max ~10 položiek, chronologicky)
        mem_rows = memory or []
        if mem_rows:
            memory_block = "\n".join([f"{r['role']}: {r['content']}" for r in mem_rows])
            enriched_input = f"[MEMORY]\n{memory_block}\n[/MEMORY]\n\n[USER QUESTION]\n{prompt}"
        else:
            enriched_input = prompt
    # Apply per-request templating (supports optional {name} in prompt template)
    esmeralda.instructions = SYSTEM_PROMPT_TEMPLATE.format(name=name)
    if use_session:
        result = Runner.run_streamed(esmeralda, input=enriched_input, session=session)
    else:
        result = Runner.run_streamed(esmeralda, input=enriched_input)
    buf = []
    usage = None
    async for ev in result.stream_events():
        # Streamujeme len textové delty a zachytíme completed event s usage
        from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            piece = ev.data.delta or ""
            buf.append(piece)
            print(piece, end="", flush=True)
        elif ev.type == "response.completed":
            try:
                usage = ev.data.response.output[0].usage
            except Exception as e:
                print(f"[Token Usage] Parse error: {e}", file=sys.stderr)
    print()  # newline
    assistant_text = "".join(buf)
    # Persisting messages for UI display is handled by the website.

    # Najprv skús usage z run contextu (Agents SDK ukladá usage tam po dokončení streamu)
    if not usage:
        try:
            ctx = getattr(result, "context_wrapper", None)
            if ctx and getattr(ctx, "usage", None):
                usage = ctx.usage
        except Exception as e:
            print(f"[Token Usage] Context usage parse error: {e}", file=sys.stderr)

    # Fallback – ak by context nemal usage, skús finálnu odpoveď
    # if not usage:
    #     try:
    #         final = await result.get_final_response()
    #         if final and getattr(final, "output", None):
    #             out0 = final.output[0]
    #             if hasattr(out0, "usage") and out0.usage:
    #                 usage = out0.usage
    #     except Exception as e:
    #         print(f"[Token Usage] Final response parse error: {e}", file=sys.stderr)

    # Uloženie usage (LLM + embeddingy). Aj keď LLM usage chýba, zaúčtujeme aspoň embeddingy.
    in_tok = 0
    out_tok = 0
    if usage:
        try:
            # Podpora dvoch konvencií názvov: input/output_tokens a prompt/completion_tokens
            in_tok = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
            out_tok = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
        except Exception as e:
            print(f"[Token Usage] Could not parse usage fields: {e}", file=sys.stderr)
    else:
        print("[Token Usage] Usage not available; will record embedding usage only if present.", file=sys.stderr)
    # Build usage record for the Website to persist
    usage_record = _build_usage_record(
        uid,
        esmeralda.model,
        in_tok,
        out_tok,
        embedding_model=usage_counters.get("embedding_model"),
        embedding_input_tokens=int(usage_counters.get("embedding_tokens", 0) or 0),
    )

    # Return usage_record so the FastAPI SSE layer can emit it as metadata.
    try:
        if session is not None:
            # Close underlying DB handle if provided by SQLiteSession
            close = getattr(session, "close", None)
            if callable(close):
                close()
    except Exception:
        pass
    return usage_record

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        uid = sys.argv[1]
        name = sys.argv[2]
        prompt = " ".join(sys.argv[3:])
        asyncio.run(run_once(uid, name, prompt, memory=[]))
    else:
        print("Použitie: python agent.py <uid> <name> <prompt>")
