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
from memory.sqlite_store import (
    build_input as mem_build_input,
    record_user as mem_record_user,
    record_assistant as mem_record_assistant,
    use_memory as mem_use,
)

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

# Memory management is implemented in memory/sqlite_store.py

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(uid: str, name: str, prompt: str, memory: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # Log session header to stderr so it isn't streamed
    print(f"[UID: {uid}] [User: {name}] -> {prompt}", file=sys.stderr)
    # Reset per-run embedding usage counters
    usage_counters["embedding_tokens"] = 0
    usage_counters["embedding_model"] = "text-embedding-3-small"
    # Build enriched input using local SQLite memory if enabled; otherwise use fallback from website
    use_session = mem_use()
    enriched_input = mem_build_input(uid, prompt, fallback_history=(memory or []))
    # Apply per-request templating (supports optional {name} in prompt template)
    esmeralda.instructions = SYSTEM_PROMPT_TEMPLATE.format(name=name)
    # Record current user prompt before running to preserve timeline
    if use_session:
        try:
            mem_record_user(uid, prompt)
        except Exception:
            pass
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
    # Append assistant answer to local memory
    if use_session:
        try:
            mem_record_assistant(uid, assistant_text)
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
