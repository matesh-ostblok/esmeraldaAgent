# agent.py
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
import sys
load_dotenv()

from agents import Agent, Runner  # Agents SDK
from pathlib import Path
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

def _extract_topic(text: str, max_words: int = 5) -> str:
    """Unicode-safe 5-word topic summary derived from the user prompt.
    Keeps word characters (unicode) and spaces, removes punctuation, drops simple stopwords, returns up to N tokens.
    """
    if not text:
        return ""
    import re as _re
    cleaned = _re.sub(r"[\n\r\t]", " ", text.lower())
    cleaned = _re.sub(r"[^\w\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t]
    stop = {
        # Slovak only
        "a","aj","alebo","ale","sa","som","sme","ste","si","by","byť","ma","mi","mu","ti","to","je","sú","bol","bola","boli","bude","budú","na","v","vo","z","zo","do","od","pre","pod","nad","pri","o","u","k","ku","ako","že","ktorý","ktora","ktoré","ktorá","čo","kde","prečo","toto","táto","ten","tá","tí","tie","nej","svoj","svoje","svojho","svojeho",
    }
    kept = [t for t in tokens if t not in stop]
    if not kept:
        orig = [t for t in text.split() if t]
        return " ".join(orig[:max_words])
    return " ".join(kept[:max_words])

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

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(uid: str, name: str, prompt: str, memory: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # Log session header to stderr so it isn't streamed
    print(f"[UID: {uid}] [User: {name}] -> {prompt}", file=sys.stderr)
    # Reset per-run embedding usage counters
    usage_counters["embedding_tokens"] = 0
    usage_counters["embedding_model"] = "text-embedding-3-small"
    # Použi pamäť dodanú z webu (max ~10 položiek, chronologicky)
    mem_rows = memory or []
    if mem_rows:
        memory_block = "\n".join([f"{r['role']}: {r['content']}" for r in mem_rows])
        enriched_input = f"[MEMORY]\n{memory_block}\n[/MEMORY]\n\n[USER QUESTION]\n{prompt}"
    else:
        enriched_input = prompt
    # Apply per-request templating (supports optional {name} in prompt template)
    esmeralda.instructions = SYSTEM_PROMPT_TEMPLATE.format(name=name)
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
    # Persisting messages is handled by the website.

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
