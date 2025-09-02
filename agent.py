# agent.py
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
import sys
load_dotenv()

from agents import Agent, Runner  # Agents SDK
from supabase import create_client, Client

from tools.searchLaw import searchLaw, usage_counters

# Supabase klient
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

## tools.searchLaw provides searchLaw tool and per-run embedding usage tracking

def save_message(uid: str, role: str, content: str) -> None:
    """
    Uloží správu do public.chatMessages v Supabase.
    Očakáva role ∈ {"user","assistant"}.
    """
    try:
        sb.table("chatMessages").insert({
            "uid": uid,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        # Nezastavuj beh agenta kvôli logovaniu (loguj na stderr)
        print(f"[Supabase] Insert error: {e}", file=sys.stderr)

def save_token_usage(
    uid: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    embedding_model: str | None = None,
    embedding_input_tokens: int = 0,
    topic: str,
) -> None:
    """
    Uloží token usage do public.tokenUsage v Supabase vrátane embeddingov.
    """
    payload = {
        "uid": uid,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "embedding_model": embedding_model,
        "embedding_input_tokens": int(embedding_input_tokens or 0),
        "topic": topic or "",
    }
    try:
        sb.table("tokenUsage").insert(payload).execute()
    except Exception as e:
        print(f"[Supabase] Token usage insert error: {e}", file=sys.stderr)

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

def fetch_memory(uid: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Načíta posledných `limit` správ zo session a vráti ich v chronologickom poradí (najstaršia -> najnovšia).
    """
    try:
        resp = sb.table("chatMessages")\
            .select("role,content,created_at")\
            .eq("uid", uid)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        rows = (getattr(resp, "data", None) or [])
        # otoč na chronologické poradie
        rows.reverse()
        return rows
    except Exception as e:
        print(f"[Supabase] Fetch memory error: {e}", file=sys.stderr)
        return []

# --- Agent: používa len tool výstupy ---
esmeralda = Agent(
    name="Esmeralda",
    model="gpt-5-mini",
    instructions=(
        "Si právna asistentka pre SR. Na vyhľdávanie v právnych textoch môžeš použiť nástroj searchLaw."
        "Odpovedaj v konverzčnom štýle, nedávaj rady, iba odporúčania ak treba. Nepoužívaj odrážky ani číslovanie."
        "Ak uvádzaš referenciu na použitý text, použi payload z qdrantu metadata.regulation."
        "Otázku používateľa rozlož semanticky na menšie frázy (2–7 slov), ktoré jednotlivo posielaj do searchLaw."
        "Pri práci s výsledkami nástroja searchLaw vyberaj a zoradzuj dokumenty podľa týchto pravidiel:"
        "- Primárne zoradenie: metadata.validFrom zostupne (najnovší ako prvý)."
        "- Sekundárne zoradenie: score zostupne."
        "- Ak metadata.validFrom chýba, použi náhradu v poradí: metadata.announcedOn, potom metadata.approvedOn; ak všetko chýba, rozhoduj iba podľa score."
        "- Ak príde viac fragmentov z toho istého predpisu, uprednostni ten s najnovším metadata.validFrom."
        "- Ak existuje novšia verzia predpisu s porovnateľným score, uprednostni novšiu pred staršou."
        "- Pri citácii vždy uveď metadata.regulation vytlačené <b>bold<b/> (html tag, príklad zákoon č. <b>378/2021 Z. z.</b>)."
    ).format(name="{name}"),

    tools=[searchLaw],
)

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(uid: str, name: str, prompt: str):
    # Log session header to stderr so it isn't streamed
    print(f"[UID: {uid}] [User: {name}] -> {prompt}", file=sys.stderr)
    # Reset per-run embedding usage counters
    usage_counters["embedding_tokens"] = 0
    usage_counters["embedding_model"] = "text-embedding-3-small"
    save_message(uid, "user", prompt)
    # Načítaj posledných 5 správ ako pamäť konverzácie
    mem_rows = fetch_memory(uid, limit=10)
    if mem_rows:
        memory_block = "\n".join([f"{r['role']}: {r['content']}" for r in mem_rows])
        enriched_input = f"[MEMORY]\n{memory_block}\n[/MEMORY]\n\n[USER QUESTION]\n{prompt}"
    else:
        enriched_input = prompt
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
    if assistant_text.strip():
        save_message(uid, "assistant", assistant_text)

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
    try:
        # Vždy zapíš; ak LLM usage nie je, ostanú 0/0 a uloží sa aspoň embedder
        topic = _extract_topic(prompt, max_words=5)
        save_token_usage(
            uid,
            esmeralda.model,
            in_tok,
            out_tok,
            embedding_model=usage_counters.get("embedding_model"),
            embedding_input_tokens=int(usage_counters.get("embedding_tokens", 0) or 0),
            topic=topic,
        )
    except Exception as e:
        print(f"[Supabase] Token usage insert error: {e}", file=sys.stderr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        uid = sys.argv[1]
        name = sys.argv[2]
        prompt = " ".join(sys.argv[3:])
        esmeralda.instructions = esmeralda.instructions.format(name=name)
        asyncio.run(run_once(uid, name, prompt))
    else:
        print("Použitie: python agent.py <uid> <name> <prompt>")
