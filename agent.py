# agent.py
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
from weakref import ref

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from agents import Agent, Runner, function_tool, set_default_openai_key  # Agents SDK
from qdrant_client import QdrantClient
from supabase import create_client, Client

# --- OpenAI & Qdrant klienti ---
oi = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# Qdrant nastavenia
qdr = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
#    api_key=os.environ.get("QDRANT_API_KEY") or None,
)

# Supabase klient
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

COLLECTION = os.environ.get("QDRANT_COLLECTION", "esmeralda")

# --- Pomocná embedding funkcia ---
def embed(text: str) -> List[float]:
    r = oi.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

# --- Tool: vyhľadávanie v Qdrante s PRESNÝM filtrom (vrátane null polí) ---
@function_tool
def search_law(query: str) -> List[Dict[str, Any]]:
    """
    Vyhľadá relevantné právne dokumenty v Qdrant kolekcii 'esmeralda' podľa textového dopytu.
    Návratová hodnota: pole {id, score, payload}.
    """
    vector = embed(query)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Filter presne podľa tvojej špecifikácie (vrátane null hodnôt)
    q_filter = {
        "should": [
            {"is_null": {"key": "metadata.validTo"}},
            {"key": "metadata.validTo", "range": {"gt": None, "gte": today, "lt": None, "lte": None}},
        ]
    }

    body = dict(
        vector=vector,
        limit=10,
        with_payload=True,
        with_vectors=False,
        query_filter=q_filter,  # qdrant-client akceptuje aj dict (prejde ako raw JSON)
    )

    # qdrant-client: search(collection_name=..., query_vector=..., ...)
    res = qdr.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=5,
        with_payload=True,
        with_vectors=False,
#        query_filter=q_filter,
        timeout=120,
    )
    return [
        {"id": p.id, "score": p.score, "payload": p.payload}
        for p in res.points
    ]
    print(">>> ID:", p.id, "SCORE:", p.score, "PAYLOAD:", p.payload)

def save_message(session_id: str, role: str, content: str) -> None:
    """
    Uloží správu do public.chatMessages v Supabase.
    Očakáva role ∈ {"user","assistant"}.
    """
    try:
        sb.table("chatMessages").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        # Nezastavuj beh agenta kvôli logovaniu
        print(f"[Supabase] Insert error: {e}")

def save_token_usage(session_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """
    Uloží token usage do public.tokenUsage v Supabase.
    """
    try:
        sb.table("tokenUsage").insert({
            "session_id": session_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }).execute()
    except Exception as e:
        print(f"[Supabase] Token usage insert error: {e}")

def fetch_memory(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Načíta posledných `limit` správ zo session a vráti ich v chronologickom poradí (najstaršia -> najnovšia).
    """
    try:
        resp = sb.table("chatMessages")\
            .select("role,content,created_at")\
            .eq("session_id", session_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        rows = (getattr(resp, "data", None) or [])
        # otoč na chronologické poradie
        rows.reverse()
        return rows
    except Exception as e:
        print(f"[Supabase] Fetch memory error: {e}")
        return []

# --- Agent: používa len tool výstupy ---
esmeralda = Agent(
    name="Esmeralda",
    model="gpt-5-mini",
    instructions=(
        "Si právna asistentka pre SR. Odpovedaj VÝHRADNE z výsledkov nástroja search_law."
        "Ak nástroj nič použiteľné nevráti, povedz: Nenašiel som relevantné informácie."
        "Odpovedaj v konverzčnom štýle, nedávaj rady, iba odporúčania ak treba. Nepoužívaj odrážky ani číslovanie."
        "Ak uvádzaš referenciu na použitý text, použi payload z qdrantu metadata.regulation."
        "Otázku používateľa rozlož semanticky na menšie frázy (2–7 slov), ktoré jednotlivo posielaj do searchLaw."
    ).format(name="{name}"),

    tools=[search_law],
)

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(session_id: str, name: str, prompt: str):
    print(f"[Session: {session_id}] [User: {name}] -> {prompt}")
    save_message(session_id, "user", prompt)
    # Načítaj posledných 5 správ ako pamäť konverzácie
    mem_rows = fetch_memory(session_id, limit=10)
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
                print(f"[Token Usage] Parse error: {e}")
    print()  # newline
    assistant_text = "".join(buf)
    if assistant_text.strip():
        save_message(session_id, "assistant", assistant_text)

    # Najprv skús usage z run contextu (Agents SDK ukladá usage tam po dokončení streamu)
    if not usage:
        try:
            ctx = getattr(result, "context_wrapper", None)
            if ctx and getattr(ctx, "usage", None):
                usage = ctx.usage
        except Exception as e:
            print(f"[Token Usage] Context usage parse error: {e}")

    # Fallback – ak by context nemal usage, skús finálnu odpoveď
    if not usage:
        try:
            final = await result.get_final_response()
            if final and getattr(final, "output", None):
                out0 = final.output[0]
                if hasattr(out0, "usage") and out0.usage:
                    usage = out0.usage
        except Exception as e:
            print(f"[Token Usage] Final response parse error: {e}")

    if usage:
        try:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            save_token_usage(session_id, esmeralda.model, input_tokens, output_tokens)
        except Exception as e:
            print(f"[Token Usage] Could not save token usage: {e}")
    else:
        print("[Token Usage] Usage not available at all.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        session_id = sys.argv[1]
        name = sys.argv[2]
        prompt = " ".join(sys.argv[3:])
        esmeralda.instructions = esmeralda.instructions.format(name=name)
        asyncio.run(run_once(session_id, name, prompt))
    else:
        print("Použitie: python agent.py <session_id> <name> <prompt>")
