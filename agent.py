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

# --- Agent: používa len tool výstupy ---
esmeralda = Agent(
    name="Esmeralda",
    model="gpt-5-mini",
    instructions=(
        "Si právna asistentka pre SR. Odpovedaj VÝHRADNE z výsledkov nástroja search_law."
        "Ak nástroj nič použiteľné nevráti, povedz: Nenašiel som relevantné informácie."
        "Odpovedaj v konverzčnom štýle."
        "Ak uvádzaš referenciu na použitý text, použi payload z qdrantu metadata.regulation."
        "Otázku používateľa rozlož semanticky na maximálne 5 menších fráz (2–7 slov), ktoré jednotlivo posielaj do searchLaw."
        "Užívateľ sa volá {{name}}."
    ),

    tools=[search_law],
)

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(session_id: str, name: str, prompt: str):
    print(f"[Session: {session_id}] [User: {name}] -> {prompt}")
    save_message(session_id, "user", prompt)
    result = Runner.run_streamed(esmeralda, input=prompt)
    buf = []
    async for ev in result.stream_events():
        # Streamujeme len textové delty
        from openai.types.responses import ResponseTextDeltaEvent
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            piece = ev.data.delta or ""
            buf.append(piece)
            print(piece, end="", flush=True)
    print()  # newline
    assistant_text = "".join(buf)
    if assistant_text.strip():
        save_message(session_id, "assistant", assistant_text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        session_id = sys.argv[1]
        name = sys.argv[2]
        prompt = " ".join(sys.argv[3:])
        asyncio.run(run_once(session_id, name, prompt))
    else:
        print("Použitie: python agent.py <session_id> <name> <prompt>")
