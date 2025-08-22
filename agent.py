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

# --- OpenAI & Qdrant klienti ---
oi = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# Qdrant nastavenia
qdr = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
#    api_key=os.environ.get("QDRANT_API_KEY") or None,
)

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

# --- Agent: používa len tool výstupy ---
esmeralda = Agent(
    name="Esmeralda",
    model="gpt-5-mini",
    instructions=(
        "Si právny asistent pre SR. Odpovedaj VÝHRADNE z výsledkov nástroja search_law. "
        "Ak nástroj nič použiteľné nevráti, povedz: 'Nenašiel som relevantné informácie'. "
        "Odpovedaj v konverzčnom štýle. "
        "Ak uvádzaš referenciu na použitý text, použi payload z qdrantu metadata.regulation. "
    ),

    tools=[search_law],
)

# --- Jednorazový beh so streamom (ak chceš test bez chatu) ---
async def run_once(prompt: str):
    result = Runner.run_streamed(esmeralda, input=prompt)
    async for ev in result.stream_events():
        # Streamujeme len textové delty
        from openai.types.responses import ResponseTextDeltaEvent
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            print(ev.data.delta, end="", flush=True)
    print()  # newline

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        asyncio.run(run_once(" ".join(sys.argv[1:])))
