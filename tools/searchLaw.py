import os
from datetime import datetime, timezone
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from agents import function_tool
from qdrant_client import QdrantClient

# --- Usage accumulators (per-run) ---
usage_counters = {
    "embedding_tokens": 0,
    "embedding_model": "text-embedding-3-small",
}

# --- OpenAI & Qdrant clients ---
oi = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
qdr = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    # api_key can be added if needed: os.environ.get("QDRANT_API_KEY")
)

COLLECTION = os.environ.get("QDRANT_COLLECTION", "esmeralda")


def embed(text: str) -> List[float]:
    """Create an embedding and track token usage locally."""
    r = oi.embeddings.create(model="text-embedding-3-small", input=text)
    try:
        prompt_tokens = 0
        if hasattr(r, "usage") and getattr(r.usage, "prompt_tokens", None) is not None:
            prompt_tokens = int(r.usage.prompt_tokens)
        usage_counters["embedding_tokens"] += prompt_tokens
    except Exception:
        pass
    try:
        usage_counters["embedding_model"] = getattr(r, "model", None) or "text-embedding-3-small"
    except Exception:
        usage_counters["embedding_model"] = "text-embedding-3-small"
    return r.data[0].embedding


@function_tool
def searchLaw(query: str) -> List[Dict[str, Any]]:
    """
    Vyhľadá relevantné právne dokumenty v Qdrant kolekcii podľa textového dopytu.
    Návratová hodnota: pole {id, score, payload}.
    """
    vector = embed(query)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Filter presne podľa špecifikácie (vrátane null hodnôt)
    q_filter = {
        "should": [
            {"is_null": {"key": "metadata.validTo"}},
            {"key": "metadata.validTo", "range": {"gt": None, "gte": today, "lt": None, "lte": None}},
        ]
    }

    res = qdr.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=5,
        with_payload=True,
        with_vectors=False,
        query_filter=q_filter,
        timeout=120,
    )
    results = [
        {"id": p.id, "score": p.score, "payload": p.payload}
        for p in res.points
    ]
    try:
        # Log to stderr so SSE (which captures stdout) does not include tool logs
        print(f"[searchLaw] hits={len(results)}", file=sys.stderr)
    except Exception:
        pass
    return results
