import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from supabase import Client, create_client

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in Slovak law. "
    "Address the user by their provided name and remind them to consult a qualified lawyer for formal advice."
)
EMBEDDING_MODEL = "text-embedding-3-large"


def _embed(text: str) -> List[float]:
    embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return embedding.data[0].embedding


def _get_context(question: str, limit: int = 5) -> List[str]:
    vector = _embed(question)
    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=limit,
    )
    return [r.payload.get("content", "") for r in results]


def _store_message(session_id: str, role: str, content: str) -> None:
    supabase.table("chatMessages").insert(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
        }
    ).execute()


def _store_token_usage(
    session_id: str, model: str, input_tokens: int, output_tokens: int
) -> None:
    supabase.table("tokenUsage").insert(
        {
            "session_id": session_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    ).execute()


def chat(session_id: str, name: str, message: str) -> str:
    _store_message(session_id, "user", message)
    context = _get_context(message)
    context_block = "\n".join(context)
    prompt = f"{context_block}\n\n{name}: {message}"

    response = client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body={"verbosity": "high"},
    )

    answer = response.output[0].content[0].text
    _store_message(session_id, "assistant", answer)

    usage = response.usage
    _store_token_usage(
        session_id,
        "gpt-5-mini",
        usage.input_tokens,
        usage.output_tokens,
    )

    return f"{name}, {answer}"


if __name__ == "__main__":
    import sys

    session = sys.argv[1]
    username = sys.argv[2]
    question = " ".join(sys.argv[3:])
    print(chat(session, username, question))
