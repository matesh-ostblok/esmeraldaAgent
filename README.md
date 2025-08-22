# esmeraldaAgent

Python agent for Slovak law consultations. The agent uses OpenAI `gpt-5-mini` with medium reasoning effort and high verbosity. It retrieves context from a Qdrant collection and stores chat messages and token usage in Supabase.

## Environment variables

The following variables must be defined (for example via a `.env` file):

- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`

## Usage

```
python agent.py <session_id> <name> "<question>"
```

Requirements are listed in `requirements.txt`.
