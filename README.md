# esmeraldaAgent
**Esmeralda** je AI právna asistentka pre Slovlex.  
Používa model `gpt-5-mini`, ktorý dopĺňa o vyhľadávanie v databáze zákonov uložených v Qdrante.  
Správy a spotrebu tokenov zapisuje do Supabase.
## Ako funguje
- Otázky používateľa spracováva `agent.py` pomocou OpenAI Agents SDK.
- Kontext (relevantné paragrafy zákonov) získava z Qdrant kolekcie.
- Každý embedding sa účtuje a spolu s LLM tokenmi sa zapisuje do tabuľky `tokenUsage` v Supabase.
- Samotné správy konverzácie ukladá do tabuľky `chatMessages`.
## Spustenie cez Python (jednorazovo)

```bash
python agent.py <session_id> <name> "<otázka>"
```

Príklad:
```bash
python agent.py "748h7d6c-54g3-4cfc-98fs-ed808d55cd70" "Jozef" "Ako dlho môžem byť vo väzení za vraždu?"
```

Tento spôsob spustí funkciu `run_once`, ktorá:
- uloží správu používateľa,
- pridá kontext z pamäti,
- spustí agenta so streamovaním odpovede,
- zapíše správu asistenta a usage tokenov do DB.

## Spustenie cez FastAPI (SSE)

Beží server `app.py` (typicky za Nginxom a Gunicornom):

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 127.0.0.1:8000 app:app
```

### Volanie cez curl

Na SSE endpoint `/chat` sa dá pripojiť napríklad takto:

```bash
curl -N --http1.1 -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "session_id": "test-session",
    "name": "Matej",
    "prompt": "Ako dlho môžem byť vo väzení za vraždu?"
  }'
```

Výstup bude prichádzať postupne v SSE eventoch:

```
data: {"ready": true}

data: {"delta": "Ako"}
data: {"delta": " dlho"}
data: {"delta": " môžem"}
...
data: {"done": true}
```

## Premenné prostredia

Definuj si ich napr. v `.env`:

## Requirements

Sú v `requirements.txt`.