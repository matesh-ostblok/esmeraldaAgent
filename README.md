# esmeraldaAgent
**Esmeralda** je AI právna asistentka pre Slovlex.  
Používa model `gpt-5-mini`, ktorý dopĺňa o vyhľadávanie v databáze zákonov uložených v Qdrante.  
Správy a spotrebu tokenov zapisuje webová aplikácia do Supabase (agent už nezapisuje priamo).
## Ako funguje
- Otázky používateľa spracováva `agent.py` pomocou OpenAI Agents SDK.
- Kontext (relevantné paragrafy zákonov) získava z Qdrant kolekcie.
- Na vyhľadávanie používa nástroj `searchLaw` v súbore `tools/searchLaw.py`.
- Každý embedding sa účtuje a spolu s LLM tokenmi sa zapisuje do tabuľky `tokenUsage` v Supabase webovou aplikáciou.
  Krátky 5‑slovný „topic" generuje lacný LLM na strane webu; agent túto hodnotu nevypočítava.
- Samotné správy konverzácie ukladá do tabuľky `chatMessages`.

## Štruktúra projektu

- `agent.py`: definícia agenta a stream behu (bez priamych zápisov do DB; usage vracia ako metadáta).
- `app.py`: FastAPI server so SSE endpointom `/chat`.
- `tools/searchLaw.py`: nástroj `searchLaw` (Qdrant vyhľadávanie + embedding cez OpenAI).
## Spustenie cez Python (jednorazovo)

```bash
python agent.py <uid> <name> "<otázka>"
```

Príklad:
```bash
python agent.py "748h7d6c-54g3-4cfc-98fs-ed808d55cd70" "Jozef" "Ako dlho môžem byť vo väzení za vraždu?"
```

Tento spôsob spustí funkciu `run_once`, ktorá:
- uloží správu používateľa (robí webová aplikácia),
- pridá kontext z pamäti,
- spustí agenta so streamovaním odpovede,
- vráti usage tokenov ako metadáta pre webovú aplikáciu, ktorá ich zapíše do DB.

## Spustenie cez FastAPI (SSE)

Beží server `app.py` (typicky za Nginxom a Gunicornom):

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 127.0.0.1:8000 app:app
```

### Volanie cez curl

Na SSE endpoint `/chat` sa dá pripojiť napríklad takto (akceptuje kľúč `prompt` aj alias `message`):

```bash
curl -N --http1.1 -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "uid": "test-session",
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

```
# OpenAI
OPENAI_API_KEY=...

# Qdrant
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=...           # (voliteľné, ak máš zapnutú autorizáciu)
QDRANT_COLLECTION=esmeralda     # (voliteľné, default je "esmeralda")

# Supabase
SUPABASE_URL=https://...supabase.co
SUPABASE_KEY=...
```

## Inštalácia závislostí

```
pip install -r requirements.txt
```

Poznámka: `.env` sa načítava v `agent.py` aj v `tools/searchLaw.py` (idempotentne), aby bolo možné použiť nástroj aj samostatne.
