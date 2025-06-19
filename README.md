# JARVIS Backend

A small FastAPI application serving as the backend for the JARVIS assistant. It wraps AI interactions, simple web search and optional trading integrations.

## Setup

1. **Clone the repository** and create a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**.
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**. Copy `.env.template` to `.env` and fill in the values as needed.
   - `OPENAI_API_KEY` – optional OpenAI API key.
   - `ALPACA_KEY` and `ALPACA_SECRET` – optional trading credentials.
   - `PORT` – port for the FastAPI server (default: `8000`).

   ```bash
   cp .env.template .env
   # edit .env
   ```

## Running the server

Launch the API with Uvicorn:

```bash
uvicorn backend.main:app --reload
```

The API exposes the following endpoints:

- `POST /chat` – provide a JSON body `{ "prompt": "your message" }` and receive a response.
- `POST /trade` – trigger the example AutoTrader logic.
- `GET /health` – simple health check.

## Notes

The trading module is only a stub. Integrate your preferred broker API inside `backend/features/autotrade.py`.

Memory is stored in `data/memory.json`. Delete this file to reset conversation history.
