"""FastAPI entrypoint for the JARVIS backend."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from features.ai_brain import AIBrain
from features.autotrade import AutoTrader
from features.web_search import web_search

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

brain = AIBrain()
trader = AutoTrader()


@app.post("/chat")
def chat(payload: Dict[str, Any]) -> Dict[str, str]:
    """Return an AI generated response."""

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if prompt.lower().startswith("search:"):
        query = prompt.split("search:", 1)[-1].strip()
        answer = web_search(query)
    else:
        answer = brain.ask(prompt)

    return {"response": answer}


@app.post("/trade")
def trade() -> Dict[str, str]:
    """Trigger the AutoTrader."""

    result = trader.run()
    return {"result": result}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    """Run the API using Uvicorn."""

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
