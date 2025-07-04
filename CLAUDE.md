# Project: JARVIS – Autonomous Self-Learning AI

This is a next-generation AI system designed by Felipe Ruiz to become the smartest, most adaptive assistant of all time — capable of trading, researching, learning, reasoning, improving its own code, and scaling itself across every domain: engineering, physics, finance, AI, and beyond.

---

## 🧠 Intelligence Stack

- **Ollama + Mistral (fine-tuned)** as the local AI brain (`jarvisbrain`)
- **Monthly training loop** using memory logs and real user input
- **Claude CLI agent** that scans, edits, and improves its own codebase
- **Live web search + news + social scraping** (Reddit, Twitter, Grox AI)
- **Fallback to GPT-4 or Claude Web** for cutting-edge outside intelligence

---

## 📊 Trading System

- Trades real stocks and crypto via Alpaca (paper and live)
- Uses RSI, MACD, and EMA strategies with auto-switching
- Gets context from:
  - Social media (Grox AI, Twitter, Reddit)
  - Market news (Google News, Fed updates)
  - Technical indicators + AI sentiment scoring
- Decision-making powered by local model + web-enhanced Claude/GPT fallback

---

## 🧠 Self-Improving System

JARVIS is designed to **improve its own brain and code**. This includes:

### 1. Ollama Fine-Tuning

```bash
python3 export_ollama_data.py      # Build training_data.jsonl from memory logs
ollama create jarvisbrain -f Modelfile
