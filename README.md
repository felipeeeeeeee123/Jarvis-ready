# JARVIS Local Training

This repository fine-tunes an Ollama model using logs and memories from the JARVIS dashboard.

1. Run the exporter to build a JSONL dataset:
   ```bash
   python3 export_ollama_data.py
   ```
   This creates `training_data.jsonl` in OpenAI chat format.

2. Build the fine‑tuned model (requires Ollama installed):
   ```bash
   ollama create jarvisbrain -f Modelfile
   ```

The `Modelfile` uses the base `mistral` model and an adapter named `my-finetune`. The system message defines the assistant as **Felipe Ruiz's personal AI**.
