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

## Setup on Felipe Ruiz's PC

Follow these steps on a Windows machine to run the local JARVIS instance:

1. **Install Ollama** – [Download](https://ollama.com/download) and install.
2. **Pull the base model**:
   ```bash
   ollama pull mistral
   ```
3. **Run the fine-tune** in this repository directory:
   ```bash
   ollama create jarvisbrain -f Modelfile
   ```
4. **Launch JARVIS**:
   ```bash
   python backend/gui_dashboard.py
   ```

To preserve and improve the AI's memory, export logs and retrain monthly:

```bash
python export_ollama_data.py
ollama create jarvisbrain -f Modelfile
```
