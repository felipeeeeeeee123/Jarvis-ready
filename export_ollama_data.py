# Utility script to generate Ollama fine-tuning data from Jarvis logs
import json
from pathlib import Path


def collect() -> list[dict]:
    """Gather memory and audit logs into fine-tune messages."""
    records: list[dict] = []

    mem_path = Path("data/memory.json")
    if mem_path.exists():
        try:
            data = json.loads(mem_path.read_text())
            if isinstance(data, list):
                for item in data:
                    event = item.get("event")
                    if event:
                        records.append({
                            "messages": [
                                {"role": "system", "content": f"Memory: {event}"}
                            ]
                        })
        except Exception:
            pass

    audit_dir = Path("logs/self_audit")
    if audit_dir.exists():
        for log in sorted(audit_dir.glob("*.json")):
            try:
                items = json.loads(log.read_text())
                if isinstance(items, list):
                    for entry in items:
                        prompt = entry.get("prompt")
                        resp = entry.get("response")
                        if prompt and resp:
                            records.append({
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": resp},
                                ]
                            })
            except Exception:
                continue

    stats_path = Path("data/strategy_stats.json")
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
            for name, val in stats.items():
                summary = (
                    f"{name} strategy: wins {val.get('wins', 0)}, "
                    f"losses {val.get('losses', 0)}, "
                    f"pnl {val.get('pnl', 0.0)}"
                )
                records.append({
                    "messages": [
                        {"role": "system", "content": summary}
                    ]
                })
        except Exception:
            pass

    qa_path = Path("data/qa_memory.json")
    if qa_path.exists():
        try:
            qas = json.loads(qa_path.read_text())
            if isinstance(qas, list):
                for qa in qas:
                    q = qa.get("question")
                    a = qa.get("answer")
                    if q and a:
                        records.append({
                            "messages": [
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": a},
                            ]
                        })
        except Exception:
            pass

    return records


def main() -> None:
    records = collect()
    out = Path("training_data.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in records))
    print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
