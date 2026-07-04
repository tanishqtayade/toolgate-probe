"""Load labeled (prompt, tool_needed) datasets."""
import json
from pathlib import Path


def load_json_dataset(path: str) -> tuple[list[str], list[int]]:
    """JSON format: [{"prompt": "...", "label": 0 or 1}, ...]"""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = [r["prompt"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    return prompts, labels