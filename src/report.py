import json
from pathlib import Path
from datetime import datetime


def write_report(output_path: str, data: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        **data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
