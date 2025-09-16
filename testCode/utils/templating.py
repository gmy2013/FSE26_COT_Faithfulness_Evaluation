"""
utils/templating.py
-------------------
Lightweight helpers to render textual reports and CSV/JSONL artifacts for the pipeline.
"""
from __future__ import annotations
from typing import Dict, List, Any
import json, csv, io

def dicts_to_csv_str(rows: List[dict]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return out.getvalue()

def pretty_json(d: Any) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)
