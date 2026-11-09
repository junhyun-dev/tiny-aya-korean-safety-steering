"""Clean and de-duplicate every prompt set with CuratorKIT before any model sees it.

- exact + MinHash near-duplicate removal across all sets (a duplicate across calibration and test would leak)
- text cleaning (whitespace, control characters) with `strip_html` turned OFF: CuratorKIT strips tag-shaped
  text by default, which silently rewrites safety prompts that contain markup (Lexsi-Labs/CuratorKIT#42)
Writes data/clean/<set>.jsonl (same ids, cleaned prompt) and runs/curate/report.json.
"""
import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig

ROOT = Path(__file__).resolve().parents[1]
SETS, CLEAN, REPORT = ROOT / "data" / "sets", ROOT / "data" / "clean", ROOT / "runs" / "curate"
NAMES = ["harmful_en_cal", "harmful_ko_cal", "harmless_en_cal", "harmless_ko_cal",
         "harmful_en_test", "harmful_ko_test", "safe_en_test", "safe_ko_test"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    rows = {n: load(SETS / f"{n}.jsonl") for n in NAMES if (SETS / f"{n}.jsonl").exists()}
    tmp = REPORT / "input.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for n, rs in rows.items():
            for r in rs:
                f.write(json.dumps({"key": f"{n}::{r['id']}", "instruction": r["prompt"], "input": "", "output": ""},
                                   ensure_ascii=False) + "\n")
    res = Curator(CuratorConfig(dataset=str(tmp), format="alpaca", dedup="minhash", minhash_threshold=0.85,
                                clean=True, clean_transforms={"strip_html": False}, min_tokens=1, schema_gate=False,
                                output_dir=str(REPORT / "curatorkit"), export_formats=["alpaca"])).run()
    kept = {}
    for s in res.passed:
        meta = s.metadata or {}
        key = meta.get("key") or meta.get("raw_preview", {}).get("key")   # extra input columns land in metadata
        kept[key] = s.instruction
    assert kept and None not in kept, "could not map CuratorKIT output back to prompt ids"
    report = {"stage_counts": res.stage_counts, "sets": {}}
    for n, rs in rows.items():
        out, dropped, changed = [], [], []
        for r in rs:
            key = f"{n}::{r['id']}"
            if key not in kept:
                dropped.append(r["id"])
                continue
            if kept[key] != r["prompt"]:
                changed.append(r["id"])
            out.append({**r, "prompt": kept[key]})
        with open(CLEAN / f"{n}.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["sets"][n] = {"in": len(rs), "kept": len(out), "dropped": dropped, "changed": changed}
        print(f"{n}: {len(rs)} -> {len(out)} (changed {len(changed)}, dropped {len(dropped)})")
    json.dump(report, open(REPORT / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
