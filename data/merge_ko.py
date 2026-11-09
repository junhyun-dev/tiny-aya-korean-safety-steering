"""Merge the Korean translations (data/translations/ko.jsonl: {"id", "en", "ko"}) into prompt sets:
  safe_ko_test     XSTest safe prompts in Korean (250)
  harmless_ko_cal  Alpaca calibration instructions in Korean (60)
The translations were made by an LLM (see data/translations/README.md) and spot-checked by hand.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ko = {r["id"]: r for r in load(HERE / "translations" / "ko.jsonl")}
    for src, dst in (("safe_en_test", "safe_ko_test"), ("harmless_en_cal", "harmless_ko_cal")):
        rows = load(SETS / f"{src}.jsonl")
        missing = [r["id"] for r in rows if r["id"] not in ko or not ko[r["id"]]["ko"].strip()]
        assert not missing, f"{dst}: no Korean for {missing[:5]}"
        for r in rows:
            assert ko[r["id"]]["en"].strip() == r["prompt"], r["id"]
        with open(SETS / f"{dst}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "lang": "ko", "prompt": ko[r["id"]]["ko"].strip(), "prompt_en": r["prompt"]},
                                   ensure_ascii=False) + "\n")
        print(f"{dst}: {len(rows)}")


if __name__ == "__main__":
    main()
