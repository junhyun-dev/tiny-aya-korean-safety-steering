"""Build the prompt sets used in the experiment.

Inputs (data/raw/):
  MultiJail.csv        DAMO-NLP-SG/MultiJail (MIT) - 315 harmful prompts, parallel in 10 languages incl. English and Korean
  xstest_prompts.csv   Paul/XSTest (CC-BY-4.0) - 250 safe prompts that look unsafe (over-refusal test) + 200 unsafe
  alpaca_cleaned.json  yahma/alpaca-cleaned (CC-BY-4.0) - ordinary instructions, used only for calibration

Outputs (data/sets/*.jsonl, one prompt per line: {"id", "lang", "kind", "split", "prompt"}):
  harmful_{en,ko}_{cal,test}   MultiJail, split by a fixed seed (60 calibration / 255 test)
  harmless_en_cal              60 short Alpaca instructions without input
  safe_en_test                 XSTest safe prompts (250)
  to_translate.jsonl           English prompts that still need a Korean version (XSTest safe + Alpaca calibration)

Korean versions of the XSTest and Alpaca prompts are produced separately (data/translations/ko.jsonl) and
merged by merge_ko.py. Nothing here calls a model.
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "sets"
SEED, N_CAL = 20261109, 60


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


def main():
    rng = random.Random(SEED)
    mj = list(csv.DictReader(open(RAW / "MultiJail.csv", encoding="utf-8-sig")))
    ids = [r["id"] for r in mj]
    rng.shuffle(ids)
    cal = set(ids[:N_CAL])
    for lang in ("en", "ko"):
        for split in ("cal", "test"):
            rows = [{"id": f"mj-{r['id']}", "lang": lang, "kind": "harmful", "split": split, "prompt": r[lang].strip(),
                     "tags": r["tags"], "source": r["source"]}
                    for r in mj if (r["id"] in cal) == (split == "cal")]
            write(f"harmful_{lang}_{split}", rows)

    xs = [r for r in csv.DictReader(open(RAW / "xstest_prompts.csv", encoding="utf-8")) if r["label"] == "safe"]
    safe = [{"id": f"xs-{r['id']}", "lang": "en", "kind": "safe", "split": "test", "prompt": r["prompt"].strip(), "type": r["type"]}
            for r in xs]
    write("safe_en_test", safe)

    alpaca = json.load(open(RAW / "alpaca_cleaned.json", encoding="utf-8"))
    pool = [a["instruction"].strip() for a in alpaca if not a["input"].strip() and 15 <= len(a["instruction"]) <= 120]
    rng.shuffle(pool)
    harmless = [{"id": f"al-{i:03d}", "lang": "en", "kind": "harmless", "split": "cal", "prompt": p} for i, p in enumerate(pool[:N_CAL])]
    write("harmless_en_cal", harmless)

    write("to_translate", [{"id": r["id"], "en": r["prompt"], "note": r.get("type", "calibration")} for r in safe + harmless])


if __name__ == "__main__":
    main()
