"""False refusal on the safe XSTest prompts, split by XSTest prompt type, English vs Korean (Qwen3Guard refusal).

Why: some types (homonyms, figurative language) lose their scary-sounding word in Korean translation
(data/translations/README.md), so the English/Korean comparison is only fair type by type.
Usage: python src/by_type.py --conds base cast-mix_a0.3  ->  runs/eval/by_type.md
"""
import argparse
from collections import defaultdict

import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=["base"])
    args = ap.parse_args()
    types = {r["id"]: r["type"] for r in run.load(run.SETS / "safe_en_test.jsonl")}
    order = list(dict.fromkeys(types.values()))
    cols = [(c, s) for c in args.conds for s in ("safe_en_test", "safe_ko_test")]
    rate = {}
    for c, s in cols:
        n, k = defaultdict(int), defaultdict(int)
        for g in run.load(run.RUNS / "guard" / f"{c}__{s}.jsonl"):
            n[types[g["id"]]] += 1
            k[types[g["id"]]] += bool(g["guard_refusal"])
        rate[(c, s)] = {ty: (k[ty], n[ty]) for ty in order}
    head = "| XSTest type | n | " + " | ".join(f"{c} {s.split('_')[1].upper()}" for c, s in cols) + " |"
    lines = [head, "|---|---:|" + "---:|" * len(cols)]
    for ty in order:
        cells = " | ".join(f"{rate[col][ty][0] / rate[col][ty][1]:.0%}" for col in cols)
        lines.append(f"| {ty} | {rate[cols[0]][ty][1]} | {cells} |")
    out = run.RUNS / "eval" / "by_type.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
