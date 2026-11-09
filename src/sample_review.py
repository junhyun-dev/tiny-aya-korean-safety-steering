"""Draw a stratified sample of answers for an independent refusal check of the keyword judge.

The reviewer sees only the prompt and the answer (no condition name, no judge label) and marks each one
REFUSE / COMPLY / PARTIAL. compare_review.py then measures how often each judge agrees.
Cells are (condition, set, Qwen3Guard refusal, keyword refusal), so disagreement in any direction can show up;
compare_review.py weights each cell by its real size.
Usage: python src/sample_review.py --conds base cast-mix_a0.3 --per 6 -> runs/review/sample.jsonl (+ key.jsonl kept apart)
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN, EVAL, GUARD = ROOT / "runs" / "gen", ROOT / "runs" / "eval", ROOT / "runs" / "guard"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--per", type=int, default=6, help="answers per (condition, set, guard label, keyword label) cell")
    ap.add_argument("--per-guard-false", type=int, default=None, help="override --per for cells Qwen3Guard calls not refused")
    ap.add_argument("--sets", nargs="+", default=None, help="only these sets")
    ap.add_argument("--out", default="review")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT = ROOT / "runs" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    picked = []
    for c in args.conds:
        for f in sorted((GEN / c).glob("*.jsonl")):
            if args.sets and f.stem not in args.sets:
                continue
            rows = {r["id"]: r for r in load(f)}
            kw = {l["id"]: l["refused_keyword"] for l in load(EVAL / f"{c}__{f.stem}.labels.jsonl")}
            gd = {g["id"]: g["guard_refusal"] for g in load(GUARD / f"{c}__{f.stem}.jsonl")}
            for g_lab in (True, False):
                for k_lab in (True, False):
                    ids = sorted(i for i in rows if gd[i] == g_lab and kw[i] == k_lab)
                    per = args.per if (g_lab or args.per_guard_false is None) else args.per_guard_false
                    for i in rng.sample(ids, min(per, len(ids))):
                        picked.append({"cond": c, "set": f.stem, "guard_refused": g_lab, "keyword_refused": k_lab,
                                       "cell_size": len(ids), "set_size": len(rows), **rows[i]})
    rng.shuffle(picked)
    with open(OUT / "sample.jsonl", "w", encoding="utf-8") as s, open(OUT / "key.jsonl", "w", encoding="utf-8") as k:
        for n, r in enumerate(picked):
            rid = f"r{n:04d}"
            s.write(json.dumps({"rid": rid, "lang": r["lang"], "prompt": r["prompt"], "answer": r["output"]}, ensure_ascii=False) + "\n")
            k.write(json.dumps({"rid": rid, "cond": r["cond"], "set": r["set"], "id": r["id"], "kind": r["kind"],
                                "guard_refused": r["guard_refused"], "keyword_refused": r["keyword_refused"],
                                "cell_size": r["cell_size"], "set_size": r["set_size"]}, ensure_ascii=False) + "\n")
    print(f"{len(picked)} answers -> {OUT / 'sample.jsonl'}")


if __name__ == "__main__":
    main()
