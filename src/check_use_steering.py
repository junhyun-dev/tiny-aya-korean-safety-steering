"""Check hf_model/use_steering.py (plain PyTorch hooks, one prompt at a time) against the SafeTune runs.

For 3 prompts per test set: its gate score should match runs/gate/cast-mix.jsonl, and its answer should match the
answer in runs/gen/cast-mix_a0.3_t0.34 (generated in batches of 16 with left padding, so tiny numeric differences
can change a late token; we report exact matches and the first differing character).
Usage: STEERING_DIR=artifacts python src/check_use_steering.py
"""
import sys

import run

sys.path.insert(0, str(run.ROOT / "hf_model"))
import use_steering as U  # noqa: E402


def main():
    model, tok = run.model_and_tokenizer()
    cfg, vectors, cond = U.load_cast("mix")
    gate = {(g["set"], g["id"]): g["sim"] for g in run.load(run.RUNS / "gate" / "cast-mix.jsonl")}
    n = same = 0
    for s in run.TEST_SETS:
        ref = {r["id"]: r for r in run.load(run.RUNS / "gen" / "cast-mix_a0.3_t0.34" / f"{s}.jsonl")}
        for r in list(ref.values())[:3]:
            ans, score, fires = U.steered_generate(model, tok, r["prompt"], cfg, vectors, cond, threshold=0.34)
            ok = ans == r["output"]
            first = next((i for i, (a, b) in enumerate(zip(ans, r["output"])) if a != b), min(len(ans), len(r["output"])))
            n += 1
            same += ok
            print(f"{s} {r['id']}: score {score:.4f} vs {gate[(s, r['id'])]:.4f} fires={fires} "
                  f"answer {'same' if ok else f'differs from char {first}'}", flush=True)
    print(f"{same}/{n} answers identical")


if __name__ == "__main__":
    main()
