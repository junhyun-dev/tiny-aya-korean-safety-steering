"""Export the steering artifacts so others can reuse them without re-running calibration.

  artifacts/refusal_direction_{en,ko,mix}.safetensors   SafeTune refusal directions (unit vectors, hidden size) + layer
  artifacts/cast_{cal}.safetensors + cast_{cal}.json    SafeTune CAST: CAA behaviour vectors per layer, the condition
                                                        vector, and the gate settings (layer, threshold, comparator)
Nothing here changes Tiny Aya's weights; the files are small and are applied at inference time.
Usage: python src/export_artifacts.py --cast mix en ko
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import run

OUT = run.ROOT / "artifacts"


def plain(v):
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu().tolist() if v.numel() <= 8 else f"tensor{list(v.shape)}"
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cast", nargs="*", default=["mix"])
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for cal in ("en", "ko", "mix"):
        saved = torch.load(run.RUNS / "directions" / f"{cal}.pt")
        save_file({"direction": saved["direction"].contiguous()}, OUT / f"refusal_direction_{cal}.safetensors",
                  metadata={"model": run.MODEL_ID, "layer": str(saved["layer"]), "calibration": cal,
                            "n_harmful": str(saved["n_harmful"]), "n_harmless": str(saved["n_harmless"])})
    if args.cast:
        model, tok = run.model_and_tokenizer()
        for cal in args.cast:
            cast = run.cast_model(model, tok, cal, args.alpha)
            tensors = {f"caa_layer_{k}": v.detach().float().cpu().contiguous() for k, v in cast.steering_vectors.items()}
            cv = getattr(cast, "condition_vector", None)
            if cv is not None:
                tensors["condition_vector"] = cv.detach().float().cpu().contiguous()
            save_file(tensors, OUT / f"cast_{cal}.safetensors")
            cfg = {"model": run.MODEL_ID, "method": "SafeTune CASTTrainer (Conditional Activation Steering, arXiv:2409.05907)",
                   "calibration": cal, "alpha": args.alpha, "per_prompt_gate": True,
                   "behaviour_layers": sorted(int(k) for k in cast.steering_vectors),
                   **{k: plain(getattr(cast, k, None)) for k in ("condition_layer", "threshold", "comparator", "pool")}}
            (OUT / f"cast_{cal}.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            print("cast", cal, cfg, flush=True)
            gate_scores(cast, tok, cal)
    print("artifacts ->", OUT)


@torch.no_grad()
def gate_scores(cast, tok, cal, batch=16):
    """The gate's cosine score for every prompt, encoded exactly as run.answer() does (chat template, left padding).

    With a per-prompt gate, a prompt is either steered or answered as base, so a *stricter* threshold can be
    replayed from these scores + the existing base and steered answers without generating again
    (src/threshold_curve.py)."""
    out_dir = run.RUNS / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = run.TEST_SETS + ["harmful_en_cal", "harmful_ko_cal", "harmless_en_cal", "harmless_ko_cal"]
    with open(out_dir / f"cast-{cal}.jsonl", "w", encoding="utf-8") as f:
        for s in sets:
            rows = run.load(run.SETS / f"{s}.jsonl")
            for i in range(0, len(rows), batch):
                chunk = [[{"role": "user", "content": r["prompt"]}] for r in rows[i:i + batch]]
                enc = tok.apply_chat_template(chunk, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                              padding=True).to(cast.model.device)
                fires, sims = cast._gate_fires(enc["input_ids"], attention_mask=enc["attention_mask"])
                for r, fire, sim in zip(rows[i:i + batch], fires, sims):
                    f.write(json.dumps({"set": s, "id": r["id"], "sim": float(sim), "fires": bool(fire),
                                        "threshold": cast.threshold, "comparator": cast.comparator}) + "\n")
    print("gate scores", cal, "->", out_dir / f"cast-{cal}.jsonl", flush=True)


if __name__ == "__main__":
    main()
