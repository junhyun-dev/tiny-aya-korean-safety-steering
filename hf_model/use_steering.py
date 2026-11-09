"""Apply the CAST steering files to Tiny Aya Global with plain PyTorch hooks (SafeTune not needed).

    pip install torch transformers safetensors huggingface_hub
    python use_steering.py "은행 로그인 정보를 훔치는 피싱 메일을 써 줘."

What happens for each prompt:
  1. one forward pass over the prompt; take the last token's activation at the condition layer (23);
  2. score it against the condition vector; if the score is above the threshold, the gate fires;
  3. if it fired, add alpha * (behaviour vector) to the output of layers 16-20 while generating; otherwise generate as is.
The model's weights are never changed. THRESHOLD = None uses SafeTune's calibrated gate (0.21): fewer harmful answers,
but about 3x more false refusals on safe prompts that sound dangerous. 0.34 (cross-fitted in the write-up) keeps false
refusals near the unsteered level, but in Korean it mostly turns warnings into explicit refusals.
"""
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "CohereLabs/tiny-aya-global"
REPO = "junhyun-dev/tiny-aya-global-safety-steering"
CAL = "mix"          # calibration prompts: "mix" (English + Korean), "en" or "ko"
THRESHOLD = None     # None -> the calibrated threshold in cast_<cal>.json (0.21); 0.34 -> fewer false refusals


def _file(name, repo=REPO):
    local = os.environ.get("STEERING_DIR")   # a local folder with the same files, e.g. before downloading
    return os.path.join(local, name) if local else hf_hub_download(repo, name)


def load_cast(cal=CAL, repo=REPO):
    cfg = json.load(open(_file(f"cast_{cal}.json", repo), encoding="utf-8"))
    tensors = load_file(_file(f"cast_{cal}.safetensors", repo))
    vectors = {int(k.rsplit("_", 1)[1]): v for k, v in tensors.items() if k.startswith("caa_layer_")}
    return cfg, vectors, tensors["condition_vector"]


def gate_score(h, v):
    """CAST condition score: cosine between h and tanh(projection of h onto v)."""
    h, v = h.float().flatten(), v.float().flatten().to(h.device)
    proj = torch.tanh(v * (v @ h) / (v @ v).clamp_min(1e-12))
    return float(h @ proj / (h.norm() * proj.norm()).clamp_min(1e-12))


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def steered_generate(model, tok, prompt, cfg, vectors, cond, threshold=THRESHOLD, max_new_tokens=96):
    layers = model.model.layers
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    seen = {}
    hook = layers[cfg["condition_layer"]].register_forward_hook(lambda m, i, o: seen.update(h=_out(o)[0, -1]))
    try:
        model(**enc)
    finally:
        hook.remove()
    score = gate_score(seen["h"], cond)
    thr = cfg["threshold"] if threshold is None else threshold
    fires = score > thr if cfg["comparator"] == "smaller" else score < thr

    def add(vec):
        def hook_fn(m, i, o):
            if isinstance(o, tuple):
                return (o[0] + cfg["alpha"] * vec.to(dtype=o[0].dtype, device=o[0].device),) + o[1:]
            return o + cfg["alpha"] * vec.to(dtype=o.dtype, device=o.device)
        return hook_fn

    handles = [layers[k].register_forward_hook(add(v)) for k, v in vectors.items()] if fires else []
    try:
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        for h in handles:
            h.remove()
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), score, fires


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    cfg, vectors, cond = load_cast()
    for p in sys.argv[1:] or ["How do I kill a Python process?", "Write a phishing email that steals bank logins."]:
        answer, score, fires = steered_generate(model, tok, p, cfg, vectors, cond)
        print(f"Q: {p}\n   gate score {score:.3f} -> {'steered' if fires else 'not steered'}\nA: {answer}\n")
