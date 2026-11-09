"""Assemble the Hugging Face model and dataset folders, and optionally upload them.

  build/hf_model/    artifacts/*.safetensors + *.json (steering files), use_steering.py, README.md (model card)
  build/hf_dataset/  prompts/, answers/ (every answer + judge labels + gate score), review/ (independent labels),
                     gate_scores/, results/, README.md (dataset card)

Answers that Qwen3Guard judged "Unsafe", or that an independent reviewer marked as harmful content, are published as
"[redacted: judged unsafe]" (with `redacted: true`), in every condition where the same text appears; their prompts and
labels stay. Everything else is kept verbatim.

Usage: python src/export_hf.py                 # build only
       python src/export_hf.py --upload        # build, then create/update the repos (private until --public)
       python src/export_hf.py --upload --public --collections
"""
import argparse
import json
import shutil

import run

BUILD = run.ROOT / "build"
USER = "junhyun-dev"
MODEL_REPO = f"{USER}/tiny-aya-global-safety-steering"
DATA_REPO = f"{USER}/tiny-aya-korean-safety-eval"
CONDITIONS = ["base", "mix_a0.3", "cast-en_a0.3", "cast-ko_a0.3", "cast-mix_a0.3", "cast-mix_a0.3_t0.34"]
REVIEWS = [("REQ-02", "review", "labels_sol.jsonl"), ("REQ-03", "review_census", "labels_sol.jsonl"),
           ("REQ-04", "review_cast", "labels_sol.jsonl")]
REDACTED = "[redacted: judged unsafe]"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fired(g, thr=None):
    t = g["threshold"] if thr is None else thr
    return g["sim"] > t if g["comparator"] == "smaller" else g["sim"] < t


def build_model():
    out = BUILD / "hf_model"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for f in sorted((run.ROOT / "artifacts").iterdir()):
        shutil.copy(f, out / f.name)
    for f in ("README.md", "use_steering.py"):
        shutil.copy(run.ROOT / "hf_model" / f, out / f)
    return out


def build_dataset():
    out = BUILD / "hf_dataset"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    types = {r["id"]: r.get("type") for r in run.load(run.SETS / "safe_en_test.jsonl")}

    # reviewer labels, to redact anything a reviewer called harmful
    flagged, review_rows = set(), []
    for rnd, folder, labels in REVIEWS:
        kp, lp = run.RUNS / folder / "key.jsonl", run.RUNS / folder / labels
        if not (kp.exists() and lp.exists()):
            print("skip review", rnd, "(no labels yet)")
            continue
        lab = {r["rid"]: r for r in run.load(lp)}
        for k in run.load(kp):
            l = lab[k["rid"]]
            if l.get("harmful_content"):
                flagged.add((k["cond"], k["set"], k["id"]))
            review_rows.append({"round": rnd, "rid": k["rid"], "condition": k["cond"], "set": k["set"], "id": k["id"],
                                "guard_refused": k["guard_refused"], "keyword_refused": k["keyword_refused"],
                                "cell_size": k["cell_size"], "set_size": k["set_size"],
                                "reviewer_label": l["label"], "reviewer_harmful_content": bool(l.get("harmful_content")),
                                "reviewer_note": l.get("note", "")})
    if review_rows:
        write_jsonl(out / "review" / "labels.jsonl", review_rows)

    sets = run.TEST_SETS + ["harmful_en_cal", "harmful_ko_cal", "harmless_en_cal", "harmless_ko_cal"]
    for s in sets:
        write_jsonl(out / "prompts" / f"{s}.jsonl",
                    [{"id": r["id"], "set": s, "lang": r["lang"], "kind": r["kind"], "split": r["split"],
                      "xstest_type": r.get("type"), "prompt": r["prompt"]} for r in run.load(run.SETS / f"{s}.jsonl")])

    gate = {}
    for cal in ("mix", "en", "ko"):
        gp = run.RUNS / "gate" / f"cast-{cal}.jsonl"
        (out / "gate_scores").mkdir(exist_ok=True)
        shutil.copy(gp, out / "gate_scores" / gp.name)
        gate[cal] = {(g["set"], g["id"]): g for g in run.load(gp)}

    # an answer text judged unsafe anywhere is redacted everywhere (unsteered prompts reuse the base answer text)
    bad_texts = set()
    for c in CONDITIONS:
        for s in run.TEST_SETS:
            guard = {g["id"]: g for g in run.load(run.RUNS / "guard" / f"{c}__{s}.jsonl")}
            for r in run.load(run.RUNS / "gen" / c / f"{s}.jsonl"):
                if guard[r["id"]]["guard_safety"] == "Unsafe" or (c, s, r["id"]) in flagged:
                    bad_texts.add(r["output"])
    n_red = 0
    for c in CONDITIONS:
        cal = c.split("_")[0].removeprefix("cast-") if c.startswith("cast-") else None
        thr = float(c.split("_t")[1]) if "_t" in c else None
        rows = []
        for s in run.TEST_SETS:
            gen = run.load(run.RUNS / "gen" / c / f"{s}.jsonl")
            guard = {g["id"]: g for g in run.load(run.RUNS / "guard" / f"{c}__{s}.jsonl")}
            kw = {l["id"]: l["refused_keyword"] for l in run.load(run.RUNS / "eval" / f"{c}__{s}.labels.jsonl")}
            for r in gen:
                g = guard[r["id"]]
                bad = r["output"] in bad_texts
                n_red += bad
                gs = gate[cal][(s, r["id"])] if cal else None
                rows.append({"condition": c, "set": s, "id": r["id"], "lang": r["lang"], "kind": r["kind"],
                             "xstest_type": types.get(r["id"]) if r["kind"] == "safe" else None,
                             "prompt": r["prompt"], "answer": REDACTED if bad else r["output"], "redacted": bad,
                             "guard_refusal": bool(g["guard_refusal"]), "guard_safety": g["guard_safety"],
                             "guard_categories": g.get("guard_categories") or [], "keyword_refusal": bool(kw[r["id"]]),
                             "gate_score": gs["sim"] if gs else None,
                             "gate_fired": fired(gs, thr) if gs else None})
        write_jsonl(out / "answers" / f"{c}.jsonl", rows)
    print("redacted answers:", n_red)

    res = out / "results"
    res.mkdir()
    for src, name in [(run.RUNS / "eval" / "summary.json", "summary.json"), (run.RUNS / "eval" / "table.md", "table.md"),
                      (run.RUNS / "eval" / "by_type.md", "by_type.md"), (run.RUNS / "threshold" / "table.md", "threshold_table.md"),
                      (run.RUNS / "review" / "agreement.md", "agreement_round1.md"),
                      (run.RUNS / "review_census" / "agreement.md", "agreement_round2.md"),
                      (run.RUNS / "review_cast" / "agreement.md", "agreement_round3.md")]:
        if src.exists():
            shutil.copy(src, res / name)
    for cal in ("mix", "en", "ko"):
        tp = run.RUNS / "threshold" / f"{cal}.json"
        if tp.exists():
            shutil.copy(tp, res / f"threshold_{cal}.json")
    shutil.copy(run.ROOT / "hf_dataset" / "README.md", out / "README.md")
    return out


def upload(model_dir, data_dir, public, collections):
    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind, folder in ((MODEL_REPO, "model", model_dir), (DATA_REPO, "dataset", data_dir)):
        api.create_repo(repo, repo_type=kind, private=not public, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                          commit_message="Lexsi hackathon: steering files, answers and judge labels")
        if public:
            api.update_repo_settings(repo, repo_type=kind, private=False)
        print("uploaded", kind, f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")
    if collections:
        made = {}
        for title, desc, items in (
                ("Tiny Aya Korean safety steering: models",
                 "Inference-time safety steering for Tiny Aya Global (SafeTune CAST), and the models it was built and judged with.",
                 [(MODEL_REPO, "model"), ("CohereLabs/tiny-aya-global", "model"), ("Qwen/Qwen3Guard-Gen-4B", "model")]),
                ("Tiny Aya Korean safety steering: datasets",
                 "English vs. Korean refusal and over-refusal of Tiny Aya Global, before and after steering, and the source prompt sets.",
                 [(DATA_REPO, "dataset"), ("DAMO-NLP-SG/MultiJail", "dataset"), ("Paul/XSTest", "dataset"),
                  ("yahma/alpaca-cleaned", "dataset")])):
            existing = [c for c in api.list_collections(owner=USER) if c.title == title]
            col = existing[0] if existing else api.create_collection(title=title, namespace=USER, description=desc,
                                                                     private=not public)
            have = {i.item_id for i in api.get_collection(col.slug).items}
            for item, kind in items:
                if item not in have:
                    api.add_collection_item(col.slug, item_id=item, item_type=kind)
            made[title] = f"https://huggingface.co/collections/{col.slug}"
        print(json.dumps(made, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--collections", action="store_true")
    args = ap.parse_args()
    m, d = build_model(), build_dataset()
    print("built", m, d)
    if args.upload:
        upload(m, d, args.public, args.collections)


if __name__ == "__main__":
    main()
