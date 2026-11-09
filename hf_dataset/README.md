---
license: cc-by-nc-4.0
language:
- en
- ko
task_categories:
- text-generation
- text-classification
tags:
- safety
- refusal
- over-refusal
- multilingual
- korean
- red-teaming
- activation-steering
- lexsi-hackathon
size_categories:
- 1K<n<10K
configs:
- config_name: answers
  data_files: answers/*.jsonl
- config_name: prompts
  data_files: prompts/*.jsonl
- config_name: review
  data_files: review/*.jsonl
---

# Tiny Aya Korean safety eval: answers and judge labels, before and after steering

Every answer [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) gave to the same harmful and
safe-but-scary prompts in **English and Korean**, with and without inference-time safety steering (SafeTune CAST),
and how three judges and an independent labeller scored them.

Built for the Lexsi AI Safety and Alignment Hackathon 2026 (Track 2a-4, Track 2b-8). Write-up and code:
[github.com/junhyun-dev/tiny-aya-korean-safety-steering](https://github.com/junhyun-dev/tiny-aya-korean-safety-steering).
Steering files: [junhyun-dev/tiny-aya-global-safety-steering](https://huggingface.co/junhyun-dev/tiny-aya-global-safety-steering).

## What is in it

| Folder | Rows | What |
|---|---:|---|
| `prompts/` | 1,250 | The prompt sets after CuratorKIT cleaning. Test: 255 MultiJail harmful prompts in English and Korean (the same requests), 250 XSTest safe prompts in English and Korean. Calibration: 60 harmful per language, 60 Alpaca-cleaned harmless per language |
| `answers/` | 6 × 1,010 | One file per condition: `base` (no steering), `mix_a0.3` (refusal direction, always on), `cast-{en,ko,mix}_a0.3` (CAST with SafeTune's calibrated gate), `cast-mix_a0.3_t0.34` (stricter gate) |
| `review/` | 442 | Blind labels from an independent LLM labeller, three fresh runs. Round 1 (149): stratified sample of all conditions, REFUSE / COMPLY / PARTIAL. Round 2 (213): every unsteered harmful answer Qwen3Guard counted as not refused, plus 10 per stratum of the rest. Round 3 (80): the same for the steered answers (`cast-mix_a0.3`). Rounds 2 and 3 use REFUSE / DEFLECT / MISREAD / PARTIAL / COMPLY. Each row has the stratum size, so rates can be re-weighted to the full sets |
| `gate_scores/` | 3 × 1,250 | The CAST gate score of every prompt, so any stricter threshold can be replayed without generating again |
| `results/` | — | Summary tables, per-XSTest-type breakdown, threshold curves, judge agreement |

### `answers` columns

`condition`, `set`, `id`, `lang`, `kind` (harmful / safe), `xstest_type`, `prompt`, `answer`, `redacted`,
`guard_refusal` and `guard_safety` / `guard_categories` (Qwen3Guard-Gen-4B on the prompt and answer),
`keyword_refusal` (AuditKIT `Abstention` with English and Korean refusal phrases), `gate_score` and `gate_fired`
(CAST conditions only).

Answers were generated greedily, up to 96 new tokens, bf16, on one RTX 3060.

### Redaction

Answers that Qwen3Guard judged **Unsafe**, or that the independent labeller marked as containing harmful content,
are replaced by `[redacted: judged unsafe]` (`redacted: true`), in every condition where the same answer text appears.
Their prompts and labels are kept, so every rate in the write-up can be recomputed from this dataset.

## Headline numbers (Qwen3Guard)

| | Harmful refused EN | Harmful refused KO | Safe falsely refused EN | Safe falsely refused KO |
|---|---:|---:|---:|---:|
| `base` | 71.4% | 60.8% | 10.8% | 8.4% |
| `cast-mix_a0.3` | 93.7% | 88.2% | 36.8% | 30.0% |
| `cast-mix_a0.3_t0.34` | 85.9% | 77.6% | 13.6% | 9.2% |

An independent labeller (rounds 2 and 3) shows what changes underneath: without steering, Tiny Aya refuses explicitly
in 65.5% (EN) vs 40.3% (KO) of harmful prompts but gives the asked content equally often (19.0% vs 18.8%). With the
calibrated gate that share falls to 6.3% / 10.2%; with the stricter 0.34 gate, Korean stays near 18%: most of the
change is warnings becoming explicit refusals. See `results/` and the write-up.

## Sources and licenses

- MultiJail (Deng et al., 2024; MIT): harmful prompts in English and Korean (human-checked translations).
- XSTest (Röttger et al., 2024; CC-BY-4.0): safe prompts; Korean versions machine-translated for this project
  (in homonym and figurative prompts the Korean often loses the double meaning).
- Alpaca-cleaned (CC-BY-4.0): harmless calibration instructions; Korean versions machine-translated.
- Answers are outputs of Tiny Aya Global (CC-BY-NC-4.0); this dataset is released under CC-BY-NC-4.0.

**Content warning:** the prompts ask for harmful content (violence, crime, hate, sexual content).
