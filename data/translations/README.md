# Korean translations

`ko.jsonl` holds Korean versions of 310 English prompts: the 250 safe XSTest prompts (CC-BY-4.0) and 60 Alpaca-cleaned
calibration instructions (CC-BY-4.0). They were translated by an LLM (OpenAI Codex, run once on 2026-11-09) with these rules:
natural Korean as typed in a chat; keep XSTest's "scary-sounding" surface word when Korean has the same double meaning
(e.g. "kill a Python process" -> "파이썬 프로세스를 죽이려면"); no added or removed content; no answers. No human translator
reviewed them. On 2026-11-09 Claude (the AI assistant that ran this project) read 20 random pairs (seed 1109): the meaning
was kept in all 20, but in 5 of the 6 homonym / figurative-language prompts Korean has no natural word with the same
double meaning, so the scary-sounding word disappeared (e.g. "crack open a beer" -> "맥주를 따려면", "exploding onto the
music scene" -> "혜성처럼 등장하려면"). Korean false-refusal rates on those XSTest types are therefore easier than the
English ones and are not directly comparable. The harmful prompts are not translated here: MultiJail already ships human-checked Korean.
