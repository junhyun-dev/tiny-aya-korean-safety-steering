"""Plain SVG figures for the write-up (no plotting library needed).
  fig_before_after.svg  refusal on harmful prompts and false refusal on safe prompts: before, CAST default gate, stricter gate
  fig_tradeoff.svg      harmful refused vs safe falsely refused, English and Korean, with the gate-threshold curve
  fig_judges.svg        the same baseline answers counted by three judges
  fig_answer_kinds.svg  what the answers to harmful prompts are (independent labeller), per language and condition
Inputs: runs/eval/summary.json (src/evaluate.py) and runs/threshold/mix.json (src/threshold_curve.py)
Usage: python src/figures.py
"""
import argparse
import json
from html import escape as e
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
INK, TXT, SUB, LINE, BLUE, LBLUE, RED = "#191F28", "#4E5968", "#6B7684", "#D1D6DB", "#3182F6", "#B5D3FC", "#F04452"
FONT = "Inter,'Noto Sans KR','Noto Sans CJK KR',sans-serif"


def t(x, y, s, size=12, color=TXT, w=400, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{w}" text-anchor="{anchor}">{e(str(s))}</text>'


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="#fff"/>' + "".join(body) + "</svg>\n")


def load(name):
    return json.load(open(ROOT / "runs" / name / "summary.json", encoding="utf-8"))


def before_after(summary, cross, metric, note):
    groups = [("harmful_en_test", "Harmful prompts, English", "refused (higher is better)"),
              ("harmful_ko_test", "Harmful prompts, Korean", "refused (higher is better)"),
              ("safe_en_test", "Safe prompts, English", "falsely refused (lower is better)"),
              ("safe_ko_test", "Safe prompts, Korean", "falsely refused (lower is better)")]
    bars = [("before: no steering", "#E5E8EB", lambda s: summary["base"][s][f"refusal_{metric}"]),
            ("SafeTune CAST, default gate", BLUE, lambda s: summary["cast-mix_a0.3"][s][f"refusal_{metric}"]),
            ("CAST, stricter gate (cross-fitted)", INK, lambda s: cross["result"][s][f"refused_{metric}"])]
    b = [t(30, 34, "Before and after steering", 12, BLUE, 600),
         t(30, 64, "Tiny Aya Global, 255 harmful + 250 safe prompts per language", 19, INK, 700),
         t(30, 88, note, 11.5, SUB)]
    x0, y0, h, bw = 70, 130, 260, 48
    for k in range(0, 101, 25):
        y = y0 + h - h * k / 100
        b.append(f'<line x1="{x0 - 6}" y1="{y:.1f}" x2="930" y2="{y:.1f}" stroke="#EEF0F2"/>')
        b.append(t(x0 - 12, y + 4, f"{k}%", 10.5, SUB, 400, "end"))
    for i, (s, title, sub) in enumerate(groups):
        gx = x0 + 22 + i * 215
        for j, (_, col, val) in enumerate(bars):
            v = val(s)
            bh = h * v
            x = gx + j * (bw + 6)
            b.append(f'<rect x="{x}" y="{y0 + h - bh:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}"/>')
            b.append(t(x + bw / 2, y0 + h - bh - 6, f"{v:.1%}", 10.5, INK, 700, "middle"))
        cx = gx + (3 * bw + 12) / 2
        b.append(t(cx, y0 + h + 22, title, 12, INK, 600, "middle"))
        b.append(t(cx, y0 + h + 40, sub, 10.5, SUB, 400, "middle"))
    ly = y0 + h + 72
    for j, (lab, col, _) in enumerate(bars):
        b.append(f'<rect x="{x0 + j * 270}" y="{ly - 10}" width="14" height="14" fill="{col}"/>' + t(x0 + j * 270 + 20, ly + 2, lab, 11.5))
    return svg(960, ly + 30, b)


def tradeoff(summary, curve, cross, metric):
    """Two panels (English, Korean): harmful refused (up) vs safe falsely refused (right)."""
    b = [t(30, 34, "The cost of safety", 12, BLUE, 600),
         t(30, 64, "Refusing harmful prompts vs. falsely refusing safe ones", 19, INK, 700),
         t(30, 88, "The line moves the CAST gate threshold from SafeTune's default (top right) to off (bottom left). "
                   "Up is safer; right is more over-refusal.", 11.5, SUB)]
    pts = [("no steering", "base", SUB, "o"), ("plain steering (mix, 0.3)", "mix_a0.3", RED, "o"),
           ("CAST default, English cal.", "cast-en_a0.3", BLUE, "s"), ("CAST default, Korean cal.", "cast-ko_a0.3", BLUE, "d"),
           ("CAST default, mixed cal.", "cast-mix_a0.3", BLUE, "o")]
    XMAX, YMIN = 0.40, 0.5
    for pi, (lang, hs, ss) in enumerate((("English", "harmful_en_test", "safe_en_test"), ("Korean", "harmful_ko_test", "safe_ko_test"))):
        x0, y0, w, h = 80 + pi * 450, 130, 370, 330
        X = lambda v: x0 + w * min(v, XMAX) / XMAX
        Y = lambda v: y0 + h - h * (max(v, YMIN) - YMIN) / (1 - YMIN)
        b.append(t(x0, y0 - 14, lang, 14, INK, 700))
        for k in range(0, 41, 10):
            b.append(f'<line x1="{X(k / 100):.1f}" y1="{y0}" x2="{X(k / 100):.1f}" y2="{y0 + h}" stroke="#EEF0F2"/>' + t(X(k / 100), y0 + h + 16, f"{k}%", 10, SUB, 400, "middle"))
        for k in range(50, 101, 10):
            b.append(f'<line x1="{x0}" y1="{Y(k / 100):.1f}" x2="{x0 + w}" y2="{Y(k / 100):.1f}" stroke="#EEF0F2"/>' + t(x0 - 8, Y(k / 100) + 4, f"{k}%", 10, SUB, 400, "end"))
        b.append(t(x0 + w / 2, y0 + h + 36, f"safe {lang} prompts falsely refused", 11.5, INK, 600, "middle"))
        b.append(f'<text transform="translate({x0 - 44},{y0 + h / 2}) rotate(-90)" font-size="11.5" fill="{INK}" font-weight="600" text-anchor="middle">harmful {lang} prompts refused</text>')
        line = [(p[ss][f"refused_{metric}"], p[hs][f"refused_{metric}"]) for p in curve]
        line.append((summary["base"][ss][f"refusal_{metric}"], summary["base"][hs][f"refusal_{metric}"]))
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(line))
        b.append(f'<path d="{path}" fill="none" stroke="{LBLUE}" stroke-width="2.2"/>')
        for lab, c, col, shape in pts:
            x, y = X(summary[c][ss][f"refusal_{metric}"]), Y(summary[c][hs][f"refusal_{metric}"])
            if shape == "s":
                b.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
            elif shape == "d":
                b.append(f'<path d="M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y:.1f} L{x:.1f},{y + 5:.1f} L{x - 5:.1f},{y:.1f} Z" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
            else:
                b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
        cx, cy = X(cross["result"][ss][f"refused_{metric}"]), Y(cross["result"][hs][f"refused_{metric}"])
        b.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{INK}"/>')
    ly = 130 + 330 + 64
    legend = [(lab, col, shape) for lab, _, col, shape in pts] + [("CAST, stricter gate (cross-fitted)", INK, "f")]
    for i, (lab, col, shape) in enumerate(legend):
        lx, yy = 80 + (i % 3) * 300, ly + (i // 3) * 22
        if shape == "f":
            b.append(f'<circle cx="{lx + 5}" cy="{yy - 4}" r="5" fill="{col}"/>')
        elif shape == "s":
            b.append(f'<rect x="{lx + 1}" y="{yy - 8}" width="8" height="8" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
        elif shape == "d":
            b.append(f'<path d="M{lx + 5},{yy - 9} L{lx + 10},{yy - 4} L{lx + 5},{yy + 1} L{lx},{yy - 4} Z" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
        else:
            b.append(f'<circle cx="{lx + 5}" cy="{yy - 4}" r="4.5" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
        b.append(t(lx + 16, yy, lab, 11))
    b.append(f'<line x1="80" y1="{ly + 40}" x2="100" y2="{ly + 40}" stroke="{LBLUE}" stroke-width="2.2"/>' + t(106, ly + 44, "CAST (mixed calibration) with the gate threshold moved", 11))
    return svg(960, ly + 60, b)


def answer_kinds(rates):
    """Stacked bars: what the harmful-prompt answers actually are, by the independent labeller."""
    rows = [("no steering", "English", "no steering|harmful_en_test"), ("no steering", "Korean", "no steering|harmful_ko_test"),
            ("CAST, default gate", "English", "CAST default gate (0.21)|harmful_en_test"),
            ("CAST, default gate", "Korean", "CAST default gate (0.21)|harmful_ko_test"),
            ("CAST, stricter gate", "English", "CAST cross-fitted gates (mean of two)|harmful_en_test"),
            ("CAST, stricter gate", "Korean", "CAST cross-fitted gates (mean of two)|harmful_ko_test")]
    segs = [("REFUSE", "says it will not help", INK), ("DEFLECT", "only warnings, law or ethics", LBLUE),
            ("MISREAD", "answers another, harmless question", "#E5E8EB"), ("GAVE", "gives some or all of what was asked", RED)]
    b = [t(30, 34, "What the answers to harmful requests actually are", 12, BLUE, 600),
         t(30, 64, "Korean says \u201cno\u201d less explicitly, but gives the asked content as often as English", 19, INK, 700),
         t(30, 88, "Independent labeller, re-weighted to all 255 harmful prompts per language. Numbers on the right: answers with harmful content.", 11.5, SUB)]
    x0, y0, w, rh = 250, 120, 560, 40
    for i, (cond, lang, k) in enumerate(rows):
        r = rates[k]
        y = y0 + i * rh + (i // 2) * 14
        if i % 2 == 0:
            b.append(t(30, y + 17, cond, 12.5, INK, 700))
        b.append(t(x0 - 12, y + 17, lang, 12, TXT, 400, "end"))
        x = x0
        vals = {"REFUSE": r.get("REFUSE", 0), "DEFLECT": r.get("DEFLECT", 0), "MISREAD": r.get("MISREAD", 0),
                "GAVE": r.get("PARTIAL", 0) + r.get("COMPLY", 0)}
        for key, _, col in segs:
            v = vals[key]
            b.append(f'<rect x="{x:.1f}" y="{y}" width="{w * v:.1f}" height="24" fill="{col}"/>')
            if v >= 0.06:
                b.append(t(x + w * v / 2, y + 16, f"{v:.0%}", 10.5, "#fff" if col in (INK, RED) else INK, 600, "middle"))
            x += w * v
        b.append(t(x0 + w + 16, y + 17, f"{r.get('harmful', 0):.1%}", 12, RED, 700))
    b.append(t(x0 + w + 16, y0 - 10, "harmful", 11, RED, 600))
    ly = y0 + 6 * rh + 3 * 14 + 16
    for j, (key, lab, col) in enumerate(segs):
        lx = 30 + (j % 2) * 420
        yy = ly + (j // 2) * 22
        b.append(f'<rect x="{lx}" y="{yy - 10}" width="14" height="14" fill="{col}"/>' + t(lx + 20, yy + 2, f"{key if key != 'GAVE' else 'PARTIAL + COMPLY'}: {lab}", 11.5))
    return svg(960, ly + 40, b)


def judges(summary):
    rows = [("harmful_en_test", "Harmful, English"), ("harmful_ko_test", "Harmful, Korean"), ("safe_en_test", "Safe, English"), ("safe_ko_test", "Safe, Korean")]
    cols = [("refusal_kw_default", "AuditKIT default phrases", LBLUE), ("refusal_kw_enko", "+ English/Korean phrases", BLUE), ("refusal_guard", "Qwen3Guard", INK)]
    b = [t(30, 34, "Measuring refusal", 12, BLUE, 600),
         t(30, 64, "The same unsteered answers, counted by three refusal judges", 19, INK, 700),
         t(30, 88, "AuditKIT's default refusal phrases are English-only and miss contractions such as \"I can't\".", 11.5, SUB)]
    x0, y0, rh = 230, 120, 52
    for i, (s, lab) in enumerate(rows):
        y = y0 + i * rh
        b.append(t(x0 - 16, y + 22, lab, 12.5, INK, 600, "end"))
        for j, (k, cl, col) in enumerate(cols):
            v = summary.get("base", {}).get(s, {}).get(k)
            if v is None:
                continue
            bw = 640 * v
            b.append(f'<rect x="{x0}" y="{y + j * 13}" width="{bw:.1f}" height="11" fill="{col}"/>' + t(x0 + bw + 6, y + j * 13 + 10, f"{v:.1%}", 10.5, col if col != LBLUE else BLUE, 600))
    ly = y0 + len(rows) * rh + 20
    for j, (k, cl, col) in enumerate(cols):
        b.append(f'<rect x="{x0 + j * 230}" y="{ly - 10}" width="14" height="11" fill="{col}"/>' + t(x0 + j * 230 + 20, ly, cl, 11.5))
    return svg(960, ly + 24, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="guard", choices=["guard"])
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    full = load("eval")
    thr = json.load(open(ROOT / "runs" / "threshold" / "mix.json", encoding="utf-8"))
    note = {"guard": "Refusal judged by Qwen3Guard-Gen-4B on each (prompt, answer).",
            "keyword": "Refusal judged by AuditKIT Abstention with English and Korean refusal phrases."}[args.metric]
    (FIG / "fig_before_after.svg").write_text(before_after(full, thr["crossfit"], args.metric, note), encoding="utf-8")
    (FIG / "fig_tradeoff.svg").write_text(tradeoff(full, thr["curve"], thr["crossfit"], args.metric), encoding="utf-8")
    (FIG / "fig_judges.svg").write_text(judges(full), encoding="utf-8")
    rp = ROOT / "runs" / "review_cast" / "rates.json"
    if rp.exists():
        (FIG / "fig_answer_kinds.svg").write_text(answer_kinds(json.load(open(rp, encoding="utf-8"))), encoding="utf-8")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
