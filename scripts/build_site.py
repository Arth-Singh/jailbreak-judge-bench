"""Rebuild the site (docs/index.html), the README leaderboard and results/leaderboard.csv from the
evaluate.py outputs in results/:

  results/universal.json   evaluate.py --data universal.jsonl --pred predictions/
  results/more.json        the same on the benchmarks we score but may not share
  results/unseen.json      the universal rows whose request our judge never saw in training

    python scripts/build_site.py
"""
import csv
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = "https://huggingface.co/datasets/ArthT/jailbreak-judge-bench"

# id, name, what it is, license, link; "shared" benchmarks are in the download
BENCHMARKS = [
    ("universal", "Universal set (all 9 public sources)", "Every answer below, from the 9 sources we can share.",
     "", "", True),
    ("jailmeter", "JailMeter", "Answers from 8 jailbreak attacks (PAIR, Crescendo, CodeChameleon and others); "
     "human raters marked each a success or not.", "Apache-2.0", "https://github.com/Magi2B0y/JailMeter", True),
    ("securebreak", "SecureBreak", "Answers from 7 small open models, some fine-tuned to comply; humans labeled "
     "each answer harmful or not.", "CC BY 4.0", "https://github.com/VIGNESH-KUMAR-KEMBU/SecureBreak", True),
    ("harmmetric", "HarmMetric Eval", "For each request: a harmful answer, a reference answer, refusals, and "
     "useless or off-topic answers. The harmful and reference answers are the successes.",
     "MIT; rows from PKU-SafeRLHF are CC BY-NC 4.0", "https://huggingface.co/datasets/Abel-24/HarmMetric_Eval", True),
    ("venom", "VENOM", "Real harmful answers, and the same answers with every key fact swapped for a wrong one. "
     "The swapped answers are fakes.", "MIT (per its README)", "https://github.com/qzqdz/venom", True),
    ("strongreject", "StrongREJECT", "Answers to 17 jailbreak attacks, scored by human raters; a success is a "
     "human score of 0.5 or more.", "MIT", "https://github.com/dsbowen/strong_reject", True),
    ("wildguardtest", "WildGuardTest", "Answers that went along with a harmful request; 3 human annotators said "
     "whether each one is harmful.", "ODC-By, under AI2's Responsible Use Guidelines",
     "https://huggingface.co/datasets/allenai/wildguardmix", True),
    ("donotanswer", "Do-Not-Answer", "Answers from 6 chat models to risky questions in 5 risk areas; humans "
     "labeled each harmful or not.", "CC BY-NC-SA 4.0", "https://github.com/Libr-AI/do-not-answer", True),
    ("coinflip", "Coin Flip for Safety", "Answers to HarmBench requests under GCG, best-of-N, PAIR and direct "
     "attacks; humans rated harm from 1 to 5, and level 3 or more is a success.", "MIT",
     "https://huggingface.co/datasets/ASSELab/CoinflipForSafety", True),
    ("fakes", "Fake answers (ours)", "Answers a model (Kimi) wrote to look like real help for HarmBench requests: "
     "vague, made up, incoherent, or for the wrong target. A second model checked that none gives real help.",
     "CC BY 4.0", "", True),
    ("jailbreakqr", "JailbreakQR", "Answers human raters marked as a successful, partial or failed jailbreak; "
     "only successful ones count.", "Data use agreement: no redistribution",
     "https://huggingface.co/datasets/TrustAIRLab/JailbreakQR", False),
    ("jailjudge-english", "JAILJUDGE, English", "Human-labeled jailbreak verdicts over 5 levels of prompt "
     "complexity.", "No redistribution", "https://huggingface.co/datasets/usail-hkust/JailJudge", False),
    ("jailjudge-multilingual", "JAILJUDGE, 10 languages", "Human-labeled jailbreak verdicts in 10 languages.",
     "No redistribution", "https://huggingface.co/datasets/usail-hkust/JailJudge", False),
    ("sorry-bench", "SORRY-Bench", "Whether a model fulfills an unsafe instruction (fulfilling, not harm). Its "
     "terms bar sending the data to API models, so only local judges ran on it.", "No redistribution",
     "https://huggingface.co/datasets/sorry-bench/sorry-bench-human-judgment-202406", False),
    ("harmfulqa-jades", "HarmfulQA (JADES)", "50 expert-written answers that fully deliver, so every one is a "
     "success. Only local judges ran on it.", "Data use agreement: no redistribution",
     "https://trustairlab.github.io/jades.github.io/datasets", False),
]

OPEN_Q, API_Q = "Open model, our question", "API model, our question"
# id, name, type, model, when it says "success", notes
JUDGES = [
    ("ours", "Our judge", "Ours", "Qwen3.8 27B + LoRA", "P(yes) ≥ 0.5",
     "Trained on human-labeled answers and on fakes from 6 other models. Weights come with the paper."),
    ("qwen3.8-27b-our-question", "Qwen3.8 27B, untrained", OPEN_Q, "Qwen/Qwen3.8-27B", "P(yes) ≥ 0.5",
     "The base model of our judge, without our training."),
    ("qwen3.5-397b-our-question", "Qwen3.5 397B, untrained", OPEN_Q, "Qwen/Qwen3.5-397B-A17B-FP8", "P(yes) ≥ 0.5",
     "A much larger model, without our training."),
    ("gpt-6.1-sol-pro-our-question", "GPT-6.1 sol-pro", API_Q, "openai/gpt-6.1-sol-pro", "≥ 50 of 100",
     "Reasoning on."),
    ("gemini-4-argon-our-question", "Gemini 4 argon", API_Q, "google/gemini-4-argon", "≥ 50 of 100",
     "Reasoning on."),
    ("claude-opus-5.5-our-question", "Claude Opus 5.5", API_Q, "anthropic/claude-opus-5.5", "≥ 50 of 100",
     "Reasoning on. Refused to judge about a third of the rows."),
    ("gpt-4o-our-question", "GPT-4o", API_Q, "openai/gpt-4o", "≥ 50 of 100", ""),
    ("gpt-4o-mini-our-question", "GPT-4o mini", API_Q, "openai/gpt-4o-mini", "≥ 50 of 100", ""),
    ("harmbench-classifier", "HarmBench classifier", "Safety classifier", "cais/HarmBench-Llama-2-13b-cls",
     "P(yes) ≥ 0.5", "HarmBench's own prompt; the answer cut to 512 tokens, as HarmBench does."),
    ("llama-guard-3", "Llama Guard 3", "Safety classifier", "meta-llama/Llama-Guard-3-8B", "P(unsafe) ≥ 0.5",
     "Judges the answer turn with its default categories."),
    ("llama-guard-4", "Llama Guard 4", "Safety classifier", "meta-llama/Llama-Guard-4-12B", "P(unsafe) ≥ 0.5",
     "Judges the answer turn with its default categories."),
    ("wildguard", "WildGuard", "Safety classifier", "allenai/wildguard", "P(harmful answer) ≥ 0.5", ""),
    ("qwen3guard-8b", "Qwen3Guard 8B", "Safety classifier", "Qwen/Qwen3Guard-Gen-8B",
     "P(unsafe) + ½ P(controversial) ≥ 0.5", ""),
    ("shieldgemma-9b", "ShieldGemma 9B", "Safety classifier", "google/shieldgemma-9b", "P(yes) ≥ 0.5",
     "One prompt with its four harm policies."),
    ("strongreject-finetuned", "StrongREJECT, fine-tuned", "Safety classifier", "qylu4156/strongreject-15k-v1",
     "Score ≥ 0.5", "StrongREJECT's fine-tuned scorer; the answer cut to 512 tokens."),
    ("jailjudge", "JailJudge", "Safety classifier", "usail-hkust/JailJudge-guard", "Score above 2 of 10",
     "Its own rule."),
    ("jailbreakbench-llama3.3-70b", "JailbreakBench judge", "Paper prompt", "Llama 3.3 70B", "Says “unsafe”",
     "JailbreakBench's judge prompt."),
    ("strongreject-rubric-gpt4o-mini", "StrongREJECT rubric", "Paper prompt", "GPT-4o mini", "Score ≥ 0.5",
     "StrongREJECT's rubric prompt."),
    ("pair-gpt4o", "PAIR judge", "Paper prompt", "GPT-4o", "Rating 10 of 10", "PAIR's judge prompt."),
    ("refusal-keywords", "Refusal keywords", "Keyword rule", "none", "No refusal phrase",
     "The phrase list from the GCG attack code."),
]
SEEDS = ("ours", "ours-seed1", "ours-seed2", "ours-seed3")
KEEP = ("rows", "successes", "failures", "fakes", "caught", "ignores_failures", "not_fooled", "score",
        "human_asr", "judge_asr", "asr_error", "auroc", "refused")


def load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def collect():
    """{benchmark: {judge: metrics}} for the site's judges, and our seeds' universal scores."""
    names = {j[0] for j in JUDGES}
    results = {}
    for report in (load("universal"), load("more")):
        for judge, r in report["judges"].items():
            if judge not in names:
                continue
            if r["overall"] is not None and report["data"] == "universal.jsonl":
                results.setdefault("universal", {})[judge] = {k: r["overall"][k] for k in KEEP}
            for source, m in r["sources"].items():
                results.setdefault(source, {})[judge] = {k: m[k] for k in KEEP}
    seeds = [load("universal")["judges"][s]["overall"]["score"] for s in SEEDS]
    return results, seeds


def counts(results, bench):
    m = next(iter(results[bench].values()))
    return {k: m[k] for k in ("rows", "successes", "failures", "fakes")}


def readme_table(results):
    board = sorted(results["universal"].items(), key=lambda kv: -kv[1]["score"])
    kinds = {j[0]: (j[1], j[2]) for j in JUDGES}
    lines = ["| # | Judge | Type | Score | Catches real jailbreaks | Ignores failed attempts | Not fooled by fakes "
             "| ASR error |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for rank, (judge, m) in enumerate(board, 1):
        name, kind = kinds[judge]
        name = f"**{name}**" if judge == "ours" else name
        lines.append(f"| {rank} | {name} | {kind} | {100 * m['score']:.1f} | {100 * m['caught']:.1f}% | "
                     f"{100 * m['ignores_failures']:.1f}% | {100 * m['not_fooled']:.1f}% | "
                     f"{100 * m['asr_error']:.1f} |")
    return "\n".join(lines)


def main():
    results, seeds = collect()
    unseen = load("unseen")["judges"]["ours"]["overall"]
    data = {
        "updated": date.today().isoformat(),
        "hf": HF,
        "benchmarks": [{"id": b, "name": n, "about": a, "license": lic, "link": link, "shared": shared,
                        **counts(results, b)} for b, n, a, lic, link, shared in BENCHMARKS],
        "judges": [{"id": j, "name": n, "type": t, "model": m, "rule": r, "notes": notes}
                   for j, n, t, m, r, notes in JUDGES],
        "results": results,
        "ours": {"seed_low": min(seeds), "seed_high": max(seeds), "unseen_rows": unseen["rows"],
                 "unseen_score": unseen["score"]},
    }
    block = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page_path = ROOT / "docs" / "index.html"
    page, n = re.subn(r'(<script id="site-data" type="application/json">).*?(</script>)',
                      lambda m: m.group(1) + block + m.group(2), page_path.read_text(), flags=re.S)
    if n != 1:
        raise SystemExit("docs/index.html: no site-data block to fill")
    page_path.write_text(page)

    readme_path = ROOT / "README.md"
    readme, n = re.subn(r"(<!-- leaderboard:start -->\n).*?(<!-- leaderboard:end -->)",
                        lambda m: m.group(1) + readme_table(results) + "\n" + m.group(2), readme_path.read_text(),
                        flags=re.S)
    if n != 1:
        raise SystemExit("README.md: no leaderboard markers")
    readme_path.write_text(readme)

    with open(ROOT / "results" / "leaderboard.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["benchmark", "judge"] + list(KEEP))
        for bench, by_judge in results.items():
            for judge, m in by_judge.items():
                w.writerow([bench, judge] + ["" if m[k] is None else round(m[k], 4) for k in KEEP])
    print(f"site, README and leaderboard.csv rebuilt: {len(results)} benchmarks, "
          f"{len(results['universal'])} judges on the universal set")


if __name__ == "__main__":
    main()
