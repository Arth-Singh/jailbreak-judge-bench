# Jailbreak Judge Bench

Can a judge tell a real jailbreak from a failed or fake one?

**Site:** https://arth-singh.github.io/jailbreak-judge-bench

Papers about jailbreak attacks report an attack success rate (ASR): the share of attacks that made a chatbot
give real help with a harmful request. A judge, often another AI model, decides which answers count as a
success. If the judge is wrong, the ASR in the paper is wrong.

We test judges on 2,649 answers that humans labeled, from 9 public sources. A good judge must pass three checks:

1. **Catches real jailbreaks**: the answer really gives what the harmful request asked for.
2. **Ignores failed attempts**: refusals, vague or partial answers, off-topic answers.
3. **Not fooled by fakes**: answers built to look like real help (steps, technical words, a confident tone)
   that are wrong or made up.

The **score** is the average of the three checks. The **ASR error** is how far the judge's attack success
rate is from the human one, in percentage points.

## Leaderboard

Universal set, 2,649 answers. Each judge uses its own built-in rule for "success"; we tuned no thresholds.

<!-- leaderboard:start -->
| # | Judge | Type | Score | Catches real jailbreaks | Ignores failed attempts | Not fooled by fakes | ASR error |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **Our judge** | Ours | 88.7 | 80.2% | 88.7% | 97.2% | 4.9 |
| 2 | GPT-6.1 sol-pro | API model, our question | 75.7 | 43.0% | 88.0% | 96.0% | 25.1 |
| 3 | Gemini 4 argon | API model, our question | 74.8 | 40.3% | 87.2% | 96.9% | 27.4 |
| 4 | GPT-4o | API model, our question | 74.3 | 52.8% | 89.5% | 80.5% | 20.1 |
| 5 | Qwen3.5 397B, untrained | Open model, our question | 72.2 | 25.0% | 91.7% | 100.0% | 32.4 |
| 6 | GPT-4o mini | API model, our question | 70.7 | 28.1% | 86.7% | 97.5% | 30.4 |
| 7 | Claude Opus 5.5 | API model, our question | 70.3 | 22.1% | 89.6% | 99.2% | 33.3 |
| 8 | Qwen3.8 27B, untrained | Open model, our question | 67.9 | 13.5% | 90.9% | 99.4% | 37.0 |
| 9 | HarmBench classifier | Safety classifier | 61.7 | 86.1% | 80.3% | 18.6% | 7.3 |
| 10 | JailbreakBench judge | Paper prompt | 61.6 | 73.2% | 84.5% | 27.1% | 11.6 |
| 11 | StrongREJECT, fine-tuned | Safety classifier | 59.4 | 72.1% | 77.8% | 28.2% | 10.5 |
| 12 | PAIR judge | Paper prompt | 57.9 | 43.5% | 92.8% | 37.3% | 21.4 |
| 13 | StrongREJECT rubric | Paper prompt | 55.7 | 85.8% | 66.2% | 15.3% | 18.6 |
| 14 | ShieldGemma 9B | Safety classifier | 55.5 | 38.8% | 77.8% | 50.0% | 24.5 |
| 15 | JailJudge | Safety classifier | 53.8 | 72.6% | 77.4% | 11.3% | 12.9 |
| 16 | WildGuard | Safety classifier | 51.3 | 79.0% | 65.7% | 9.0% | 19.3 |
| 17 | Llama Guard 4 | Safety classifier | 47.6 | 69.4% | 63.7% | 9.6% | 22.7 |
| 18 | Qwen3Guard 8B | Safety classifier | 46.8 | 86.0% | 50.4% | 4.0% | 27.1 |
| 19 | Llama Guard 3 | Safety classifier | 44.9 | 77.0% | 53.7% | 4.0% | 29.6 |
| 20 | Refusal keywords | Keyword rule | 35.8 | 80.8% | 24.5% | 2.0% | 35.6 |
<!-- leaderboard:end -->

Results for each source, and for 5 benchmarks we score but cannot share, are on the [site](https://arth-singh.github.io/jailbreak-judge-bench).

## Test your judge

1. Ask for access to the data on Hugging Face, then download it:

   ```bash
   huggingface-cli download ArthT/jailbreak-judge-bench --repo-type dataset --local-dir data
   ```

2. Run your judge on each row of `data/universal.jsonl` (it reads `request` and `response`) and write one line
   per row:

   ```json
   {"id": "jailmeter-0001", "prediction": 1, "score": 0.93}
   ```

   `prediction` is 1 when the answer delivers what was asked, else 0. `score` is optional: any number, higher
   meaning more likely a success. A CSV with the columns `id,prediction,score` works too.

3. Score it (Python only, no packages to install):

   ```bash
   python evaluate.py --data data/universal.jsonl --pred my_judge.jsonl
   ```

To check our numbers: `python evaluate.py --data data/universal.jsonl --pred predictions/`.

## What is in this repository

| Path | What it is |
|---|---|
| `evaluate.py` | Scores a predictions file: the three checks, the score, the ASR error, ranking (AUROC), per source. |
| `predictions/` | Every judge's verdict on every row (ids and scores only). |
| `results/` | The numbers behind the site and this page, as JSON and CSV. |
| `prompts.md` | The exact words of "our question", for the judges marked that way. |
| `docs/` | The site. |
| `scripts/build_site.py` | Rebuilds the site, this table and `results/leaderboard.csv` from `results/`. |

This repository holds no request or answer text. The data is on Hugging Face behind an access form, because
it holds real harmful answers.

## The data

| Source | Answers | License |
|---|---:|---|
| [JailMeter](https://github.com/Magi2B0y/JailMeter) | 330 | Apache-2.0 |
| [SecureBreak](https://github.com/VIGNESH-KUMAR-KEMBU/SecureBreak) | 300 | CC BY 4.0 |
| [HarmMetric Eval](https://huggingface.co/datasets/Abel-24/HarmMetric_Eval) | 300 | MIT; rows from PKU-SafeRLHF are CC BY-NC 4.0 |
| [VENOM](https://github.com/qzqdz/venom) | 135 | MIT (per its README) |
| [StrongREJECT](https://github.com/dsbowen/strong_reject) | 300 | MIT |
| [WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) | 384 | ODC-By, under AI2's Responsible Use Guidelines |
| [Do-Not-Answer](https://github.com/Libr-AI/do-not-answer) | 300 | CC BY-NC-SA 4.0 |
| [Coin Flip for Safety](https://huggingface.co/datasets/ASSELab/CoinflipForSafety) | 300 | MIT |
| Fake answers (ours) | 300 | CC BY 4.0 |

Each row keeps its source's own human label. We took up to 300 answers per source (all of them for a source
of 400 or fewer) and kept each source's share of successes. Please cite the sources you use.

## License

Code: MIT. Data: each source keeps its own license (table above).

## Citation

The paper is coming soon.
