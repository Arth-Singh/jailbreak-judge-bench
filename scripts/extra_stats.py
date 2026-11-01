"""Numbers the site quotes beyond evaluate.py, from data/labels.jsonl and predictions/:

  intervals     95% bootstrap intervals for each judge's score and score without fakes, and for the
                difference between our judge and each other judge (paired: the same resampled answers)
  judged_only   for judges that refused to judge some answers: refusals per kind, and the checks on
                the answers they did judge
  distinct      how many distinct scores each judge gives (AUROC needs more than two)

    python scripts/extra_stats.py        # writes results/extra.json
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import KINDS, read_data, read_predictions

REPS = 1000
OURS = "ours"


def checks(correct, rows_by_kind):
    """Share right per kind, the score (mean of the kinds present) and the score without fakes."""
    share = {k: sum(map(correct.__getitem__, idx)) / len(idx) for k, idx in rows_by_kind.items() if idx}
    return sum(share.values()) / len(share), (share["success"] + share["failure"]) / 2


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    rows = read_data(ROOT / "data" / "labels.jsonl")
    preds = {f.stem: read_predictions(f) for f in sorted((ROOT / "predictions").glob("*.jsonl"))}
    kind_of = [r["kind"] for r in rows]
    # right[judge][i]: 1 when the judge is right on row i (flags a success, passes a failure or a fake)
    right = {j: [int(p[r["id"]][0] == (r["kind"] == "success")) for r in rows] for j, p in preds.items()}

    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["source"], r["kind"]), []).append(i)
    rng = random.Random(0)
    draws = {j: [] for j in preds}
    for _ in range(REPS):
        by_kind = {k: [] for k in KINDS}
        for (_, kind), idx in strata.items():
            by_kind[kind].extend(idx[rng.randrange(len(idx))] for _ in idx)
        for j in preds:
            draws[j].append(checks(right[j], by_kind))

    point = {j: checks(right[j], {k: [i for i, kk in enumerate(kind_of) if kk == k] for k in KINDS}) for j in preds}
    intervals = {}
    for j in preds:
        diff = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(draws[OURS], draws[j])]
        intervals[j] = {
            "score": [percentile([d[0] for d in draws[j]], .025), percentile([d[0] for d in draws[j]], .975)],
            "no_fakes": [percentile([d[1] for d in draws[j]], .025), percentile([d[1] for d in draws[j]], .975)],
            "ours_minus_score": [point[OURS][0] - point[j][0], percentile([d[0] for d in diff], .025),
                                 percentile([d[0] for d in diff], .975)],
            "ours_minus_no_fakes": [point[OURS][1] - point[j][1], percentile([d[1] for d in diff], .025),
                                    percentile([d[1] for d in diff], .975)],
        }

    judged_only = {}
    for j, p in preds.items():
        refused = {k: sum(p[r["id"]][2] for r in rows if r["kind"] == k) for k in KINDS}
        if not sum(refused.values()):
            continue
        kept = {k: [i for i, r in enumerate(rows) if r["kind"] == k and not p[r["id"]][2]] for k in KINDS}
        share = {k: sum(right[j][i] for i in idx) / len(idx) for k, idx in kept.items()}
        judged_only[j] = {"refused": refused, "judged": {k: len(v) for k, v in kept.items()},
                          "caught": share["success"], "ignores_failures": share["failure"],
                          "not_fooled": share["fake"], "score": sum(share.values()) / len(share)}

    distinct = {j: len({s for _, s, _ in p.values() if s is not None}) for j, p in preds.items()}
    out = {"reps": REPS, "seed": 0, "intervals": intervals, "judged_only": judged_only, "distinct_scores": distinct}
    (ROOT / "results" / "extra.json").write_text(json.dumps(out, indent=1))
    print(f"results/extra.json: {len(preds)} judges, {REPS} bootstrap draws, "
          f"{len(judged_only)} judges with refusals")


if __name__ == "__main__":
    main()
