"""Compare Higgs heads on identical truth and reconstruct each training loss's target mix."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import json
import sys

import numpy as np
import uproot

from paths import ANCESTRY_RUN, RUN, TREE_RUN

sys.path.insert(0, str(RUN / "frozen/plots"))
from process_catalog import GROUPS, process_code_map, parent_processes

I, J = np.triu_indices(12, 1)
OBJECT = {"lepton": np.arange(12) < 3, "jet": np.arange(12) >= 3}
PAIR = {"ll": J < 3, "lj": (I < 3) & (J >= 3), "jj": I >= 3}
NAMES = ("HH", *GROUPS.values())
BINS = 4096


def reader(stack, path):
    return stack.enter_context(uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None))["tree"]


def targets(a, valid):
    anc = a["ancestryHiggs"]
    mask = valid[:, I] & valid[:, J]
    if "higgsPairKnown" in a:
        known = mask & (((a["higgsPairKnown"][:, I] >> J) & 1) == 1)
        same = ((a["higgsPairSame"][:, I] >> J) & 1) == 1
    else:
        ids = a["relationHiggs"]
        known = mask & (ids[:, I] >= 0) & (ids[:, J] >= 0)
        same = (ids[:, I] > 0) & (ids[:, I] == ids[:, J])
    return {"ancestry": (valid & (anc >= 0), anc == 1), "pair": (known, same)}


def group_map():
    parents = parent_processes(RUN / "dataset/summary.json")
    return process_code_map(parents, parents, ("hhml", *GROUPS.values()))


def groups(values, mapping):
    ids, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[k] for k in ids], dtype=np.int8)[inverse]


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_pair_tradeoff(rows, rankings, folder, fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    kinds = ("ll", "lj", "jj")
    names = ("Lepton–lepton", "Lepton–jet", "Jet–jet")
    y = np.arange(len(kinds))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 3.7), sharey=True)
    for model, label, offset, color in (("previous", "Previous", -.16, "#4879a7"), ("current", "Current", .16, "#ae5037")):
        weights = [sum(r["absolute_logit_gradient_share"] for r in rows if r["model"] == model
                       and r["task"] == "pair" and r["objects"] == k) for k in kinds]
        left.barh(y+offset, weights, .28, color=color, label=label)
    left.set_xlim(0, 1)
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlabel("Share of total |pair BCE logit gradient|")
    left.legend(frameon=False)
    points = {r["objects"]: r for r in rankings if r["reference"] == "tree_HH"}
    gain = [100*points[k]["paired_accuracy_change"] for k in kinds]
    errors = [100*points[k]["paired_mc_error"] for k in kinds]
    right.errorbar(gain, y, xerr=errors, color="#253d50", fmt="o", capsize=3)
    right.axvline(0, color=".5", lw=.9)
    right.set_xlim(min(gain)-4, max(gain)+5)
    right.set_xlabel("Change in correct best-pair rate (percentage points)")
    left.set_yticks(y, names)
    left.invert_yaxis()
    left.set_title("Higgs pair supervision across all processes")
    right.set_title("Within-event choice in test HH")
    for ax in (left, right):
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"Higgs pair prediction  ·  Fold {fold} test")
    fig.text(.115, .035, "The gradient share uses each fitted model’s own truth. Accuracy uses identical tree-resolved candidates and paired MC errors.", fontsize=8.7)
    fig.tight_layout(rect=[0, .08, 1, .96])
    fig.savefig(folder / "higgs_pair_tradeoff.pdf")
    plt.close(fig)


def training_mix(fold, folder, lookup):
    rem = (fold + 2) % 3
    sow = np.array(json.loads((RUN / "dataset/summary.json").read_text())["sum_abs_weights_by_remainder"][rem])
    factors = sow[1] / sow
    sums = defaultdict(lambda: np.zeros((2, len(NAMES))))
    for src, run in (("previous", ANCESTRY_RUN), ("current", RUN)):
        with ExitStack() as stack:
            t = reader(stack, run / f"dataset/3fold_{rem}.root")
            fields = ["valid", "isMET", "ancestryHiggs", "weight", "class_id", "dsid"]
            fields += ["relationHiggs"] if src == "previous" else ["higgsPairKnown", "higgsPairSame"]
            for a in t.iterate(fields, step_size=24000, library="np"):
                valid = a["valid"].astype(bool) & ~a["isMET"].astype(bool)
                w = np.abs(a["weight"].astype("f8")) * factors[a["class_id"]]
                g = groups(a["dsid"], lookup)
                for task, (known, same) in targets(a, valid).items():
                    unit = w / np.maximum(1, known.sum(1))
                    kinds = PAIR if task == "pair" else OBJECT
                    for kind, cols in kinds.items():
                        for truth, mask in enumerate((known & ~same, known & same)):
                            value = mask[:, cols].sum(1) * unit
                            sums[src, task, kind][truth] += np.bincount(g, weights=value, minlength=len(NAMES))
    rows = []
    for src in ("previous", "current"):
        for task, kinds in (("ancestry", OBJECT), ("pair", PAIR)):
            total = sum(sums[src, task, kind].sum() for kind in kinds)
            for kind in kinds:
                for code, name in enumerate(NAMES):
                    neg, pos = sums[src, task, kind][:, code]
                    rows.append(dict(run=src, task=task, objects=kind, process=name,
                                     negative_share=float(neg / total), positive_share=float(pos / total)))
    csv_write(folder / "training_higgs_loss_weight.csv", rows)


class Histogram:
    def __init__(self):
        self.h = np.zeros((2, BINS), dtype=np.float64)
        self.s = np.zeros((2, 2), dtype=np.float64)
        self.count = np.zeros(2, dtype=np.int64)

    def update(self, score, truth, weight):
        if not len(score):
            return
        if not np.all((score >= 0) & (score <= 1)):
            raise ValueError("Invalid probabilities on a supervised physical object or pair")
        bins = np.minimum(BINS-1, (BINS * score).astype(int))
        self.h += np.bincount(BINS*truth.astype(int) + bins, weights=weight, minlength=2*BINS).reshape(2, BINS)
        self.count += np.bincount(truth, minlength=2)
        self.s[0] += np.bincount(truth, weights=weight*score, minlength=2)
        p = np.clip(score.astype("f8"), 1e-7, 1-1e-7)
        nll = np.where(truth, -np.log(p), -np.log1p(-p))
        self.s[1] += np.bincount(truth, weights=weight*nll, minlength=2)

    def metrics(self):
        neg, pos = self.h
        den = self.h.sum(1)
        auc = (pos*(neg.cumsum() - neg/2)).sum() / den.prod() if den.prod() else None
        means = [self.s[0, i]/den[i] if den[i] else None for i in (0, 1)]
        return dict(negatives=int(self.count[0]), positives=int(self.count[1]), mean_score_negative=means[0],
                    mean_score_positive=means[1], auc=auc, bce=self.s[1].sum()/den.sum() if den.sum() else None)


def head_comparison(fold, folder, lookup):
    hists = defaultdict(Histogram)
    residuals = defaultdict(lambda: np.zeros((3, len(NAMES)), dtype="f8"))
    train_rem = (fold + 2) % 3
    train_sow = np.array(json.loads((RUN / "dataset/summary.json").read_text())["sum_abs_weights_by_remainder"][train_rem])
    class_factors = train_sow[1] / train_sow
    # Compare the best candidate inside each event, restricting to nontrivial choices.
    assignments = defaultdict(lambda: np.zeros(9, dtype="f8"))
    with ExitStack() as stack:
        scores = [reader(stack, run / f"output/fold_{fold}/test/test_scores.root") for run in (ANCESTRY_RUN, RUN)]
        tree = reader(stack, TREE_RUN / f"dataset/3fold_{fold}.root")
        if any(t.num_entries != tree.num_entries for t in scores):
            raise ValueError("Score row counts differ")
        id_fields = ["eventNumber", "dsid"]
        pred = ["p_pair_higgs", "p_ancestry_higgs"]
        old_truth, new_truth = ["ancestryHiggs", "relationHiggs"], ["ancestryHiggs", "higgsPairKnown", "higgsPairSame"]
        for start in range(0, tree.num_entries, 18000):
            kw = dict(entry_start=start, entry_stop=min(start+18000, tree.num_entries), library="np")
            a = scores[0].arrays([*id_fields, *pred, *old_truth], **kw)
            b = scores[1].arrays([*id_fields, *pred, *new_truth, "weight", "class_id", "valid", "isMET", "wAssignment"], **kw)
            c = tree.arrays([*id_fields, *new_truth], **kw)
            if any(not np.array_equal(a[k], b[k]) or not np.array_equal(a[k], c[k]) for k in id_fields):
                raise ValueError("Score rows and truth rows have different identities")
            w = np.abs(b["weight"].astype("f8"))
            g = groups(a["dsid"], lookup)
            valid = b["valid"].astype(bool) & ~b["isMET"].astype(bool)
            previous, current, hard = [targets(d, valid) for d in (a, b, c)]
            for task, kinds in (("ancestry", OBJECT), ("pair", PAIR)):
                pknown, psame = previous[task]
                cknown, csame = current[task]
                tknown, tsame = hard[task]
                field = "p_" + task + "_higgs"
                # Fractional contributions to BCE and its absolute derivative w.r.t. each logit.
                # Use the model's own truth, actual per-event normalization, and training class balance.
                for model, values, known, truth in (("previous", a, pknown, psame), ("current", b, cknown, csame)):
                    weight = w * class_factors[b["class_id"]] / np.maximum(1, known.sum(1))
                    prob = values[field]
                    residual = np.where(truth, 1. - prob, prob)
                    for kind, cols in kinds.items():
                        for code, choose in ((0, known & ~truth), (1, known & truth)):
                            mask = choose[:, cols]
                            error = np.where(mask, residual[:, cols], 0.).astype("f8")
                            quantities = (mask.sum(1), error.sum(1), -np.log1p(-np.minimum(error, 1.-1e-7)).sum(1))
                            dest = residuals[model, task, kind, code]
                            for idx, q in enumerate(quantities):
                                dest[idx] += np.bincount(g, weights=weight*q, minlength=len(NAMES))
                selections = {
                    "tree_all": (tknown, tsame), "tree_HH": (tknown & (g == 0)[:, None], tsame),
                    "tree_singleH": (tknown & (g == 7)[:, None], tsame),
                    "tree_Wmatched_HH": (tknown & ((g == 0) & b["wAssignment"].astype(bool))[:, None], tsame),
                    "previous_HH": (pknown & (g == 0)[:, None], psame),
                    "label_agreement_HH": (tknown & pknown & (tsame == psame) & (g == 0)[:, None], tsame),
                    "fallback_HH": (cknown & ~tknown & (g == 0)[:, None], csame)}
                for reference, (known, truth) in selections.items():
                    for kind, cols in kinds.items():
                        allowed = known & cols
                        row, column = np.nonzero(allowed)
                        if not len(row):
                            continue
                        y, wt = truth[row, column].astype(np.int8), w[row]
                        for name, data in (("previous", a), ("current", b)):
                            hists[reference, task, kind, name].update(data[field][row, column], y, wt)
                        if task == "pair" and reference in ("tree_HH", "tree_Wmatched_HH", "tree_singleH"):
                            positive = allowed & truth
                            use = positive.any(1) & (allowed & ~truth).any(1)
                            if use.any():
                                weight = w[use]
                                success = []
                                for data in (a, b):
                                    best = np.argmax(np.where(allowed[use], data[field][use], -np.inf), axis=1)
                                    success.append(positive[use][np.arange(len(best)), best].astype("f8"))
                                prev, cur = success
                                delta = cur - prev
                                chance = positive[use].sum(1) / allowed[use].sum(1)
                                assignments[reference, kind] += np.array([
                                    len(weight), weight.sum(), (weight*chance).sum(), (weight*prev).sum(),
                                    (weight*cur).sum(), (weight**2).sum(), (weight**2*delta).sum(),
                                    (weight**2*delta**2).sum(), (prev != cur).sum()])
    rows = [dict(reference=ref, task=task, objects=kind, model=model, **h.metrics())
            for (ref, task, kind, model), h in sorted(hists.items())]
    csv_write(folder / "common_truth_heads.csv", rows)
    rankings = []
    for (ref, kind), (n, den, chance, prev, cur, sw2, wd, wd2, flips) in sorted(assignments.items()):
        delta = (cur-prev) / den
        err = np.sqrt(max(0., wd2 - 2*delta*wd + delta**2*sw2)) / den
        rankings.append(dict(reference=ref, objects=kind, eligible_events=int(n), events_changing_correctness=int(flips),
                             random_accuracy=chance/den, previous_accuracy=prev/den, current_accuracy=cur/den,
                             paired_accuracy_change=delta, paired_mc_error=err))
    csv_write(folder / "common_truth_best_pair.csv", rankings)
    rows = []
    for model in ("previous", "current"):
        for task, kinds in (("ancestry", OBJECT), ("pair", PAIR)):
            totals = sum(residuals[model, task, kind, y].sum(1) for kind in kinds for y in (0, 1))
            for kind in kinds:
                for y in (0, 1):
                    for g, name in enumerate(NAMES):
                        mass, err, bce = residuals[model, task, kind, y][:, g]
                        rows.append(dict(model=model, task=task, objects=kind, truth=y, process=name,
                                         coefficient_share=mass/totals[0], absolute_logit_gradient_share=err/totals[1],
                                         bce_share=bce/totals[2], mean_absolute_residual=err/mass if mass else None))
    csv_write(folder / "test_higgs_loss_and_residuals.csv", rows)
    plot_pair_tradeoff(rows, rankings, folder, fold)


def main(fold, only):
    out = RUN / f"output/fold_{fold}/evaluation/ancestry_comparison"
    out.mkdir(parents=True, exist_ok=True)
    lookup = group_map()
    if only in ("training", "both"):
        training_mix(fold, out, lookup)
    if only in ("test", "both"):
        head_comparison(fold, out, lookup)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", choices=(0, 1), type=int, required=True)
    p.add_argument("--only", choices=("training", "test", "both"), default="both")
    args = p.parse_args()
    main(args.fold, args.only)
