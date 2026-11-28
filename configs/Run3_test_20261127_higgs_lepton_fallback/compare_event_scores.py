"""Compare two selected Synapse checkpoints on identical validation and test events."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from paths import ANCESTRY_RUN, RUN, TREE_RUN

sys.path.insert(0, str(RUN / "frozen/plots"))
sys.path.insert(0, str(RUN / "frozen/synapse/core"))
from process_catalog import GROUPS, process_codes
from working_point_significance import asimov_significance, working_point_significance


MODELS = ("previous", "current")
PAIRS = (("previous", "previous"), ("current", "previous"),
         ("previous", "current"), ("current", "current"))
FIELDS = ("eventNumber", "dsid", "weight", "n_jet", "class_id", "in_extconvcr",
          "wAssignment", "evenet_pS", "evenet_pB", "evenet_pConv", "evenet_valid", "xgb_score")
SCORES = ("p_signal", "p_extconv_bkg")
GROUP_NAMES = tuple(GROUPS.values())
NBOOT = 250


def open_root(path):
    return uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None)


def score_path(run, fold, split):
    return run / "output" / f"fold_{fold}" / split / f"{split}_scores.root"


def common_fields(old, new, fields):
    if any(not np.array_equal(old[name], new[name], equal_nan=True) for name in fields):
        bad = [name for name in fields if not np.array_equal(old[name], new[name], equal_nan=True)]
        raise ValueError(f"Compared files disagree on {bad}")


def load_split(fold, split):
    remainder = (fold + 1) % 3 if split == "validation" else fold
    with open_root(score_path(ANCESTRY_RUN, fold, split)) as prev, open_root(score_path(RUN, fold, split)) as cur:
        a, b = [f["tree"].arrays([*FIELDS, *SCORES], library="np") for f in (prev, cur)]
    common_fields(a, b, FIELDS)
    if not np.all(a["eventNumber"] % 3 == remainder):
        raise ValueError("Unexpected event remainder")
    verify = {"scored_rows": len(a["weight"]), "signed_weight": float(a["weight"].sum(dtype="f8")),
              "jet_preselected_rows": int((a["n_jet"] >= 1).sum())}
    keep = a["n_jet"] >= 1
    common = {k: a[k][keep] for k in ("dsid", "weight", "wAssignment", "class_id", "n_jet")}
    common["hh"] = common["class_id"] == 0
    common["weight"] = common["weight"].astype("f8")
    common["process"] = process_codes(common["dsid"], RUN / "dataset/summary.json")
    if not np.array_equal(common["process"] == -1, common["hh"]):
        raise ValueError("HH truth and DSID disagree")
    common["score"] = {name: x["p_signal"][keep] for name, x in zip(MODELS, (a, b))}
    common["veto_pass"] = {name: x["p_extconv_bkg"][keep] < .8 for name, x in zip(MODELS, (a, b))}
    # Use the published fold to verify tree-only and lepton-fallback labels on the same HH events.
    fallback = np.zeros(len(keep), dtype=bool)
    with open_root(TREE_RUN / f"dataset/3fold_{remainder}.root") as tree, open_root(score_path(RUN, fold, split)) as current:
        old, new = tree["tree"], current["tree"]
        if old.num_entries != new.num_entries or new.num_entries != len(keep):
            raise ValueError("Scored rows differ from the labelled dataset")
        fields = ("eventNumber", "dsid", "ancestryHiggs", "higgsPairKnown")
        for start in range(0, len(keep), 64000):
            stop = min(start + 64000, len(keep))
            x, y = [t.arrays(fields, entry_start=start, entry_stop=stop, library="np") for t in (old, new)]
            common_fields(x, y, ("eventNumber", "dsid"))
            fallback[start:stop] = (np.any((x["ancestryHiggs"] < 0) & (y["ancestryHiggs"] >= 0), axis=1)
                                    | np.any(y["higgsPairKnown"] != x["higgsPairKnown"], axis=1))
    common["fallback"] = fallback[keep]
    return common, verify


def bin_indices(score, veto, cuts):
    bins = np.full(len(score), 3, dtype="i1")
    bins[veto] = np.searchsorted(-np.asarray(cuts, dtype="f8"), -score[veto], side="left")
    return bins


def metric(data, score_name, veto_name, cuts=None):
    w, hh = data["weight"], data["hh"]
    score, veto = data["score"][score_name], data["veto_pass"][veto_name]
    result = working_point_significance(score, hh, w, np.ones_like(hh), veto, cuts)
    bins = bin_indices(score, veto, result["cuts"])
    positive_b = ~hh & (w > 0)
    positive_total = w[positive_b].sum()
    result["positive_background_rejection"] = [float(positive_total / w[positive_b & (bins <= k)].sum())
                                               for k in range(3)]
    return result, bins


def paired_bootstrap(w, hh, old, new, seed):
    # Resample a paired event once and reuse its Poisson count in both score selections.
    # Background variance in each replica sums the replicated original squared weights.
    use = (old < 3) | (new < 3)
    w, hh, old, new = w[use], hh[use], old[use], new[use]
    joint = (4 * old + new).astype(np.int32)
    h, b = joint[hh], joint[~hh]
    wh, wb = w[hh], w[~hh]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(NBOOT):
        sw = rng.poisson(1., len(h)) * wh
        bw = rng.poisson(1., len(b)) * wb
        mats = [np.bincount(index, weights=weights, minlength=16).reshape(4, 4)
                for index, weights in ((h, sw), (b, bw), (b, bw * wb))]
        z = []
        for axis in (1, 0):
            yields = [m.sum(axis=axis)[:3] for m in mats]
            z.append(np.linalg.norm([asimov_significance(*values) for values in zip(*yields)]))
        differences.append(z[1] - z[0])
    arr = np.asarray(differences)
    return {"replicas": NBOOT, "seed": int(seed), "sd": float(arr.std(ddof=1)),
            "percentiles_2p5_16_50_84_97p5": np.percentile(arr, [2.5, 16, 50, 84, 97.5]).tolist()}


def changes(w, mask, previous, current):
    enter, leave = mask & current & ~previous, mask & previous & ~current
    bp, bc = float(w[mask & previous].sum()), float(w[mask & current].sum())
    return dict(previous_yield=bp, current_yield=bc, delta=bc - bp,
                paired_mc_error=float(np.sqrt(np.square(w[enter | leave]).sum())),
                entered_rows=int(enter.sum()), left_rows=int(leave.sum()),
                entered_yield=float(w[enter].sum()), left_yield=float(w[leave].sum()))


def split_tables(data, split, mode, old, new, proc, topologies):
    w, hh = data["weight"], data["hh"]
    types = (("fully matched HH → 4W", data["wAssignment"].astype(bool)),
             ("other HH with a fallback", ~data["wAssignment"].astype(bool) & data["fallback"]),
             ("other HH without a fallback", ~data["wAssignment"].astype(bool) & ~data["fallback"]))
    for index, wp in enumerate((20, 30, 50)):
        for name, a, b in (("cumulative", old <= index, new <= index), ("exclusive", old == index, new == index)):
            for code, group in [(-2, "all backgrounds"), (-1, "HH"), *enumerate(GROUP_NAMES)]:
                mask = ~hh if code == -2 else data["process"] == code
                proc.append(dict(split=split, cuts=mode, band_type=name, wp=wp, process=group,
                                 **changes(w, mask, a, b)))
        for group, sel in types:
            mask = hh & sel & (w > 0)
            den = w[mask].sum()
            diff = changes(w, mask, old <= index, new <= index)
            topologies.append(dict(split=split, cuts=mode, wp=wp, group=group, preselected_rows=int(mask.sum()),
                                   positive_weight=float(den), previous_eff=diff["previous_yield"] / den,
                                   current_eff=diff["current_yield"] / den,
                                   paired_eff_error=diff["paired_mc_error"] / den, **diff))


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def figure(folder, fold, rows):
    fig, panels = plt.subplots(1, 3, figsize=(12.8, 4.3), sharey=True)
    ys = np.arange(len(GROUP_NAMES))
    for i, (ax, wp) in enumerate(zip(panels, (20, 30, 50))):
        for j, (split, cuts) in enumerate((("validation", "split"), ("test", "split"))):
            r = [r for r in rows if r["split"] == split and r["cuts"] == cuts and r["wp"] == wp
                 and r["band_type"] == "cumulative" and r["process"] in GROUP_NAMES]
            ax.errorbar([s["delta"] for s in r], ys+(j-.5)*.25, xerr=[s["paired_mc_error"] for s in r],
                        fmt=("o", "s")[j], ms=4, capsize=2, color=("#4776a4", "#ae5037")[j], label=split.capitalize())
        ax.axvline(0, color=".35", lw=.8)
        ax.set_title(f"{wp}% HH efficiency")
        ax.set_xlabel("Change in expected events")
        ax.grid(axis="x", alpha=.18)
        if i == 0:
            ax.legend(frameon=False)
    panels[0].set_yticks(ys, GROUP_NAMES)
    panels[0].invert_yaxis()
    fig.suptitle(f"Current minus previous background yield  ·  Fold {fold}")
    fig.text(.1, .032, "Each split is compared at matched HH efficiency. Bars show paired MC statistical errors at the observed cuts.", fontsize=9)
    fig.tight_layout(rect=[0, .075, 1, .965])
    fig.savefig(folder / "background_yield_change.pdf")
    plt.close(fig)


def main(fold):
    folder = RUN / "output" / f"fold_{fold}/evaluation/ancestry_comparison"
    folder.mkdir(parents=True, exist_ok=True)
    report, val_cuts, proc, topologies = {}, {}, [], []
    for split in ("validation", "test"):
        data, identity = load_split(fold, split)
        comparison = {"identity": identity}
        report[split] = comparison
        for mode in (("split",) if split == "validation" else ("split", "validation")):
            metrics, bins = {}, {}
            for rank, veto in PAIRS:
                key = rank + "_score_" + veto + "_veto"
                m, b = metric(data, rank, veto, val_cuts[key] if mode == "validation" else None)
                metrics[key] = m
                if split == "validation":
                    val_cuts[key] = m["cuts"]
                if rank == veto:
                    bins[rank] = b
            previous, current = bins["previous"], bins["current"]
            split_tables(data, split, mode, previous, current, proc, topologies)
            comparison[mode] = metrics
            metrics["paired_delta_z_bootstrap"] = paired_bootstrap(data["weight"], data["hh"], previous, current,
                                                                   7331+fold+(100 if split == "validation" else 0))
            print(split, mode, [(m, round(metrics[m+"_score_"+m+"_veto"]["quadrature"], 6)) for m in MODELS], flush=True)
        del data
    (folder / "event_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    csv_file(folder / "process_migrations.csv", proc)
    csv_file(folder / "hh_topologies.csv", topologies)
    figure(folder, fold, proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 1))
    main(parser.parse_args().fold)
