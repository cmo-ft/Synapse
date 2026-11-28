"""Signed background composition at inclusive HH-efficiency cuts on an applied Synapse fold."""
import argparse
import csv
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import uproot


SPLITS = ("validation", "test")
EFFICIENCIES = (0.20, 0.30, 0.50)
METHODS = ("Synapse", "EveNet", "XGBoost")
XGB_SYN = "XGBoost in Synapse SR"
PANELS = ("Synapse", "EveNet", XGB_SYN, "XGBoost")
PANEL_LABELS = ("Synapse", "EveNet", "XGBoost\nSynapse SR", "XGBoost\nEveNet SR")
COLORS = ("#4477AA", "#74ADD1", "#228833", "#CCBB44", "#EE7755", "#AA3377", "#857AAF")
DISPLAY = {"singleH": "Single H", "Vjets": "V + jets", "Vy": "Vγ", "top": "Top"}


def read_split(run, fold, split):
    """Load the scored MC file and the same DSID map and selections as the published ROC."""
    sys.path.insert(0, str(run.parents[1] / "hhml_eval"))
    sys.path.insert(0, str(run / "frozen/plots"))
    import plot_background_rejection as benchmark

    folder = run / "output" / f"fold_{fold}"
    uproot.open.defaults["handler"] = uproot.source.file.MemmapSource
    data, groups = benchmark.load_scores(folder / split / (split + "_scores.root"), run / "dataset/summary.json")
    rem = (fold + 1) % 3 if split == "validation" else fold
    if not np.all(data["eventNumber"] % 3 == rem):
        raise ValueError(f"Incorrect event numbers in {split}")
    keep = data["n_jet"] >= 1
    data = {key: values[keep] for key, values in data.items()}
    groups = groups[keep]
    eve_sr = data["evenet_valid"].astype(bool) & (data["evenet_pConv"] < .8)
    syn_sr = data["p_extconv_bkg"] < .8
    regions = {"Synapse": syn_sr, "EveNet": eve_sr, "XGBoost": eve_sr, XGB_SYN: syn_sr}
    scores = {key: data[field] for key, field in benchmark.METHODS.items()}
    scores[XGB_SYN] = scores["XGBoost"]
    for method, region in regions.items():
        inside = scores[method][region]
        if not ((inside >= 0) & (inside <= 1)).all():
            raise ValueError(f"Invalid {method} scores in chosen signal region")
    return data, groups, list(benchmark.GROUPS.values()), regions, scores


def hh_cuts(weight, signal, regions, scores, efficiencies):
    """Find inclusive cuts, including ties, relative to all positive-weight, jet-preselected HH."""
    picked = signal & (weight > 0)
    total = weight[picked].sum(dtype=np.float64)
    if total <= 0:
        raise ValueError("No positive HH weight")
    cuts = {}
    for name, score in scores.items():
        ids = np.flatnonzero(picked & regions[name])
        order = ids[np.argsort(-score[ids], kind="stable")]
        cw = weight[order].cumsum(dtype=np.float64)
        index = np.searchsorted(cw, total * np.asarray(efficiencies))
        if np.any(index >= len(order)):
            raise ValueError(f"{name} cannot reach a requested HH efficiency")
        cuts[name] = score[order[index]]
    return cuts


def composition(data, groups, names, regions, scores):
    w = data["weight"].astype("f8")
    hh = groups == -1
    hh_pos = hh & (w > 0)
    cuts = hh_cuts(w, hh, regions, scores, EFFICIENCIES)
    rows = []
    for method in PANELS:
        for efficiency, cut in zip(EFFICIENCIES, cuts[method]):
            accept = regions[method] & (scores[method] >= cut)
            bkg = accept & ~hh
            indices, selected = groups[bkg], w[bkg]
            yields = np.bincount(indices, weights=selected, minlength=len(names))
            variances = np.bincount(indices, weights=selected**2, minlength=len(names))
            counts = np.bincount(indices, minlength=len(names))
            total = yields.sum()
            if total <= 0:
                raise ValueError(f"Nonpositive total background yield for {method}, {efficiency}")
            np.testing.assert_allclose(total, selected.sum(), rtol=1e-9)
            for i, name in enumerate(names):
                rows.append(dict(method=method, target_eff=efficiency, cut=float(cut),
                                 actual_positive_hh_eff=w[accept & hh_pos].sum() / w[hh_pos].sum(),
                                 signed_hh_yield=w[accept & hh].sum(), background=name, entries=counts[i],
                                 signed_yield=yields[i], sumw2=variances[i], fraction=yields[i]/total,
                                 total_signed_bkg=total, total_bkg_mc_stat=np.sqrt(variances.sum())))
    return rows


def draw(rows, names, fold, split, out):
    fig, axes = plt.subplots(2, len(EFFICIENCIES), sharex="col", figsize=(13.6, 7.6),
                             gridspec_kw={"height_ratios": [1.13, 1]})
    pos = np.arange(len(PANELS))
    by_key = {(r["method"], float(r["target_eff"]), r["background"]): r for r in rows}
    for column, eff in enumerate(EFFICIENCIES):
        yax, fax = axes[:, column]
        yax.set_title(f"{eff:.0%} HH efficiency", loc="left", fontweight="bold", fontsize=13, pad=9)
        for ax, field in ((yax, "signed_yield"), (fax, "fraction")):
            up, down = np.zeros(len(pos)), np.zeros(len(pos))
            for i, name in enumerate(names):
                values = np.array([float(by_key[method, eff, name][field]) for method in PANELS])
                ax.bar(pos, values, .73, bottom=np.where(values >= 0, up, down),
                       color=COLORS[i], edgecolor="white", linewidth=.45, label=DISPLAY.get(name, name))
                up += np.maximum(values, 0)
                down += np.minimum(values, 0)
            ax.axhline(0, color=".35", linewidth=.6)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[["right", "top"]].set_visible(False)
        totals = np.array([float(by_key[method, eff, names[0]]["total_signed_bkg"]) for method in PANELS])
        errors = np.array([float(by_key[method, eff, names[0]]["total_bkg_mc_stat"]) for method in PANELS])
        yax.errorbar(pos, totals, yerr=errors, color="#282828", fmt="none", capsize=3, linewidth=1)
        for x, value, error in zip(pos, totals, errors):
            yax.annotate(f"{value:.2f}", (x, value + error), xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=9)
        yax.set_ylim(top=max(yax.get_ylim()[1], max(totals + errors) * 1.2))
        fax.yaxis.set_major_formatter(PercentFormatter(1))
        fax.set_xticks(pos, PANEL_LABELS, fontsize=9)
        fax.tick_params(axis="x", length=0, pad=6)
    axes[0, 0].set_ylabel("Expected background events")
    axes[1, 0].set_ylabel("Background fraction")
    # Use one fraction scale and give the low-yield 20% panel its own event scale.
    limits = [ax.get_ylim() for ax in axes[1]]
    for ax in axes[1]:
        ax.set_ylim(min(lo for lo, _ in limits), max(hi for _, hi in limits))
    fig.suptitle(f"Background composition at inclusive HH working points  ·  Fold {fold} {split}",
                 x=.062, ha="left", fontsize=16)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.055, .938),
               ncol=len(names), frameon=False)
    fig.subplots_adjust(left=.073, right=.99, top=.82, bottom=.14, hspace=.17, wspace=.22)
    fig.text(.073, .025, "HH efficiency uses positive-weight SR + ExtConvCR events with ≥1 jet before the classifier veto. "
             "Background yields use signed weights. Error bars show MC statistical uncertainty.", fontsize=9)
    fig.savefig(out / "composition_at_hh_working_points.pdf")
    plt.close(fig)


def wait_for_product(path, fold_dir):
    while not path.is_file():
        status = fold_dir / "exit_status.txt"
        if status.is_file():
            raise RuntimeError(f"Worker exited {status.read_text().strip()} before publishing {path}")
        time.sleep(45)


def publish(run, fold, split):
    data, groups, names, regions, scores = read_split(run, fold, split)
    rows = composition(data, groups, names, regions, scores)
    folder = run / "output" / f"fold_{fold}/evaluation" / split / "background_rejection"
    folder.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=folder) as tmp:
        stage = Path(tmp)
        table = stage / "composition_at_hh_working_points.csv"
        with table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        draw(rows,names,fold,split,stage)
        for p in stage.iterdir():
            p.replace(folder/p.name)
    print(folder / "composition_at_hh_working_points.pdf",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fold", required=True, choices=(0,1,2), type=int)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--wait-for-benchmark", action="store_true",
                        help="Wait for the native benchmark to finish reading each score tree.")
    args=parser.parse_args()
    run = args.run.resolve()
    folder = run / "output" / f"fold_{args.fold}"
    for split in ((args.split,) if args.split else SPLITS):
        if args.wait_for_benchmark:
            wait_for_product(folder / "evaluation" / split / "background_rejection/working_points.csv", folder)
        publish(run, args.fold, split)
