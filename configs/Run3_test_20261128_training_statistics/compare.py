"""Compare best-by-significance validation checkpoints on identical fold-1 events."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import uproot
from sklearn.metrics import roc_auc_score

from selection import BASE, REPO, STUDY, TRIALS, load_selections

sys.path[:0] = [str(BASE / "frozen/plots"), str(REPO / "hhml_eval"), str(BASE / "frozen")]
import plot_background_rejection as benchmark
from synapse.core.working_point_significance import working_point_significance

KEYS = ["eventNumber", "dsid", "class_id", "weight", "n_jet"]
PROBS = ["p_signal", "p_bkg", "p_extconv_bkg"]
PLOT_LABELS = {"quarter": "25% MC", "half": "50% MC", "full": "100% MC",
               "double": "2× MC, 25 epochs", "double_50epochs": "2× MC, 50 epochs"}
PLOT_STYLES = {"quarter": ("#D55E00", "--"), "half": ("#0072B2", "-."),
               "full": ("#383838", "-"), "double": ("#008568", "--"),
               "double_50epochs": ("#843B92", "-")}


def read_scores(path):
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
        return source["tree"].arrays([*KEYS, *PROBS], library="np")


def collect_history(workspace):
    return [json.loads(path.read_text()) for path in sorted((workspace / "fit/epochs").glob("epoch_*.json"))]


def compare(point, reference="full"):
    manifest = load_selections()
    original = BASE / "output/fold_1"
    ref_workspace = original if reference == "full" else STUDY / reference
    trial = original if point == "full" else STUDY / point
    relative = "validation/validation_scores.root"
    ref = read_scores(ref_workspace / relative)
    a = ref if point == reference else read_scores(trial / relative)
    for key in KEYS:
        np.testing.assert_array_equal(ref[key], a[key])
    if len(a["class_id"]) != manifest["full"]["validation_rows"] or not np.all(a["eventNumber"] % 3 == 2):
        raise ValueError("Unexpected validation sample")
    groups = benchmark.process_codes(a["dsid"], BASE / "dataset/summary.json")
    hh = a["class_id"] == 0
    np.testing.assert_array_equal(groups == -1, hh)
    w = a["weight"].astype("f8")
    balanced = abs(w) * np.asarray(manifest["full"]["class_balance"])[a["class_id"]]
    pre = a["n_jet"] >= 1
    pos = pre & (w > 0)
    bgw = w[pos]
    group = groups[pos]
    regions, scores, results = {}, {}, {}
    models = {reference: (ref, ref_workspace)}
    if point != reference:
        models[point] = (a, trial)
    for name, (data, workspace) in models.items():
        probs = np.column_stack([data[key] for key in PROBS])
        matches = probs[np.arange(len(hh)), a["class_id"]].astype("f8")
        if not np.isfinite(probs).all() or (probs < 0).any():
            raise ValueError(f"Invalid predicted probabilities in {name}")
        ce = np.average(-np.log(np.maximum(matches, np.finfo("f4").tiny)), weights=balanced)
        auc = roc_auc_score(a["class_id"], probs, multi_class="ovr", average="macro", sample_weight=balanced)
        sr = data["p_extconv_bkg"] < .8
        z = working_point_significance(data["p_signal"], hh, w, pre, sr)
        regions[name] = sr[pos]
        scores[name] = data["p_signal"][pos]
        history = collect_history(workspace)
        best = max(history, key=lambda row: row["val_Significance20_30_50_epoch"])
        results[name] = dict(rows=manifest[name]["rows"], best_epoch=best["epoch"], final_epoch=history[-1],
                             common_weight_cross_entropy=float(ce), common_weight_macro_auc=float(auc),
                             significance_20_30_50=z, best_epoch_log=best)
    categories, curves, points = benchmark.measure(group, bgw, regions, scores, reference=reference)
    rows = [p for p in points if p["normalization"] == "from_preselection"]
    for row in rows:
        if row["category"] == "All backgrounds" and row["target_signal_eff"] == .2:
            results[row["method"]]["inclusive_r20"] = float(row["rejection"])
    dest = STUDY / point / ("comparison" if reference == "full" else f"comparison_to_{reference}")
    dest.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest) as scratch:
        tmp = Path(scratch)
        benchmark.plot(categories, curves, points, "Fold 1 validation", tmp,
                       styles=PLOT_STYLES, labels=PLOT_LABELS, reference=reference)
        with (tmp / "working_points.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        (tmp / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
        for path in tmp.iterdir():
            path.replace(dest / path.name)
    print(point, {m: (r["best_epoch"], r["inclusive_r20"], r["significance_20_30_50"]["quadrature"])
                  for m, r in results.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point", choices=(*TRIALS, "full"))
    parser.add_argument("--reference", choices=("full", "double"), default="full")
    args = parser.parse_args()
    compare(args.point, args.reference)
