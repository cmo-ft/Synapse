"""Compare applying the selected fold 1 models, on the same validation and test events."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import uproot


def read_tree(path, fields):
    with uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None) as root:
        return root["tree"].arrays(fields, library="np")


def score_path(run, fold, split):
    return run / "output" / f"fold_{fold}" / split / f"{split}_scores.root"


def benchmark(run, fold, split, method):
    base = run / "output" / f"fold_{fold}" / "evaluation/hhml_eval/summary.json"
    info = json.loads(base.read_text())["folds"]
    key = "fold_" + str(fold if split == "test" else (fold + 1) % 3)
    return info[key]["models"][method]["trafo60_z"]


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, reference, fold):
    sys.path[:0] = [str(run / "frozen/plots"), str(run / "frozen/synapse/core")]
    from process_catalog import process_codes, GROUPS
    from working_point_significance import working_point_significance
    common = ["dsid", "eventNumber", "n_jet", "weight", "class_id"]
    scores = ["p_signal", "p_extconv_bkg"]
    folder = run / "output" / f"fold_{fold}"
    out = folder / "evaluation" / ("comparison_" + reference.name)
    out.mkdir(parents=True, exist_ok=True)
    cuts, results, migrations = {}, [], []
    for split in ("validation", "test"):
        a = read_tree(score_path(run, fold, split), common + scores +
                      ["evenet_pS", "evenet_pConv", "evenet_valid", "xgb_score"])
        b = read_tree(score_path(reference, fold, split), common + scores)
        for k in common:
            np.testing.assert_array_equal(a[k], b[k])
        remainder = (fold + 1) % 3 if split == "validation" else fold
        if not np.all(a["eventNumber"] % 3 == remainder):
            raise ValueError(f"Wrong split for {split}")
        w, hh, pre = a["weight"].astype("f8"), a["class_id"] == 0, a["n_jet"] >= 1
        process = process_codes(a["dsid"], run / "dataset/summary.json")
        np.testing.assert_array_equal(process == -1, hh)
        eve_region = a["evenet_valid"].astype(bool) & (a["evenet_pConv"] < .8)
        methods = {"New Synapse": (a["p_signal"], a["p_extconv_bkg"] < .8, run, "Synapse"),
                   "Previous Synapse": (b["p_signal"], b["p_extconv_bkg"] < .8, reference, "Synapse"),
                   "EveNet": (a["evenet_pS"], eve_region, run, "EveNet"),
                   "XGBoost": (a["xgb_score"], eve_region, run, "XGB_EveNetSR")}
        for mode in ("matched", "validation_cuts"):
            pass20 = {}
            for model, (score, region, bench, bench_model) in methods.items():
                values = working_point_significance(score, hh, w, pre, region,
                            cuts=cuts[model] if mode == "validation_cuts" else None)
                if split == "validation" and mode == "matched":
                    cuts[model] = values["cuts"]
                passed = pre & region & (score >= values["cuts"][0])
                pass20[model] = passed
                row = dict(split=split, cuts=mode, model=model, r20=None, z_20_30_50=values["quadrature"],
                           trafo60=benchmark(bench,fold,split,bench_model), signal_eff=values["cumulative"][0]["positive_hh_efficiency"])
                for code, group in [(-2,"all"),*enumerate(GROUPS.values())]:
                    category = (~hh) if code == -2 else (process == code)
                    positive = category & pre & (w>0)
                    num = w[positive & passed].sum()
                    row["r20" if group=="all" else "r20_"+group] = float(w[positive].sum()/num) if num else float("inf")
                results.append(row)
            old, new = pass20["Previous Synapse"], pass20["New Synapse"]
            for k, group in [(-2, "All backgrounds"), (-1,"HH"), *enumerate(GROUPS.values())]:
                sel = (~hh) if k == -2 else (process == k)
                d = new.astype("i1")-old.astype("i1")
                migrations.append(dict(split=split,cuts=mode,category=group,reference=float(w[sel & old].sum()),
                    candidate=float(w[sel & new].sum()),delta=float((w[sel]*d[sel]).sum()),
                    paired_mc_error=float(np.sqrt(np.square(w[sel]*d[sel]).sum())),
                    entering=int((sel & new & ~old).sum()),leaving=int((sel & old & ~new).sum())))
    save_csv(out / "results.csv", results)
    save_csv(out / "signed_yield_at_20.csv", migrations)
    records = [json.loads(p.read_text()) for p in sorted((folder/"fit/epochs").glob("epoch_*.json"))]
    best=max(records,key=lambda d:d["val_Significance20_30_50_epoch"])
    summary=[f"# Fold {fold} evaluation", "", f"Candidate {run.name}. Reference {reference.name}.",
             f"The maximum validation three-bin significance selects epoch {best['epoch']} of {len(records)}.",
             "", "The table uses score thresholds fitted separately on each split at 20% HH efficiency.",
             "", "| Split | Model | Inclusive R20 | WZ R20 | VVV R20 | Three-bin Z | Trafo60 Z |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for d in results:
        if d["cuts"] == "matched":
            summary.append(f"| {d['split']} | {d['model']} | {d['r20']:.1f} | {d['r20_WZ']:.1f} | {d['r20_VVV']:.1f} | {d['z_20_30_50']:.5f} | {d['trafo60']:.5f} |")
    summary.extend(["","Signed changes in expected background at the matched 20% working point use the same MC events in both score selections.",""])
    for split in ("validation","test"):
        for cat in ("All backgrounds","WZ","VVV","singleH","top"):
            x=next(z for z in migrations if z["split"]==split and z["cuts"]=="matched" and z["category"]==cat)
            summary.append(f"- {split.capitalize()} {cat}, {x['delta']:+.4f} ± {x['paired_mc_error']:.4f} expected events.")
    summary.extend(["", "The CSV also gives test results using thresholds fitted on validation.",
                    "Paired uncertainties hold the fitted working-point thresholds fixed.", ""])
    (out / "comparison.md").write_text("\n".join(summary))
    print("\n".join(summary),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--fold",type=int,default=1)
    a=p.parse_args()
    analyze(a.run.resolve(),a.reference.resolve(),a.fold)
