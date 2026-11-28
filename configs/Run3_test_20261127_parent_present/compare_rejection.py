"""Overlay a fold's new and preceding NN, EveNet and XGBoost on the common rejection benchmark."""
import argparse
import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import uproot


def triboson_yields(data, old, groups, group_index, points, path):
    weight = data["weight"].astype("f8")
    pre = data["n_jet"] >= 1
    eve_region = data["evenet_valid"].astype(bool) & (data["evenet_pConv"] < .8)
    scores = {"Synapse": (data["p_signal"], data["p_extconv_bkg"] < .8),
              "Previous": (old["p_signal"], old["p_extconv_bkg"] < .8),
              "EveNet": (data["evenet_pS"], eve_region)}
    cuts = {(p["method"], p["target_signal_eff"]): p["score_min"] for p in points
            if p["normalization"] == "from_preselection" and p["category"] == "All backgrounds"}
    rows = []
    triboson = groups == group_index
    samples = {"all VVV": triboson}
    samples.update({str(dsid): triboson & (data["dsid"] == dsid) for dsid in np.unique(data["dsid"][triboson])})
    for efficiency in (.20, .30, .50):
        passes = {m: pre & region & (s >= cuts[m, efficiency]) for m, (s, region) in scores.items()}
        for name, sample in samples.items():
            for baseline in ("Previous","EveNet"):
                new, before = passes["Synapse"][sample], passes[baseline][sample]
                w = weight[sample]
                change = w * (new.astype("i1") - before.astype("i1"))
                rows.append(dict(dsid=name, target_HH_eff=efficiency, reference=baseline,
                                 reference_yield=w[before].sum(), candidate_yield=w[new].sum(),
                                 delta=change.sum(), paired_mc_error=np.linalg.norm(change),
                                 entering=int((new & ~before).sum()), leaving=int((before & ~new).sum())))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_split(run, reference, fold, split, labels, out, benchmark):
    name = f"output/fold_{fold}/{split}/{split}_scores.root"
    a, groups = benchmark.load_scores(run / name, run / "dataset/summary.json")
    identity = ("eventNumber", "dsid", "class_id", "weight", "n_jet", "evenet_valid", "evenet_pS", "evenet_pConv", "xgb_score")
    with uproot.open(reference / name) as source:
        b = source["tree"].arrays([*identity, "p_signal", "p_extconv_bkg"], library="np")
    for key in identity:
        np.testing.assert_array_equal(a[key], b[key])
    remainder = (fold + 1) % 3 if split == "validation" else fold
    if not np.all(a["eventNumber"] % 3 == remainder):
        raise ValueError(f"Unexpected remainder in {split}")
    group, weights, regions, scores, _ = benchmark.roc_inputs(a, groups)
    use = (a["n_jet"] >= 1) & (a["weight"] > 0)
    regions["Previous"] = b["p_extconv_bkg"][use] < .8
    scores["Previous"] = b["p_signal"][use]
    cats, curves, points = benchmark.measure(group, weights, regions, scores, reference="Synapse")
    # Check each published working point, including the two original signal-region conventions.
    mapping = {(x["normalization"], x["category"], x["method"], x["target_signal_eff"]): x for x in points}
    for path, method in ((run,"Synapse"), (reference,"Previous")):
        before = path / f"output/fold_{fold}/evaluation/{split}/background_rejection/working_points.csv"
        with before.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["method"] != "Synapse":
                    continue
                current = mapping[row["normalization"], row["category"], method, float(row["target_signal_eff"])]
                for key in ("signal_eff","rejection","rejection_mc_stat","score_min"):
                    np.testing.assert_allclose(current[key], float(row[key]), rtol=1e-10, atol=1e-12, equal_nan=True)
    styles = {k: benchmark.STYLES[k] for k in ("Synapse","EveNet","XGBoost")}
    styles["Previous"] = ("#843b92", ":")
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        triboson_yields(a, b, groups, list(benchmark.GROUPS.values()).index("VVV"), points,
                       stage / "vvv_working_points.csv")
        benchmark.plot(cats, curves, points, f"Training fold {fold} · {split}", stage, styles=styles,
                       labels={"Synapse":labels[0],"Previous":labels[1]}, reference="Synapse")
        with (stage / "working_points.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=points[0])
            writer.writeheader()
            writer.writerows(points)
        for file in stage.iterdir():
            file.replace(out / file.name)
    print(split, "verified both NN curves at published working points", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--labels", nargs=2, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    run, ref = a.run.resolve(), a.reference.resolve()
    sys.path[:0] = [str(run.parent.parent/"hhml_eval"), str(run/"frozen/plots")]
    import plot_background_rejection as benchmark
    uproot.open.defaults["handler"] = uproot.source.file.MemmapSource
    dest = a.out or run / f"output/fold_{a.fold}/evaluation" / ("comparison_" + ref.name)
    for split in ("validation", "test"):
        plot_split(run, ref, a.fold, split, a.labels, dest / split, benchmark)


if __name__ == "__main__":
    main()
