"""Summarize the frozen Higgs and W labels for inclusive WH and matched HH → 4W."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import awkward as ak
import numpy as np
import uproot


WH = (604305, 604306)
FIELDS = ("dsid", "eventNumber", "weight", "n_jet", "valid", "label_signal", "wAssignment",
          "ancestryHiggs", "ancestryW", "relationHiggs", "relationW")
PARENTS = ("Higgs", "W")


def read_sample(run, fold, split, remainder):
    data = {field: [] for field in FIELDS}
    data["pass_r20"] = []
    folder = run / "output" / f"fold_{fold}"
    score_file = None
    if split != "training":
        path = folder / "evaluation" / split / "background_rejection/working_points.csv"
        with path.open() as stream:
            cuts = {float(row["score_min"]) for row in csv.DictReader(stream)
                    if row["method"] == "Synapse" and row["normalization"] == "from_preselection"
                    and float(row["target_signal_eff"]) == .2}
        cut, = cuts
        score_file = uproot.open(folder / split / f"{split}_scores.root")
    try:
        with uproot.open(run / "dataset" / f"3fold_{remainder}.root") as root:
            tree = root["tree"]
            index = tree.arrays(["dsid", "wAssignment"], library="np")
            selected = np.isin(index["dsid"], WH) | index["wAssignment"].astype(bool)
            for start in range(0, len(selected), 24000):
                stop = min(start + 24000, len(selected))
                use = selected[start:stop].copy()
                if not use.any():
                    continue
                a = tree.arrays(FIELDS, entry_start=start, entry_stop=stop, library="np")
                use &= a["n_jet"] >= 1
                veto_pass = np.zeros(len(use), dtype=bool)
                if score_file is not None:
                    s = score_file["tree"].arrays(["dsid", "eventNumber", "p_signal", "p_extconv_bkg"],
                                                 entry_start=start, entry_stop=stop, library="np")
                    for key in ("dsid", "eventNumber"):
                        np.testing.assert_array_equal(a[key], s[key])
                    veto_pass = (s["p_signal"] >= cut) & (s["p_extconv_bkg"] < .8) & (a["weight"] > 0)
                data["pass_r20"].append(veto_pass[use])
                for field in FIELDS:
                    data[field].append(a[field][use])
    finally:
        if score_file is not None:
            score_file.close()
    return {key: np.concatenate(chunks) for key, chunks in data.items()}


def counts(values):
    return {name: int(np.count_nonzero(values == number))
            for number, name in ((-1, "unknown"), (0, "negative"), (1, "positive"))}


def pair_counts(ids):
    first, second = ids.T
    known = (first >= 0) & (second >= 0)
    same = (first > 0) & (first == second)
    return counts(np.where(known, same.astype("i1"), -1))


def summarize(data, keep, matched=False):
    valid = data["valid"][keep, 3:9].astype(bool)
    jets = (data["relationW"][keep, 3:9] > 0) if matched else valid[:, :2]
    two = jets.sum(1) == 2
    if matched and not two.all():
        raise ValueError("Expected exactly two matched W jets for each resolved HH event")
    summary = dict(events=int(keep.sum()), pairs=int(two.sum()), label_counts={})
    for parent in PARENTS:
        a = data["ancestry" + parent][keep]
        r = data["relation" + parent][keep, 3:9] if matched else data["relation" + parent][keep, 3:5]
        pair = r[two][jets[two]].reshape(-1, 2)
        summary["label_counts"][parent] = dict(leptons=counts(a[:, :3]), l3=counts(a[:, 2]),
             all_valid_central_jets=counts(a[:, 3:9][valid]),
             two_jet_ancestry=counts((a[:, 3:9] if matched else a[:, 3:5])[two][jets[two]]),
             two_jet_same_parent=pair_counts(pair))
    return summary


def add_example_truth(results, plan_path):
    examples = {}
    for split in ("validation", "test"):
        for dsid in WH:
            for event in results[split][str(dsid)]["at_R20"]["examples"]:
                examples[dsid, event["event"]] = event
                event["raw_higgs_decays"] = []
    ids = {dsid: [number for sample, number in examples if sample == dsid] for dsid in WH}
    fields = [f"truth_{level}_fromH{h}_pdgId" for h in (1, 2) for level in ("children", "grandchildren")]
    for job in json.loads(plan_path.read_bytes())["jobs"]:
        if job["kind"] != "weights":
            continue
        sample = job["sample"]
        prefix = sample.split("_")[0]
        if not prefix.isnumeric() or not ids.get(int(prefix)):
            continue
        dsid = int(prefix)
        for path in job["files"]:
            with uproot.open(path) as root:
                tree = root["AnalysisMiniTree"]
                numbers = tree["eventNumber"].array(library="np")
                for i in np.flatnonzero(np.isin(numbers, ids[dsid])):
                    row = tree.arrays(fields, entry_start=int(i), entry_stop=int(i+1))
                    examples[dsid, int(numbers[i])]["raw_higgs_decays"].append(dict(sample=sample, **ak.to_list(row)[0]))


def main(run, fold, caf_plan):
    uproot.open.defaults["handler"] = uproot.source.file.MemmapSource
    results = {}
    for split, r in (("training", (fold+2)%3), ("validation", (fold+1)%3), ("test", fold)):
        data = read_sample(run, fold, split, r)
        full = data["wAssignment"].astype(bool)
        if not data["label_signal"][full].all():
            raise ValueError("W assignment outside HH")
        groups = {"HH_fully_matched": summarize(data, full, matched=True)}
        for dsid in WH:
            wh = data["dsid"] == dsid
            groups[str(dsid)] = summarize(data, wh)
            if split == "training":
                continue
            passed = np.flatnonzero(wh & data["pass_r20"])
            top = passed[np.argsort(-data["weight"][passed])[:3]]
            ncentral = data["valid"][passed, 3:9].sum(1)
            w = data["weight"][passed].astype("f8")
            groups[str(dsid)]["at_R20"] = dict(events=len(passed), positive_weight=float(w.sum()),
                one_central_jet_events=int((ncentral == 1).sum()),
                one_central_jet_positive_weight=float(w[ncentral == 1].sum()), examples=[
                dict(event=int(data["eventNumber"][i]), positive_weight=float(data["weight"][i]),
                     central_jets=int(data["valid"][i, 3:9].sum()),
                     **{key: data[key][i, :5].tolist() for key in FIELDS if key.startswith(("ancestry","relation"))})
                for i in top])
        results[split] = groups
    if caf_plan is not None:
        add_example_truth(results, caf_plan)
    out = run / "output" / f"fold_{fold}/evaluation/validation_test"
    with TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / "wh_hh_truth_labels.json"
        path.write_text(json.dumps(dict(fold=fold, selection="Base SR + ExtConvCR MC with ≥1 jet. Counts are unweighted. "
            "HH requires the stored full four-W match and uses the actual W jet pair among the six central slots. "
            "WH uses only inclusive DSIDs 604305/604306 and pairs the two leading valid central jets. "
            "R20 examples use positive nominal weights, ordered by weight. The five example slots are l1,l2,l3,j1,j2.",
            splits=results), indent=2) + "\n")
        path.replace(out / path.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--caf-plan", type=Path, help="Read the Higgs daughter PDG IDs for the passing WH examples.")
    args = p.parse_args()
    main(args.run.resolve(), args.fold, args.caf_plan)
