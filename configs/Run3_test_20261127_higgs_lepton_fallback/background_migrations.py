"""Break down WZ, top and single-Higgs migrations at the matched 20% HH working point."""
import argparse
import csv
import json

import numpy as np

from paths import ANCESTRY_RUN, PLAN, RUN
from compare_event_scores import changes, common_fields, open_root, score_path
from process_catalog import process_codes


def production_names():
    samples = json.loads(PLAN.read_bytes())["samples"]
    labels = {}
    for sample in samples.values():
        if sample["process"] not in ("singleH", "VV__WZ", "top"):
            continue
        dsid = int(sample["dsid"])
        parts = sample["caf_path"].split("/")
        category = sample["description"] if sample["process"] == "VV__WZ" else parts[1]
        if category == "ttbar":
            category += "/" + parts[2]
        if category in ("WmH", "WpH"):
            category = "WH"
        if dsid in labels and labels[dsid] != category:
            raise ValueError(f"Different descriptions for DSID {dsid}")
        labels[dsid] = category
    return labels


def details(fold, split, cuts, labels):
    remainder = fold if split == "test" else (fold + 1) % 3
    scores = []
    for run in (ANCESTRY_RUN, RUN):
        with open_root(score_path(run, fold, split)) as f:
            scores.append(f["tree"].arrays(["weight", "eventNumber", "dsid", "n_jet", "p_signal", "p_extconv_bkg"], library="np"))
    common_fields(*scores, ("dsid", "eventNumber", "n_jet", "weight"))
    a, b = scores
    rank = [d["p_signal"] >= cuts[k] for d, k in zip(scores, ("previous", "current"))]
    old, new = [(d["n_jet"] >= 1) & (d["p_extconv_bkg"] < .8) & pass_cut for d, pass_cut in zip(scores, rank)]
    # All events contributing to either yield are retained.
    codes = process_codes(a["dsid"], RUN / "dataset/summary.json")
    take = (old | new) & np.isin(codes, (0, 5, 6))
    selected = np.flatnonzero(take)
    with open_root(RUN / f"dataset/3fold_{remainder}.root") as f:
        inputs = {key: [] for key in ("pdg", "charge", "m_l1l2", "m_l1l3", "m_l2l3", "relationZ", "in_extconvcr")}
        t = f["tree"]
        for start in range(0, len(take), 32000):
            stop = min(start + 32000, len(take))
            rows = selected[(selected>=start) & (selected<stop)]
            if not len(rows):
                continue
            x = t.arrays([*inputs, "eventNumber", "dsid"], entry_start=start, entry_stop=stop, library="np")
            for k in ("eventNumber", "dsid"):
                if not np.array_equal(x[k][rows-start], a[k][rows]):
                    raise ValueError("Raw and scored row identities disagree")
            for k in inputs:
                inputs[k].append(x[k][rows-start])
        inputs = {k: np.concatenate(v) for k, v in inputs.items()}
    w, codes, old, new, ids, jets = [x[take] for x in (a["weight"], codes, old, new, a["dsid"], a["n_jet"])]
    w = w.astype("f8")
    pdg, charge = abs(inputs["pdg"][:, :3]), inputs["charge"][:, :3]
    i, j = np.triu_indices(3, 1)
    sfos = (pdg[:, i] == pdg[:, j]) & (charge[:, i] * charge[:, j] < 0)
    masses = np.stack([inputs[k] for k in ("m_l1l2", "m_l1l3", "m_l2l3")], axis=1) * .001
    dz = np.where(sfos, abs(masses-91.1876), np.inf).min(1)
    rid = inputs["relationZ"][:, :3]
    verified = ((rid[:, i] > 0) & (rid[:, i] == rid[:, j])).any(1)
    rows = []
    for code, process in ((0, "WZ"), (5, "top"), (6, "singleH")):
        process_mask = codes == code
        prod = np.array([labels[k] for k in ids[process_mask]])
        factors = {
            "jet multiplicity": [("1 jet", jets == 1), ("2 jets", jets == 2), ("3+ jets", jets >= 3)],
            "lepton flavour": [(name, (pdg == 11).sum(1)==n) for n, name in enumerate(("3mu", "1e2mu", "2e1mu", "3e"))],
            "closest SFOS mass": [("no SFOS", ~np.isfinite(dz)), ("within 10 GeV of Z", dz < 10),
                                  ("10-25 GeV from Z", (dz >= 10) & (dz < 25)),
                                  ("25+ GeV from Z", (dz >= 25) & np.isfinite(dz))],
            "selected ntuple region": [("SR", ~inputs["in_extconvcr"].astype(bool)), ("ExtConvCR", inputs["in_extconvcr"].astype(bool))],
            "lepton pair with a known common Z": [("yes", verified), ("no", ~verified)],
            "production": [(label, np.isin(ids, ids[process_mask][prod == label])) for label in sorted(set(prod))]}
        for feature, items in factors.items():
            if np.any(sum(mask.astype(int) for _, mask in items)[process_mask] != 1):
                raise ValueError(f"{feature} categories overlap or miss a selected event")
            for category, mask in items:
                rows.append(dict(split=split, process=process, feature=feature, category=category,
                                 **changes(w, mask & process_mask, old, new)))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    fold = parser.parse_args().fold
    folder = RUN / f"output/fold_{fold}/evaluation/ancestry_comparison"
    metrics = json.loads((folder/"event_comparison.json").read_text())
    labels = production_names()
    entries = []
    for split in ("validation", "test"):
        cuts = {k: metrics[split]["split"][k+"_score_"+k+"_veto"]["cuts"][0] for k in ("previous", "current")}
        entries.extend(details(fold, split, cuts, labels))
    with (folder/"background_subgroups_at_20.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=entries[0])
        w.writeheader()
        w.writerows(entries)
