"""Compare the Z pair selected by each checkpoint for WZ events in the HH signal tail."""
import argparse
import csv
import json
from contextlib import ExitStack

import numpy as np

from compare_event_scores import MODELS, changes, common_fields, open_root, score_path
from paths import ANCESTRY_RUN, RUN
from process_catalog import process_codes


def lepton_pairs(trees, indices):
    """Read the three lepton probabilities and their parent IDs for the selected rows."""
    first, second = np.triu_indices(12, k=1)
    columns = np.flatnonzero((first < 3) & (second < 3))
    truth_ids = []
    probabilities = [[], []]
    for start in range(0, trees[0].num_entries, 32000):
        stop = min(start + 32000, trees[0].num_entries)
        low, high = np.searchsorted(indices, [start, stop])
        if low == high:
            continue
        rows = indices[low:high] - start
        options = dict(entry_start=start, entry_stop=stop, library="np")
        for k, tree in enumerate(trees):
            branch = tree["p_pair_z"].array(**options)
            scores = branch[rows][:, columns]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite Z-pair prediction")
            probabilities[k].append(scores)
        labels = [t["relationZ"].array(**options)[rows, :3] for t in trees]
        if not np.array_equal(*labels):
            raise ValueError("The two score files differ on lepton Z parent IDs")
        flags = trees[1].arrays(["isLepton", "valid"], **options)
        if not (flags["valid"][rows, :3] & flags["isLepton"][rows, :3]).all():
            raise ValueError("Unexpected reconstructed lepton slot order")
        truth_ids.append(labels[0])
    return np.concatenate(truth_ids), [np.concatenate(p) for p in probabilities]


def compare_split(fold, split, metrics, reference):
    cuts = [metrics[split]["split"][f"{m}_score_{m}_veto"]["cuts"][0] for m in MODELS]
    common = ["dsid", "eventNumber", "n_jet", "weight", "in_extconvcr"]
    with ExitStack() as stack:
        trees = [stack.enter_context(open_root(score_path(run, fold, split)))["tree"]
                 for run in (ANCESTRY_RUN, RUN)]
        scores = [t.arrays([*common, "p_signal", "p_extconv_bkg"], library="np") for t in trees]
        common_fields(*scores, common)
        shared = scores[0]
        passing = [(s["n_jet"] >= 1) & (s["p_signal"] >= cut) & (s["p_extconv_bkg"] < .8)
                   for s, cut in zip(scores, cuts)]
        wz = process_codes(shared["dsid"], RUN / "dataset/summary.json") == 0
        idx = np.flatnonzero(wz & (passing[0] | passing[1]))
        ids, pair_scores = lepton_pairs(trees, idx)
    a, b = [mask[idx] for mask in passing]
    weight = shared["weight"][idx].astype(np.float64)
    first, second = np.triu_indices(3, k=1)
    same = (ids[:, first] > 0) & (ids[:, first] == ids[:, second])
    unique = same.sum(axis=1) == 1
    all_known = (ids >= 0).all(axis=1)

    def summarise(status, decision, mask):
        return dict(split=split, truth=status, selection=decision, passing_either_rows=int(mask.sum()),
                    passing_extconvcr_rows=int(shared["in_extconvcr"][idx][mask].sum()),
                    previous_rows=int((a & mask).sum()), current_rows=int((b & mask).sum()),
                    **changes(weight, mask, a, b))

    correct = [np.where(same, p, -np.inf).max(axis=1) > np.where(same, -np.inf, p).max(axis=1)
               for p in pair_scores]
    p, q = correct
    picks = [("both pick the tagged pair", p & q), ("only previous picks it", p & ~q),
             ("only current picks it", ~p & q), ("neither picks it uniquely", ~p & ~q)]
    rows = [summarise("no unique tagged Z pair", "unresolved", ~unique)]
    for label, status in [("all three IDs known", unique & all_known),
                          ("third ID unknown", unique & ~all_known)]:
        rows.extend(summarise(label, decision, status & mask) for decision, mask in picks)
    total = summarise("all WZ", "all", np.ones(len(idx), dtype=bool))
    for key in ("passing_either_rows", "previous_yield", "current_yield", "delta", "entered_rows", "left_rows"):
        np.testing.assert_allclose(sum(r[key] for r in rows), total[key], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm([r["paired_mc_error"] for r in rows]), total["paired_mc_error"])
    for key in ("previous_yield", "current_yield", "delta", "paired_mc_error"):
        np.testing.assert_allclose(total[key], float(reference[key]), rtol=1e-11, atol=1e-12)
    return [total, *rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    fold = parser.parse_args().fold
    folder = RUN / f"output/fold_{fold}/evaluation/ancestry_comparison"
    metrics = json.loads((folder / "event_comparison.json").read_text())
    with (folder / "process_migrations.csv").open() as stream:
        reference = {r["split"]: r for r in csv.DictReader(stream) if r["cuts"] == "split"
                     and r["band_type"] == "cumulative" and r["wp"] == "20" and r["process"] == "WZ"}
    rows = []
    for split in ("test", "validation"):
        rows.extend(compare_split(fold, split, metrics, reference[split]))
    with (folder / "wz_z_pair_at_20.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
