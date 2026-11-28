"""Select nested MC samples from the frozen fold-1 training files using only metadata."""
import json
from pathlib import Path

import numpy as np
import uproot


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "run_dir/Run3_test_20261127_parent_present"
STUDY = REPO / "run_dir" / Path(__file__).resolve().parent.name
SELECTIONS = STUDY / "selections"
LABELS = ("quarter", "half", "double")
TRIALS = (*LABELS, "double_50epochs")
SALT = np.uint64(0xCF4BD811382DE5A7)
FIELDS = ["dsid", "eventNumber", "class_id", "weight", "in_wzcr"]


def load_selections():
    records = json.loads((SELECTIONS / "manifest.json").read_text())
    double = records["double"]
    records["double_50epochs"] = {**double, "draws_per_interval": double["rows"]}
    return records


def buckets(dsid, event):
    # A distinct salt and the high bits keep this independent of the earlier WZCR prescale.
    z = (event.astype("u8") ^ (dsid.astype("u8") << np.uint64(32)) ^ SALT) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return (z ^ (z >> np.uint64(31))) >> np.uint64(62)


def scan(path, remainder, control):
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
        a = source["tree"].arrays(FIELDS, library="np")
    event, dsid, cls = (a[key] for key in ("eventNumber", "dsid", "class_id"))
    if not (np.all(event % 3 == remainder) and np.all(dsid > 0) and np.all(a["in_wzcr"] == control)
            and np.all(np.isin(cls, (0, 1, 2))) and np.all(np.isfinite(a["weight"]))):
        raise ValueError(f"Invalid fold, region, MC label, or weight in {path}")
    return buckets(dsid, event), cls, a["weight"].astype("f8")


def prepare():
    SELECTIONS.mkdir(parents=True, exist_ok=True)
    dataset = BASE / "dataset"
    meta = json.loads((dataset / "summary.json").read_text())
    manifests = {label: {"files": [], "class_counts": [], "main_sum_abs_weight": np.zeros(3)} for label in (*LABELS, "full")}
    masks = {"quarter": [], "half": []}
    for remainder in (0, 1):
        for prefix in ("3fold", "wzcr"):
            path = dataset / f"{prefix}_{remainder}.root"
            control = prefix == "wzcr"
            bucket, cls, weight = scan(path, remainder, control)
            counts = np.bincount(cls, minlength=3)
            expected = meta["train_only_control_counts" if control else "counts_by_remainder"][remainder]
            np.testing.assert_array_equal(counts, expected)
            if not control:
                np.testing.assert_allclose(np.bincount(cls, weights=abs(weight), minlength=3),
                                           meta["sum_abs_weights_by_remainder"][remainder], rtol=1.e-5)
            all_rows = np.ones(len(cls), dtype=bool)
            selections = {"double": all_rows}
            if remainder == 0:
                selections.update(full=all_rows, quarter=bucket == 0, half=bucket < 2)
                for label in masks:
                    masks[label].append(selections[label])
            for label, keep in selections.items():
                report = manifests[label]
                report["files"].append(str(path))
                report["class_counts"].append(np.bincount(cls[keep], minlength=3).tolist())
                if not control:
                    report["main_sum_abs_weight"] += np.bincount(cls[keep], weights=abs(weight[keep]), minlength=3)
    # Check the actual validation IDs as well as their count. Modulo makes overlap impossible.
    val = dataset / "3fold_2.root"
    with uproot.open(val, handler=uproot.source.file.MemmapSource) as source:
        ids = source["tree"]["eventNumber"].array(library="np")
    if len(ids) != sum(meta["counts_by_remainder"][2]) or not np.all(ids % 3 == 2):
        raise ValueError("Invalid fold-1 validation file")
    keep25, keep50 = (np.concatenate(masks[key]) for key in ("quarter", "half"))
    assert np.all(~keep25 | keep50)
    for label, mask in (("quarter", keep25), ("half", keep50)):
        np.save(SELECTIONS / f"{label}.npy", mask, allow_pickle=False)
    draws = sum(map(sum, manifests["full"]["class_counts"]))
    for label, record in manifests.items():
        record.update(rows=sum(map(sum, record["class_counts"])), draws_per_interval=draws,
                      validation_file=str(val), validation_rows=len(ids))
        weights = record["main_sum_abs_weight"]
        record["class_balance"] = (weights[1] / weights).tolist()
        record["main_sum_abs_weight"] = weights.tolist()
        print(label, record["rows"], record["class_counts"], flush=True)
    tmp = SELECTIONS / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifests, indent=2) + "\n")
    tmp.replace(SELECTIONS / "manifest.json")


if __name__ == "__main__":
    prepare()
