"""Maintain the two-fold epoch comparison and apply validation thresholds to each held-out sample."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time


METRICS = {
    "loss": ("val_loss_epoch", "validation_loss"),
    "auc": ("val_AUC_epoch", "validation_auc"),
    "r20": ("val_Rejection20_epoch", "validation_rejection20"),
    "z": ("val_Significance20_30_50_epoch", "validation_significance"),
}


def write_table(run, epochs, previous):
    columns = ["epoch"] + [f"fold_{fold}_{kind}{name}" for name in METRICS
                           for fold in range(2) for kind in ("", "previous_")]
    path = run / "epochs.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=columns)
        table.writeheader()
        for epoch in range(1, max(map(len, epochs)) + 1):
            row = {"epoch": epoch}
            for fold, records in enumerate(epochs):
                for name, (key, old_key) in METRICS.items():
                    if epoch in records:
                        row[f"fold_{fold}_{name}"] = records[epoch][key]
                    if epoch in previous:
                        row[f"fold_{fold}_previous_{name}"] = previous[epoch].get(f"fold_{fold}_{old_key}")
            table.writerow(row)
    temp.replace(path)


def compare_working_points(fold_dir, fold):
    import numpy as np
    import uproot
    from synapse.core.working_point_significance import working_point_significance

    cuts = {}
    results = {}
    branches = ["eventNumber", "n_jet", "label_signal", "weight", "p_signal", "p_extconv_bkg",
                "evenet_valid", "evenet_pS", "evenet_pConv"]
    for split, remainder in (("validation", (fold + 1) % 3), ("test", fold)):
        with uproot.open(fold_dir / split / (split + "_scores.root"),
                         handler=uproot.source.file.MemmapSource) as root:
            data = root["tree"].arrays(branches, library="np")
        if not np.all(data["eventNumber"] % 3 == remainder):
            raise ValueError(f"Wrong events in {split}")
        pre = data["n_jet"] >= 1
        results[split] = {}
        for name, score, region in (
            ("Synapse", data["p_signal"], data["p_extconv_bkg"] < .8),
            ("EveNet", data["evenet_pS"], data["evenet_valid"].astype(bool) & (data["evenet_pConv"] < .8)),
        ):
            args = (score, data["label_signal"], data["weight"], pre, region)
            same = working_point_significance(*args)
            if split == "validation":
                cuts[name] = same["cuts"] if same is not None else None
            fixed = working_point_significance(*args, cuts=cuts[name]) if cuts[name] is not None else None
            results[split][name] = {"same_split": same, "validation_cuts": fixed}
    path = fold_dir / "evaluation/working_points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def monitor(run, reference, interval):
    sys.path.insert(0, str(run / "frozen"))
    with (reference / "epochs.csv").open() as stream:
        previous = {int(row["epoch"]): row for row in csv.DictReader(stream)}
    epochs = [{}, {}]
    while True:
        changed = False
        done = []
        for fold, records in enumerate(epochs):
            folder = run / "output" / f"fold_{fold}"
            epoch = len(records) + 1
            while (path := folder / "fit/epochs" / f"epoch_{epoch:03d}.json").is_file():
                record = json.loads(path.read_text())
                if record["epoch"] != epoch:
                    raise ValueError(f"Unexpected epoch in {path}")
                print(f"Fold {fold} epoch {epoch}: validation loss {record['val_loss_epoch']:.5f}, "
                      f"AUC {record['val_AUC_epoch']:.5f}, R20 {record['val_Rejection20_epoch']:.1f}, "
                      f"Z {record['val_Significance20_30_50_epoch']:.6f}", flush=True)
                records[epoch] = record
                changed = True
                epoch += 1
            if (folder / "test/test_scores.root").is_file() and not (folder / "evaluation/working_points.json").is_file():
                compare_working_points(folder, fold)
            done.append((folder / "exit_status.txt").is_file())
        if changed or not (run / "epochs.csv").is_file():
            write_table(run, epochs, previous)
        if all(done):
            return
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    monitor(args.run.resolve(), args.reference.resolve(), max(5, args.interval))
