"""Rotate the three Run 3 MC remainders and freeze one fit recipe."""
import argparse
import json
import math
from pathlib import Path
import shutil

import yaml


def fold_roles(held_out):
    if held_out not in (0, 1, 2):
        raise ValueError(f"Invalid held-out fold {held_out}")
    return {"train": (held_out + 2) % 3, "validation": (held_out + 1) % 3, "test": held_out}


def training_factors(summary, train):
    weights = summary["sum_abs_weights_by_remainder"][train]
    if len(weights) != 3 or not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError(f"Invalid absolute class weights on training remainder {train}")
    return [weights[1] / w for w in weights]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="Frozen fold-0 recipe")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    roles = fold_roles(args.fold)
    summary = json.loads((args.dataset / "summary.json").read_text())
    data = yaml.safe_load((args.template / "data_config.yaml").read_text())
    for key, role in (("train_files", "train"), ("val_files", "validation"), ("test_files", "test")):
        filename = f"3fold_{roles[role]}.root"
        if not (args.dataset / filename).is_file():
            raise FileNotFoundError(args.dataset / filename)
        data[key] = [filename]
    data["weights"]["weight_balance_factors"] = training_factors(summary, roles["train"])
    run = yaml.safe_load((args.template / "run_fold_0.yaml").read_text())
    run["run_dir"] = f"output/fold_{args.fold}"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "data_config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (args.out / f"run_fold_{args.fold}.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    shutil.copy2(args.template / "model_config.yaml", args.out / "model_config.yaml")
    print(f"Fold {args.fold}: {roles}, event balance {data['weights']['weight_balance_factors']}")


if __name__ == "__main__":
    main()
