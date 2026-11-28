"""Fit and apply the augmented recipe on a small stratified MC sample on the CPU batch node."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import uproot


def subset(source, dest):
    with uproot.open(source, handler=uproot.source.file.MemmapSource) as event_file, uproot.recreate(dest) as output:
        tree = event_file["tree"]
        cols = ["class_id", "partOrigin", "isWZ", "wAssignment", "hh_one_jet_valid",
                "hh_two_jet_valid", "hh_two_jet_compatible_fraction", "weight"]
        a = tree.arrays(cols, library="np")
        masks = [a["class_id"] == k for k in (0, 1, 2)]
        masks += [(a["partOrigin"][:, :3] == k).any(1) for k in (1, 2, 3, 4)]
        masks += [a["isWZ"].astype(bool), a["wAssignment"].astype(bool),
                  a["hh_one_jet_valid"] > a["hh_two_jet_valid"], a["hh_two_jet_compatible_fraction"] > 0]
        selected = []
        for mask in masks:
            indices = np.flatnonzero(mask & (a["weight"] > 0))
            if len(indices):
                # Keep the first 64 to amortise branch reads within the same ROOT baskets.
                selected.extend(indices[:64])
        rows = np.unique(selected)
        if not len(rows):
            raise ValueError(f"Empty CPU trial partition in {source}")
        writer = None
        for page in np.unique(rows // 1024):
            start = int(page) * 1024
            chunk = tree.arrays(entry_start=start, entry_stop=start+1024, library="np")
            sub = rows[rows // 1024 == page] - start
            values = {key: x[sub] for key, x in chunk.items()}
            if writer is None:
                writer = output.mktree("tree", {key: np.dtype((x.dtype, x.shape[1:])) for key, x in values.items()})
            writer.extend(values)
        return len(rows)


def main(run):
    frozen = run / "frozen"
    sys.path[:0] = [str(frozen / "recipe"), str(frozen)]
    from launch import configs, execute
    from hypothesis import FEATURES
    with TemporaryDirectory(prefix="synapse-hypothesis-check-") as tmp:
        folder = Path(tmp)
        names = ("3fold_0.root", "wzcr_0.root", "3fold_2.root", "3fold_1.root")
        report = {"mini_sample_rows": {name: subset(run / "dataset" / name, folder / name) for name in names}}
        checkpoint = None
        for stage in ("train", "validation", "test"):
            out, data, model, recipe = configs(frozen / "recipe", run / "dataset", folder / "out", 1, stage, checkpoint)
            for k in ("train_files", "val_files", "test_files"):
                data[k] = [str(folder / Path(f).name) for f in data[k]]
            inputs = data["inputs"]["evt_feats"]
            assert inputs[-len(FEATURES):] == list(FEATURES)
            assert len(inputs) == model["model_params"]["global_input_dim"] == 45
            recipe.update(device="cpu", epochs=1, batch_size=128, n_devices=1, num_workers=0)
            execute(out, data, model, recipe)
            if stage == "train":
                checkpoint, = (out / "checkpoints").glob("BEST-*.ckpt")
                continue
            cols = ["eventNumber", "dsid", "higgsPairKnown", "higgsPairSame"]
            with uproot.open(out / f"{stage}_scores.root") as prediction, uproot.open(data["test_files"][0]) as source:
                y = prediction["tree"].arrays([*cols, "p_signal"], library="np")
                x = source["tree"].arrays(cols, library="np")
                for k in cols:
                    np.testing.assert_array_equal(x[k], y[k])
                assert np.isfinite(y["p_signal"]).all() and ((0 <= y["p_signal"]) & (y["p_signal"] <= 1)).all()
                report[stage] = {"scored_rows": len(y["p_signal"])}
    (run / "smoke_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
