"""Save illustrative matched inputs and scores, or check the exported checkpoint on CPU."""
import argparse
import csv
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory

import numpy as np
import uproot


def write_tree(path, chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    with uproot.recreate(path, compression=uproot.LZ4(4)) as root:
        tree = root.mktree("tree", {k: np.dtype((v.dtype, v.shape[1:])) for k, v in arrays.items()})
        tree.extend(arrays)


def export_examples(run, package):
    sys.path.insert(0, str(run / "frozen/plots"))
    from process_catalog import GROUPS, process_codes

    dest = package / "examples"
    (dest / "dataset").mkdir(parents=True)
    catalog = run / "dataset/summary.json"
    copyfile(catalog, dest / "dataset/summary.json")
    copyfile(Path(__file__), dest / "check_checkpoint.py")
    groups = list(GROUPS.values())
    with (dest / "entries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "source_entry", "dsid", "eventNumber", "class_id", "process"])
        for split, remainder in (("validation", 2), ("test", 1)):
            fname = f"3fold_{remainder}.root"
            scored = run / f"output/fold_1/{split}/{split}_scores.root"
            options = dict(handler=uproot.source.file.MemmapSource, array_cache=None)
            with uproot.open(run / "dataset" / fname, **options) as source, uproot.open(scored, **options) as out:
                events, scores = source["tree"], out["tree"]
                if events.num_entries != scores.num_entries:
                    raise ValueError(f"Fold and application count differ for {split}")
                ids = events.arrays(["class_id", "dsid"], library="np")
                codes = process_codes(ids["dsid"], catalog)
                # One short window per process, plus one covering the external-conversion event class.
                masks = [codes == k for k in range(-1, len(groups))] + [ids["class_id"] == 2]
                indices = []
                for mask in masks:
                    hits = np.flatnonzero(mask)
                    mid = hits[len(hits) // 2]
                    start = max(0, min(mid - 16, len(codes) - 32))
                    indices.extend(range(start, start + 32))
                indices = np.unique(indices)
                windows = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
                inputs, outputs = [], []
                for window in windows:
                    read = dict(entry_start=int(window[0]), entry_stop=int(window[-1]) + 1, library="np")
                    a, b = events.arrays(**read), scores.arrays(**read)
                    for key in ("dsid", "eventNumber", "class_id", "weight", "n_jet"):
                        np.testing.assert_array_equal(a[key], b[key])
                    if not np.all(a["eventNumber"] % 3 == remainder):
                        raise ValueError(f"Unexpected fold in {split}")
                    for entry, dsid, event, cls in zip(window, a["dsid"], a["eventNumber"], a["class_id"]):
                        code = codes[entry]
                        writer.writerow([split, entry, dsid, event, cls, "HH" if code == -1 else groups[code]])
                    inputs.append(a)
                    outputs.append(b)
                write_tree(dest / "dataset" / fname, inputs)
                write_tree(dest / f"{split}_reference.root", outputs)


def check(package):
    import torch
    torch.set_num_threads(2)
    source = package / "source/frozen"
    sys.path[:0] = [str(source), str(source / "recipe")]
    from launch import configs, execute

    examples = package / "examples"
    with TemporaryDirectory() as scratch:
        for split in ("validation", "test"):
            out, data, model, run = configs(source / "recipe", examples / "dataset", Path(scratch),
                                            1, split, package / "model/best.ckpt")
            run.update(device="cpu", num_workers=0)
            execute(out, data, model, run)
            with uproot.open(examples / f"{split}_reference.root") as ref, uproot.open(out / f"{split}_scores.root") as test:
                expected = ref["tree"]
                actual = test["tree"]
                for name in ("eventNumber", "dsid", "class_id", "weight"):
                    np.testing.assert_array_equal(actual[name].array(library="np"), expected[name].array(library="np"))
                names = [k for k in expected.keys() if k.startswith("p_")]
                largest = 0.
                for name in names:
                    a, b = actual[name].array(library="np"), expected[name].array(library="np")
                    np.testing.assert_allclose(a, b, atol=5e-4, rtol=1e-4, err_msg=name)
                    largest = max(largest, float(np.max(np.abs(a - b))))
                print(f"{split}: {actual.num_entries} events, {len(names)} score branches, max absolute difference {largest:.3g}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Unpacked review package")
    check(parser.parse_args().package.resolve())
