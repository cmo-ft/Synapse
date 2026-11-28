"""Assemble disjoint fold files and verify the SR and ExtConvCR populations against the previous run."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import numpy as np
import uproot

from prepare_physics import HERE, BASE, PLAN, SM_HH, BACKGROUNDS, open_root, VERSION


def main():
    dest = HERE / "dataset"
    if dest.is_dir():
        print(f"Published dataset at {dest}", flush=True)
        return
    plan = json.loads(PLAN.read_bytes())
    source = json.loads((BASE / "dataset/summary.json").read_bytes())
    jobs = sorted(
        (
            j
            for j in plan["jobs"]
            if not plan["samples"][j["sample"]]["is_data"]
            and plan["samples"][j["sample"]]["process"] in (*SM_HH, *BACKGROUNDS)
        ),
        key=lambda j: (j["sample"], j["id"]),
    )
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    sums = np.zeros((3, 3), dtype="f8")
    with TemporaryDirectory(prefix="synapse-physics-merge-") as td, ExitStack() as stack:
        scratch = Path(td)
        build = scratch / "build"
        build.mkdir()
        outputs = {}
        trees = {}
        buffers = {}
        sizes = {}
        for role, label in enumerate(("3fold", "wzcr")):
            for k in range(3):
                key = role, k
                outputs[key] = stack.enter_context(
                    uproot.recreate(build / f"{label}_{k}.root", compression=uproot.LZ4(4))
                )
                buffers[key] = []
                sizes[key] = 0

        def flush(key):
            if buffers[key]:
                arrays = {k: np.concatenate([b[k] for b in buffers[key]]) for k in buffers[key][0]}
                trees[key].extend(arrays)
                buffers[key].clear()
                sizes[key] = 0

        for index, j in enumerate(jobs, 1):
            shard = HERE / "shards" / f"{j['id']:06}.root"
            local = scratch / "input.root"
            shutil.copyfile(shard, local)
            with open_root(local) as f:
                md = json.loads(str(f["metadata"]))
                if md["version"] != VERSION or md["sample"] != j["sample"]:
                    raise ValueError(shard)
                for role, t in enumerate(("base", "control")):
                    if t not in f:
                        continue
                    for a in f[t].iterate(step_size=24000, library="np"):
                        if not np.all(a["in_wzcr"] == bool(role)):
                            raise ValueError(f"Region mismatch in {shard}")
                        rem = a["eventNumber"] % 3
                        for n in range(3):
                            cut = rem == n
                            if not np.any(cut):
                                continue
                            b = {k: v[cut] for k, v in a.items()}
                            key = role, n
                            if key not in trees:
                                schema = {k: np.dtype((v.dtype, v.shape[1:])) for k, v in b.items()}
                                trees[key] = outputs[key].mktree("tree", schema)
                            counts[role, n] += np.bincount(b["class_id"], minlength=3)
                            if role == 0:
                                sums[n] += np.bincount(
                                    b["class_id"], weights=np.abs(b["weight"]), minlength=3
                                )
                            buffers[key].append(b)
                            sizes[key] += len(b["weight"])
                            if sizes[key] >= 16000:
                                flush(key)
            local.unlink()
            if index % 50 == 0:
                print(f"Merged {index}/{len(jobs)} shards", flush=True)
        for key in buffers:
            flush(key)
        stack.close()
        np.testing.assert_array_equal(counts[0], source["counts_by_remainder"])
        np.testing.assert_allclose(sums, source["sum_abs_weights_by_remainder"], rtol=1e-10)
        for n in range(3):
            with open_root(build / f"3fold_{n}.root") as a, open_root(BASE / f"dataset/3fold_{n}.root") as b:
                for start in range(0, a["tree"].num_entries, 100000):
                    names = ["eventNumber", "dsid", "class_id", "weight"]
                    x, y = [
                        f["tree"].arrays(names, entry_start=start, entry_stop=start + 100000, library="np")
                        for f in (a, b)
                    ]
                    for k in names:
                        np.testing.assert_array_equal(x[k], y[k])
            with open_root(build / f"wzcr_{n}.root") as f:
                if f["tree"].num_entries != counts[1, n].sum():
                    raise ValueError("Wrong control remainder size")
        source.update(
            components_version=VERSION,
            train_only_control_counts=counts[1].tolist(),
            control_sampling="SplitMix64(eventNumber xor (dsid << 32)), low three bits zero",
            object_slots=source["object_slots"] + ["forward1", "forward2"],
            main_evaluation_domain=["sr", "extconvcr"],
            upstream_plan_digest=plan["digest"],
        )
        (build / "summary.json").write_text(json.dumps(source, indent=2) + "\n")
        with TemporaryDirectory(dir=HERE, prefix=".physics-publish-") as stage:
            path = Path(stage) / "dataset"
            shutil.copytree(build, path)
            for f in build.iterdir():
                if f.stat().st_size != (path / f.name).stat().st_size:
                    raise IOError(f.name)
            path.rename(dest)
    print(f"Published {counts.sum(axis=2).tolist()} MC events by role and fold", flush=True)


if __name__ == "__main__":
    main()
