"""Build the Run 3 MC folds, keeping object truth as training targets only."""
import argparse
import json
import re
import shutil
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot
import yaml

RECIPE = Path(__file__).resolve().parent
REPO = RECIPE.parents[1]
RUN = REPO / "run_dir" / RECIPE.name
INPUT = REPO.parents[1] / "caf/hhml3l0tau/output/nominal-run3-20261122/ntuples/run3/parts/NOSYS"
sys.path.insert(0, str(RECIPE.parent / "run3_common"))
from preprocess import BACKGROUNDS, CLASSES, EVENET, JETS, LEPTONS, SM_HH, convert, object_branches

FLAGS = ("isConv", "isPrompt", "isQMisID", "isHF", "isLF")
RAW = ("truthType", "truthOrigin", "firstEgMotherPdgId")
PART_SLOTS = (*LEPTONS, "j1", "j2", "met")
REGIONS = ("sr", "extconvcr")


def discover(root):
    parts = []
    for directory in sorted(root.iterdir()):
        if directory.name.startswith("data"):
            continue
        if not re.fullmatch(r"\d{6}_c23[ade]", directory.name) or not directory.is_dir():
            raise ValueError(f"Unexpected sample directory: {directory}")
        selected = {region: {} for region in REGIONS}
        for path in sorted(directory.iterdir()):
            match = re.fullmatch(r"(\d+)_(sr|extconvcr)\.root", path.name)
            if match:
                job, region = match.groups()
                selected[region][job] = path
        if not selected["sr"] or selected["sr"].keys() != selected["extconvcr"].keys():
            raise ValueError(f"Missing SR or extconvCR partner in {directory}")
        for job in selected["sr"]:
            parts.extend(selected[region][job] for region in REGIONS)
    if not parts:
        raise ValueError(f"No MC parts in {root}")
    return parts


def stage_inputs(parts, scratch):
    # Keep a bounded number of compressed input files on local disk to avoid
    # frequent small EOS reads for each TTree branch.
    def fetch(path):
        local = scratch / (path.parent.name + "_" + path.name)
        shutil.copyfile(path, local)
        if local.stat().st_size != path.stat().st_size:
            raise IOError(f"Incomplete copy of {path}")
        return path, local

    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = iter(parts)
        pending = deque()
        for _ in range(4):
            path = next(paths, None)
            if path is not None:
                pending.append(pool.submit(fetch, path))
        while pending:
            path, local = pending.popleft().result()
            upcoming = next(paths, None)
            if upcoming is not None:
                pending.append(pool.submit(fetch, upcoming))
            try:
                yield path, local
            finally:
                local.unlink()


def add_object_truth(out, a):
    valid = out["valid"]
    origins = np.full(valid.shape, -1, dtype=np.int8)
    for obj, slot in zip(PART_SLOTS, (0, 1, 2, 3, 4, 9)):
        raw = a[f"{obj}_partOrigin"]
        maximum = 5 if obj in LEPTONS else (0 if obj == "met" else 3)
        if np.any((raw < 0) | (raw > maximum)):
            raise ValueError(f"Unexpected {obj}_partOrigin")
        origins[:, slot] = raw
    lepton_mask = np.zeros_like(valid)
    lepton_mask[:, :3] = valid[:, :3]
    origin_mask = np.zeros_like(valid)
    origin_mask[:, :5] = valid[:, :5]
    out.update(partOrigin=origins, partOrigin_mask=origin_mask, lepton_truth_mask=lepton_mask)
    for flag in FLAGS:
        values = np.column_stack([a[f"{obj}_{flag}"] for obj in LEPTONS])
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"Non-binary lepton label {flag}")
        slots = np.zeros_like(valid)
        slots[:, :3] = values
        out[flag] = slots
    for obj in LEPTONS:
        for field in RAW:
            name = f"{obj}_{field}"
            out[name] = a[name]


class FoldWriter:
    def __init__(self, stack, directory):
        self.files = [stack.enter_context(uproot.recreate(directory / f"3fold_{i}.root",
                      compression=uproot.LZ4(4))) for i in range(3)]
        self.trees = []
        self.chunks = [[], [], []]
        self.pending = [0, 0, 0]
        self.counts = np.zeros((2, 3, 3), dtype=np.int64)
        self.sumabs = np.zeros((3, 3), dtype=np.float64)
        self.invalid_teacher = np.zeros(3, dtype=np.int64)
        self.conv_leptons = np.zeros((3, 3), dtype=np.int64)

    def append(self, out, region):
        if not self.trees:
            schema = {k: np.dtype((v.dtype, v.shape[1:])) for k, v in out.items()}
            self.trees = [f.mktree("tree", schema) for f in self.files]
        remainder = out["eventNumber"] % 3
        for i in range(3):
            take = remainder == i
            if not take.any():
                continue
            chunk = {k: v[take] for k, v in out.items()}
            labels = chunk["class_id"]
            self.counts[REGIONS.index(region), i] += np.bincount(labels, minlength=3)
            self.sumabs[i] += np.bincount(labels, weights=np.abs(chunk["weight"]), minlength=3)
            self.invalid_teacher[i] += (~chunk["evenet_valid"]).sum()
            self.conv_leptons[i] += chunk["isConv"][:, :3].sum(axis=0)
            self.chunks[i].append(chunk)
            self.pending[i] += len(labels)
            if self.pending[i] >= 16000:
                self.flush(i)

    def flush(self, i):
        chunks = self.chunks[i]
        if chunks:
            self.trees[i].extend({k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]})
            chunks.clear()
            self.pending[i] = 0


def process_part(path, local, fields, evt, object_fields, writer):
    region = path.stem.split("_")[-1]
    dsid = int(path.parent.name.split("_")[0])
    kept = Counter()
    excluded = Counter()
    with uproot.open(local, handler=uproot.source.file.MemmapSource) as f:
        tree = f[region]
        metadata = json.loads(str(f["__hhml_metadata__"]))
        if (metadata["sample"], metadata["variation"], metadata["tree"], metadata["entries"]) != (
            path.parent.name, "NOSYS", region, tree.num_entries
        ):
            raise ValueError(f"Unexpected source metadata: {path}")
        missing = fields.difference(tree.keys())
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        for a in tree.iterate(sorted(fields), step_size=32000, library="np"):
            if not np.all(a["dsid"] == dsid):
                raise ValueError(f"DSID disagrees with MC sample {path}")
            process = a.pop("sample_process")
            tags, tag_counts = np.unique(process, return_counts=True)
            for tag, count in zip(tags, tag_counts):
                if "data" in tag.lower():
                    raise ValueError(f"Data process inside an MC input: {path}")
                dest = kept if tag in (*SM_HH, *BACKGROUNDS) else excluded
                dest[str(tag)] += int(count)
            signal = np.isin(process, SM_HH)
            select = signal | np.isin(process, BACKGROUNDS)
            if not select.any():
                continue
            if not select.all():
                a = {k: v[select] for k, v in a.items()}
                signal = signal[select]
            if not np.array_equal(a["is_conv"], a["mc_category"] == 4):
                raise ValueError(f"Event conversion truth inconsistent: {path}")
            out = convert(a, signal, evt, region, object_fields)
            add_object_truth(out, a)
            writer.append(out, region)
        if sum(kept.values()) + sum(excluded.values()) != tree.num_entries:
            raise ValueError(f"Event accounting failed: {path}")
    return dict(part=f"{path.parent.name}/{path.name}", region=region, kept=dict(kept),
                excluded=dict(excluded), entries=metadata["entries"], plan_digest=metadata["plan_digest"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT, help="Selected CAF MC part directories")
    parser.add_argument("--output", type=Path, default=RUN / "dataset", help="New directory for the three folds and summary")
    args = parser.parse_args()
    root, destination = args.input_dir, args.output
    if destination.exists():
        raise FileExistsError(destination)
    parts = discover(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((RECIPE / "data_config.yaml").read_text())
    evt = config["inputs"]["evt_feats"]
    objects = {k: object_branches(k, JETS) for k in ("pt", "eta", "phi", "e")}
    fields = set(evt) | set(EVENET) | {"sample_process", "dsid", "eventNumber", "weight", "is_conv",
                                     "mc_category", "met_met", "met_phi", "XGB"}
    for names in objects.values():
        fields.update(names)
    fields.update(f"{obj}_{k}" for obj in LEPTONS for k in ("pdg", "charge", *FLAGS, *RAW))
    fields.update(f"{obj}_partOrigin" for obj in PART_SLOTS)
    print(f"Processing {len(parts)} MC parts from {root.resolve()}", flush=True)
    with TemporaryDirectory(prefix="hhml-run3-folds-") as tmp:
        scratch = Path(tmp)
        build = scratch / "dataset"
        build.mkdir()
        manifest = []
        with ExitStack() as stack:
            writer = FoldWriter(stack, build)
            for i, (path, local) in enumerate(stage_inputs(parts, scratch), 1):
                manifest.append(process_part(path, local, fields, evt, objects, writer))
                if i % 50 == 0 or i == len(parts):
                    print(f"{i}/{len(parts)} parts, {writer.counts.sum():,} retained", flush=True)
            for i in range(3):
                writer.flush(i)

        totals = writer.counts.sum(axis=0)
        if np.any(totals <= 0) or np.any(writer.sumabs <= 0) or not np.isfinite(writer.sumabs).all():
            raise ValueError("Invalid class yields")
        if totals.sum() != sum(sum(p["kept"].values()) for p in manifest):
            raise ValueError("Output accounting differs from input")
        factors = writer.sumabs[2, 1] / writer.sumabs[2]
        config["weights"]["weight_balance_factors"] = factors.tolist()
        (build / "data_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        summary = dict(input_parts=str(root.resolve()), classes=CLASSES,
                       folds={"train": 2, "validation": 1, "test": 0},
                       object_slots=[*LEPTONS, *JETS, "met"], mc_parts=len(parts),
                       counts_by_remainder=totals.tolist(),
                       counts_by_region={k: v.tolist() for k, v in zip(REGIONS, writer.counts)},
                       sum_abs_weights_by_remainder=writer.sumabs.tolist(),
                       weight_balance_factors=factors.tolist(),
                       invalid_teacher_by_remainder=writer.invalid_teacher.tolist(),
                       conversion_leptons_by_remainder=writer.conv_leptons.tolist(),
                       partOrigin_mask="valid l1-l3 and j1-j2; j3-j6 unlabelled (-1), MET constant 0",
                       lepton_truth_mask="valid l1-l3 for isConv/isPrompt/isQMisID/isHF/isLF",
                       parts=manifest)
        (build / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with TemporaryDirectory(dir=destination.parent, prefix=".publish-") as tmp_publish:
            staged = Path(tmp_publish) / "dataset"
            shutil.copytree(build, staged)
            for i in range(3):
                name = f"3fold_{i}.root"
                if (staged / name).stat().st_size != (build / name).stat().st_size:
                    raise IOError(f"Incomplete EOS copy: {name}")
                with uproot.open(staged / name, handler=uproot.source.file.MemmapSource) as f:
                    if f["tree"].num_entries != totals[i].sum():
                        raise IOError(f"Entry count mismatch: {name}")
            staged.rename(destination)
    print(f"Ready: {destination}, fold sizes {totals.sum(axis=1).tolist()}", flush=True)


if __name__ == "__main__":
    main()
