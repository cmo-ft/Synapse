"""Build physics-aware MC shards from the frozen CAF selection and its indexed EasyJet inputs."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import uproot
import yaml

RECIPE = Path(__file__).resolve().parent
SYNAPSE = RECIPE.parents[1]
HERE = SYNAPSE / "run_dir" / RECIPE.name
COMMON = RECIPE.parent / "run3_common"
BASE = HERE.parent / "Run3_test_20261123"
BASE_RECIPE = RECIPE.parent / BASE.name
CAF = SYNAPSE.parents[1] / "caf/hhml3l0tau"
PRODUCTION = CAF / "work/nominal-run3-20261122"
PLAN = PRODUCTION / "plan/plan.json"
INPUT = CAF / "output/nominal-run3-20261122/ntuples/run3/parts/NOSYS"
sys.path.insert(0, str(BASE_RECIPE))
from prepare_folds import FLAGS, RAW, PART_SLOTS, add_object_truth
from preprocess import SM_HH, BACKGROUNDS, LEPTONS, JETS, EVENET, convert, object_branches

PAIRS = ((0, 1), (0, 2), (1, 2))
REGIONS = ("sr", "extconvcr", "wzcr")
VERSION = 1
G = yaml.safe_load((BASE_RECIPE / "data_config.yaml").read_text())["inputs"]["evt_feats"]
OBJECTS = {k: object_branches(k, JETS) for k in ("pt", "eta", "phi", "e")}
BASE_FIELDS = (
    set(G)
    | set(EVENET)
    | {
        "sample_process",
        "dsid",
        "eventNumber",
        "weight",
        "is_conv",
        "mc_category",
        "met_met",
        "met_phi",
        "XGB",
        "n_forward_jet",
    }
)
for names in OBJECTS.values():
    BASE_FIELDS.update(names)
BASE_FIELDS.update(
    f"{o}_{k}"
    for o in LEPTONS
    for k in ("pdg", "charge", "iso1", "iso2", "d0sig", "z0sintheta", *FLAGS, *RAW)
)
BASE_FIELDS.update(f"{o}_partOrigin" for o in PART_SLOTS)
BASE_FIELDS.update(f"j{i}_btag_score" for i in range(1, 5))


def open_root(path):
    return uproot.open(path, handler=uproot.source.file.MemmapSource)


def sample_cr(events, dsid):
    """One eighth of each MC process, independent of eventNumber % 3."""
    z = events.astype(np.uint64) ^ (np.uint64(dsid) << np.uint64(32))
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((z ^ (z >> np.uint64(31))) & np.uint64(7)) == 0


def read_friends(job, ids):
    import ROOT

    ROOT.EnableThreadSafety()
    ROOT.gInterpreter.AddIncludePath(str(COMMON))
    if not hasattr(ROOT, "synphysics"):
        if not ROOT.gInterpreter.Declare('#include "' + str(COMMON / "physics_objects.hxx") + '"'):
            raise RuntimeError("Could not load physics feature definitions")
    ROOT.synphysics.selectEvents(ROOT.VecOps.AsRVec(np.ascontiguousarray(np.unique(ids), dtype=np.uint64)))
    df = ROOT.RDataFrame("AnalysisMiniTree", job["files"]).Filter(
        "synphysics::wanted.count(eventNumber) != 0"
    )
    raw = {str(k) for k in df.GetColumnNames()}
    leptons = []
    for name, flavor, label in (("el", 11, "Electron"), ("mu", 13, "Muon")):
        fields = [
            name + "_" + n for n in ("pt_NOSYS", "eta", "phi", "charge", "isAnalysis" + label + "_NOSYS")
        ]
        alias = "physics_" + name
        df = df.Define(alias, "synphysics::selectedLeptons(" + ",".join(fields) + f",{flavor})")
        leptons.append(alias)
    df = df.Define("physics_lep", "h3::order(" + ",".join(leptons) + ")").Filter("physics_lep.size()==3")
    prefix = "recojet_antikt4PFlow_"
    jets = [
        prefix + k
        for k in ("pt_NOSYS", "eta", "phi", "pt_NOSYS", "isAnalysisJet_NOSYS", "isAnalysisJet_NOSYS")
    ]
    df = df.Define("physics_jets", "h3::jets(" + ",".join(jets) + ",physics_lep,false,true)")
    ancestors = [f"{name}_parent{parent}ParentsMask" for parent in ("Higgs", "Z") for name in ("el", "mu")]
    args = [n if n in raw else "ROOT::VecOps::RVec<int>{}" for n in ancestors]
    df = df.Define(
        "physics_extra",
        "synphysics::inputs(physics_lep,physics_jets,"
        + prefix
        + "ftag_quantile_GN2v01_Continuous,"
        + ",".join(args)
        + ")",
    )
    names = ["eventNumber", "nForward"]
    df = df.Define("nForward", "physics_extra.nForward")
    for field, size in (("lepPt", 3), ("higgs", 3), ("z", 3), ("quantile", 6), ("forward", 8)):
        for i in range(size):
            name = f"{field}{i}"
            df = df.Define(name, f"physics_extra.{field}[{i}]")
            names.append(name)
    return df.AsNumpy(names)


def identity(events, pts):
    return np.rec.fromarrays(
        [events.astype("u8"), *(x.astype("f4") for x in pts)], names=["id", "pt1", "pt2", "pt3"]
    )


class Matches:
    def __init__(self, extra):
        self.extra = extra
        key = identity(extra["eventNumber"], [extra[f"lepPt{i}"] for i in range(3)])
        self.order = np.argsort(key, kind="stable")
        self.keys = key[self.order]
        equal = self.keys[1:] == self.keys[:-1]
        if equal.any():
            for x in extra.values():
                if not np.array_equal(x[self.order[:-1][equal]], x[self.order[1:][equal]]):
                    raise ValueError(
                        "Conflicting EasyJet entries with the same event and three lepton momenta"
                    )

    def take(self, a):
        key = identity(a["eventNumber"], [a[f"{l}_pt"] for l in LEPTONS])
        pos = np.searchsorted(self.keys, key)
        if np.any(pos == len(self.keys)) or not np.array_equal(self.keys[pos], key):
            raise ValueError("Selected CAF event did not match the original EasyJet leptons")
        i = self.order[pos]
        return {k: v[i] for k, v in self.extra.items()}


def add_physics(out, a, f, process, region):
    n = len(out["pt"])
    central_valid = out["valid"][:, 3:9].copy()
    forward = np.column_stack([f[f"forward{i}"] for i in range(8)]).reshape(n, 2, 4)
    np.testing.assert_array_equal(f["nForward"], a["n_forward_jet"])
    fv = (forward[:, :, 0] > 0) & (forward[:, :, 3] > 0)
    for i, name in enumerate(("pt", "eta", "phi", "e")):
        out[name] = np.column_stack((out[name], np.where(fv, forward[:, :, i], 0))).astype("f4")
    # Retain the historical ten slot indices, including MET at slot nine.
    for name in (
        "valid",
        "isJet",
        "isMET",
        "pdg",
        "charge",
        "partOrigin",
        "partOrigin_mask",
        "lepton_truth_mask",
        *FLAGS,
    ):
        array = out[name]
        pad = (
            fv
            if name in ("valid", "isJet")
            else np.full((n, 2), -1 if name == "partOrigin" else 0, dtype=array.dtype)
        )
        out[name] = np.column_stack((array, pad)).astype(array.dtype)
    out["isForward"] = np.zeros((n, 12), dtype=bool)
    out["isForward"][:, 10:12] = fv
    quant = np.column_stack([f[f"quantile{i}"] for i in range(6)])
    for i in range(4):
        good = central_valid[:, i]
        np.testing.assert_array_equal(quant[good, i], a[f"j{i+1}_btag_score"][good])
    available = central_valid & (quant >= 1)
    out["tagAvailable"] = np.zeros((n, 12), dtype=bool)
    out["tagAvailable"][:, 3:9] = available
    out["tagQuantile"] = np.zeros((n, 12), dtype="f4")
    out["tagQuantile"][:, 3:9] = np.where(available, quant - 1, 0)
    for name, src in (
        ("caloIsoRel", "iso1"),
        ("trackIsoRel", "iso2"),
        ("impactD0", "d0sig"),
        ("impactZ0", "z0sintheta"),
    ):
        x = np.column_stack([a[f"{l}_{src}"] for l in LEPTONS]).astype("f4")
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite lepton measurement {src}")
        if src.startswith("iso"):
            x = x / np.maximum(out["pt"][:, :3], 1.0)
        x = np.arcsinh(x)
        out[name] = np.zeros((n, 12), dtype="f4")
        out[name][:, :3] = x
    h = np.column_stack([f[f"higgs{i}"] for i in range(3)])
    z = np.column_stack([f[f"z{i}"] for i in range(3)])
    pdg = out["pdg"][:, :3]
    sf = np.column_stack([pdg[:, i] == -pdg[:, j] for i, j in PAIRS])
    prompt = out["isPrompt"][:, :3].all(1)
    hh = process in SM_HH
    single = np.isin(h, [1, 2]).all(1) & (h.min(1) != h.max(1))
    hp = np.column_stack([h[:, i] == h[:, j] for i, j in PAIRS])
    hid = np.full(n, -1, dtype="i1")
    if hh:
        hid[single] = np.argmax(hp[single], axis=1)
    elif process in ("VV__WZ", "VV__otherVV", "VVV", "Vjets", "Vy", "top"):
        hid[(h == 0).all(1)] = 3
    zp = np.column_stack([(z[:, i] > 0) & (z[:, j] > 0) & ((z[:, i] & z[:, j]) != 0) for i, j in PAIRS]) & sf
    zid = np.full(n, -1, dtype="i1")
    if process == "VV__WZ":
        weak = prompt & (sf.sum(1) == 1)
        zid[weak] = np.argmax(sf[weak], axis=1)
    truth = prompt & (zp.sum(1) == 1)
    zid[truth] = np.argmax(zp[truth], axis=1)
    if hh:
        zid[prompt & (z == 0).all(1)] = 3
    out.update(
        higgsPair=hid, zPair=zid, in_wzcr=np.full(n, region == "wzcr"), n_forward_jet=a["n_forward_jet"]
    )


def paths(job):
    return {r: INPUT / job["sample"] / f"{job['id']:06}_{r}.root" for r in REGIONS}


def prepare(job, sample, output):
    dest = output / f"{job['id']:06}.root"
    if dest.exists():
        with open_root(dest) as f:
            m = json.loads(str(f["metadata"]))
        if m["version"] != VERSION or m["sample"] != job["sample"]:
            raise RuntimeError(f"Incompatible existing shard {dest}")
        return
    process = sample["process"]
    dsid = int(sample["dsid"])
    if sample["is_data"] or process not in (*SM_HH, *BACKGROUNDS):
        raise ValueError("Unexpected MC process")
    need = []
    for region, path in paths(job).items():
        with open_root(path) as f:
            ids = f[region]["eventNumber"].array(library="np")
            need.append(ids[sample_cr(ids, dsid)] if region == "wzcr" else ids)
    nneed = sum(map(len, need))
    match = Matches(read_friends(job, np.concatenate(need))) if nneed else None
    counts = Counter()
    truth_counts = Counter()
    with TemporaryDirectory() as tmp:
        local = Path(tmp) / dest.name
        with uproot.recreate(local, compression=uproot.LZ4(4)) as out:
            trees = {}
            for region, path in paths(job).items():
                role = "control" if region == "wzcr" else "base"
                with open_root(path) as f:
                    tree = f[region]
                    md = json.loads(str(f["__hhml_metadata__"]))
                    if (md["sample"], md["variation"], md["tree"], md["entries"]) != (
                        job["sample"],
                        "NOSYS",
                        region,
                        tree.num_entries,
                    ):
                        raise ValueError(f"Source metadata mismatch in {path}")
                    for a in tree.iterate(sorted(BASE_FIELDS), step_size=24000, library="np"):
                        if region == "wzcr":
                            keep = sample_cr(a["eventNumber"], dsid)
                            a = {k: v[keep] for k, v in a.items()}
                        n = len(a["eventNumber"])
                        if not n:
                            continue
                        if not np.all(a.pop("sample_process") == process) or not np.all(a["dsid"] == dsid):
                            raise ValueError(f"Process metadata mismatch in {path}")
                        arrays = convert(a, np.full(n, process in SM_HH), G, region, OBJECTS)
                        add_object_truth(arrays, a)
                        add_physics(arrays, a, match.take(a), process, region)
                        if role not in trees:
                            schema = {k: np.dtype((v.dtype, v.shape[1:])) for k, v in arrays.items()}
                            trees[role] = out.mktree(role, schema)
                        trees[role].extend(arrays)
                        counts[region] += n
                        for target in ("higgsPair", "zPair"):
                            for v, k in zip(*np.unique(arrays[target], return_counts=True)):
                                truth_counts[f"{region}/{target}/{v}"] += int(k)
            if sum(counts.values()) != nneed:
                raise ValueError("Mismatch between selected and published rows")
            meta = dict(
                version=VERSION, sample=job["sample"], process=process, counts=counts, targets=truth_counts
            )
            out["metadata"] = json.dumps(meta)
        with NamedTemporaryFile(
            dir=output, prefix="." + dest.stem + "-", suffix=".part", delete=False
        ) as stage:
            name = Path(stage.name)
        try:
            shutil.copyfile(local, name)
            with open_root(name) as check:
                if json.loads(str(check["metadata"])) != meta:
                    raise IOError("Shard metadata verification failed")
                for t, expected in (
                    ("base", counts["sr"] + counts["extconvcr"]),
                    ("control", counts["wzcr"]),
                ):
                    if expected and check[t].num_entries != expected:
                        raise IOError("Shard count mismatch")
            os.replace(name, dest)
        finally:
            name.unlink(missing_ok=True)
        print(f"Job {job['id']} {job['sample']} {dict(counts)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", type=int, nargs="+")
    ap.add_argument("--group", type=int)
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--output", type=Path, default=HERE / "shards", help="Directory for the per-job ROOT files")
    args = ap.parse_args()
    plan = json.loads(PLAN.read_bytes())
    jobs = [
        j
        for j in plan["jobs"]
        if not plan["samples"][j["sample"]]["is_data"]
        and plan["samples"][j["sample"]]["process"] in (*SM_HH, *BACKGROUNDS)
    ]
    if args.job is not None:
        jobs = [j for j in jobs if j["id"] in args.job]
    elif args.group is not None:

        def entries(j):
            info = plan["samples"][j["sample"]]["file_info"]
            return sum(info[f]["entries"] for f in j["files"])

        jobs = sorted(jobs, key=entries, reverse=True)[args.group :: args.groups]
    else:
        ap.error("Specify selected job IDs or a worker group")
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for job in jobs:
        try:
            prepare(job, plan["samples"][job["sample"]], args.output)
        except Exception:
            if args.group is None:
                raise
            import traceback

            traceback.print_exc()
            failed.append(job["id"])
    if failed:
        raise RuntimeError(f"Failed job IDs {failed}")


if __name__ == "__main__":
    main()
