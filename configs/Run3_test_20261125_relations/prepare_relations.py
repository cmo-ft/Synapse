"""Add row-aligned MC parent identities to the frozen physics shards, using the CAF source index."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import uproot


RECIPE = Path(__file__).resolve().parent
HERE = RECIPE.parents[1] / "run_dir" / RECIPE.name
COMMON = RECIPE.parent / "run3_common"
PREVIOUS = HERE.parent / "Run3_test_20261124_physics"
PRODUCTION = HERE.parents[3] / "caf/hhml3l0tau/work/nominal-run3-20261122"
PLAN = PRODUCTION / "plan/plan.json"
PARENTS = ("Higgs", "Top", "W", "Z")
WZ = (701005, 701045)
PAIRS = ((0, 1), (0, 2), (1, 2))
VERSION = 1


def open_root(path):
    return uproot.open(path, handler=uproot.source.file.MemmapSource)


def raw_mode(sample):
    if sample["is_data"]:
        return None
    dsid = int(sample["dsid"])
    if dsid in WZ:
        return "wz"
    if dsid == 701040:
        return "zz"
    process = sample["process"]
    if sample["is_signal"] or process in ("singleH", "top"):
        return "parents"
    return None


def identity(numbers, pt):
    return np.rec.fromarrays(
        [numbers.astype("u8"), *pt[:, :3].astype("f4").T], names="event,pt1,pt2,pt3"
    )


def read_truth(job, mode, numbers, sample):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gInterpreter.AddIncludePath(str(COMMON))
    if not hasattr(ROOT, "synrelations"):
        if not ROOT.gInterpreter.Declare('#include "' + str(COMMON / "relation_targets.hxx") + '"'):
            raise RuntimeError("Could not compile the relation target definitions")
    ROOT.synphysics.selectEvents(
        ROOT.VecOps.AsRVec(np.ascontiguousarray(np.unique(numbers), dtype=np.uint64))
    )
    df = ROOT.RDataFrame("AnalysisMiniTree", job["files"]).Filter(
        "synphysics::wanted.count(eventNumber) != 0"
    )
    for kind, label, pdg in (("el", "Electron", 11), ("mu", "Muon", 13)):
        cols = [
            kind + "_" + s for s in ("pt_NOSYS", "eta", "phi", "charge", "isAnalysis" + label + "_NOSYS")
        ]
        df = df.Define("selected_" + kind, "synphysics::selectedLeptons(" + ",".join(cols) + f",{pdg})")
    df = df.Define("lep", "h3::order(selected_el,selected_mu)").Filter("lep.size()==3")
    columns = ["eventNumber"]
    for i in range(3):
        name = f"slotPt{i}"
        df = df.Define(name, f"float(lep[{i}].p4.Pt())")
        columns.append(name)
    if mode in ("wz", "zz"):
        for name in ("electron", "muon"):
            cols = ["truth" + name + "_" + s for s in ("pdgId", "px", "py", "pz", "e")]
            df = df.Define(name + "Truth", "synrelations::truthLeptons(" + ",".join(cols) + ")")
        expr = (
            "synrelations::continuumZ(lep,electronTruth,muonTruth,"
            + ("true" if mode == "zz" else "false")
            + ")"
        )
        df = df.Define("continuumPair", expr)
        columns.append("continuumPair")
    else:
        jet = "recojet_antikt4PFlow_"
        cols = [
            jet + s
            for s in ("pt_NOSYS", "eta", "phi", "pt_NOSYS", "isAnalysisJet_NOSYS", "isAnalysisJet_NOSYS")
        ]
        df = df.Define("jets", "h3::jets(" + ",".join(cols) + ",lep,false,true)")
        available = set(sample["branches"])
        for parent in ("Higgs", "Top"):
            names = [k + "_parent" + parent + "ParentsMask" for k in ("el", "mu", jet[:-1])]
            masks = [name if name in available else "ROOT::VecOps::RVec<int>{}" for name in names]
            cols = [jet + "pt_NOSYS", jet + "eta", jet + "isAnalysisJet_NOSYS", *masks]
            df = df.Define("ids" + parent, "synrelations::maskIds(lep,jets," + ",".join(cols) + ")")
        daughter = []
        for h in (1, 2):
            daughter.append(f"truth_children_fromH{h}_pdgId")
            daughter.extend(f"truth_grandchildren_fromH{h}_" + k for k in ("pdgId", "eta", "phi"))
        if all(k in available for k in daughter):
            df = df.Define(
                "decays",
                "synrelations::bosons(lep,jets,"
                + jet
                + "eta,"
                + jet
                + "phi,"
                + ",".join(daughter)
                + ")",
            )
        else:
            df = df.Define("decays", "synrelations::Decays{}")
        for parent, expr in (
            ("Higgs", "idsHiggs"),
            ("Top", "idsTop"),
            ("W", "decays.w"),
            ("Z", "decays.z"),
        ):
            for i in (*range(9), 10, 11):
                name = f"{parent}{i}"
                df = df.Define(name, f"int({expr}[{i}])")
                columns.append(name)
    return df.AsNumpy(columns)


def candidates(a, mode, role):
    pdg = a["pdg"][:, :3]
    n_sfos = sum((pdg[:, i] == -pdg[:, j]).astype(np.int8) for i, j in PAIRS)
    prompt = a["isPrompt"][:, :3].all(1)
    if mode == "wz":
        return (role == "control") & (a["zPair"] < 0) & prompt & (n_sfos == 2)
    if mode == "zz":
        return prompt & (n_sfos == 1)
    return np.ones(len(pdg), dtype=bool)


def build(job, sample, output):
    mode = raw_mode(sample)
    path = output / f"{job['id']:06}.npz"
    if path.exists():
        with np.load(path) as saved:
            if int(saved["version"]) != VERSION or str(saved["sample"]) != job["sample"]:
                raise ValueError(f"Incompatible label shard {path}")
        return
    source = {}
    wanted = {}
    with open_root(PREVIOUS / "shards" / f"{job['id']:06}.root") as file:
        for role in ("base", "control"):
            if role not in file:
                continue
            a = file[role].arrays(
                ["eventNumber", "pt", "pdg", "valid", "isPrompt", "zPair"], library="np"
            )
            source[role] = a
            wanted[role] = candidates(a, mode, role)
    numbers = [a["eventNumber"][wanted[r]] for r, a in source.items() if wanted[r].any()]
    raw = read_truth(job, mode, np.concatenate(numbers), sample) if numbers else {}
    if raw:
        key = identity(raw["eventNumber"], np.stack([raw[f"slotPt{i}"] for i in range(3)], axis=1))
        order = np.argsort(key, kind="stable")
        keys = key[order]
    result = dict(version=VERSION, sample=job["sample"])
    for role, a in source.items():
        ids = a["eventNumber"]
        length = len(ids)
        result[role + "_id"] = ids
        result[role + "_pt"] = a["pt"][:, :3].copy()
        relation = np.full((length, 4, 12), -1, np.int8)
        pair = np.full(length, -1, np.int8)
        rows = np.flatnonzero(wanted[role])
        if len(rows) and raw:
            dest = identity(ids[rows], a["pt"][rows])
            left = np.searchsorted(keys, dest)
            right = np.searchsorted(keys, dest, side="right")
            if (right == left).any():
                raise ValueError(
                    f"{job['sample']} {role}: {(right==left).sum()} selected rows lack an EasyJet match"
                )
            unique = right - left == 1
            selected = rows[unique]
            idx = order[left[unique]]
            if mode in ("wz", "zz"):
                pair[selected] = raw["continuumPair"][idx]
            else:
                for p, parent in enumerate(PARENTS):
                    for i in (*range(9), 10, 11):
                        relation[selected, p, i] = np.where(
                            a["valid"][selected, i], raw[f"{parent}{i}"][idx], -1
                        )
        result[role + "_ids"] = relation
        result[role + "_pair"] = pair
    output.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp, **result)
    Path(tmp.name).replace(path)
    print(
        f"{job['sample']} {job['id']} "
        + " ".join(
            f"{r}={len(a['eventNumber'])},chosen={(result[r+'_pair']>=0).sum()}"
            for r, a in source.items()
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--jobs", type=int, nargs="+")
    group.add_argument("--group", type=int, default=0)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument(
        "--modes", nargs="+", choices=("wz", "zz", "parents"), default=("wz", "zz", "parents")
    )
    p.add_argument("--output", type=Path, default=HERE / "parent_shards")
    args = p.parse_args()
    plan = json.loads(PLAN.read_bytes())
    jobs = [
        j
        for j in plan["jobs"]
        if j["kind"] == "weights" and raw_mode(plan["samples"][j["sample"]]) in args.modes
    ]
    if args.jobs:
        jobs = [j for j in jobs if j["id"] in args.jobs]
        if len(jobs) != len(set(args.jobs)):
            p.error("Unknown relation job")
    else:
        if not 0 <= args.group < args.groups:
            p.error("Invalid work group")
        jobs = jobs[args.group :: args.groups]
    for job in jobs:
        build(job, plan["samples"][job["sample"]], args.output)


if __name__ == "__main__":
    main()
