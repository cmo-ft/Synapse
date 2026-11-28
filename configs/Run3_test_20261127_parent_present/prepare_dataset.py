"""Publish the event-presence pair labels with byte-for-byte checks of every output branch."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if HERE.parent.name == "configs" else HERE.parents[3]
RUN = REPO / "run_dir/Run3_test_20261127_parent_present"
REFERENCE = RUN.parent / "Run3_test_20261127_higgs_lepton_fallback"
HIGGS = RUN.parent / "Run3_test_20261127_higgs_tree"
PLAN = REPO.parents[1] / "caf/hhml3l0tau/work/nominal-run3-20261122/plan/plan.json"
fallback_recipe = REFERENCE / "frozen/recipe"
if not (fallback_recipe / "mask_sources.py").is_file():
    fallback_recipe = REPO / "configs" / REFERENCE.name
sys.path.insert(0, str(fallback_recipe))
from mask_sources import mc_jobs

STATE = np.dtype([("event", "u8"), ("pt", "f4", 3), ("state", "i1")])
POPCOUNT = np.array([i.bit_count() for i in range(1 << 12)], dtype="u1")
ROLES = {"base": "3fold", "control": "wzcr"}


def open_root(path):
    return uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None)


def presence_index(plan):
    groups = defaultdict(lambda: defaultdict(list))
    jobs = mc_jobs(plan)
    for job in jobs:
        dsid = int(plan["samples"][job["sample"]]["dsid"])
        with np.load(HIGGS / "shards" / f"{job['id']:06}.npz") as shard:
            if str(shard["sample"]) != job["sample"]:
                raise ValueError(f"Incorrect sample for job {job['id']}")
            for role in ROLES:
                if role + "_id" not in shard:
                    continue
                events = shard[role + "_id"]
                records = np.empty(len(events), dtype=STATE)
                records["event"], records["pt"], records["state"] = events, shard[role + "_pt"], shard[role + "_state"]
                if not np.isin(records["state"], (-1, 0, 1, 2)).all():
                    raise ValueError("Unexpected Higgs record multiplicity")
                remainder = events % 3
                for fold in range(3):
                    rows = records[remainder == fold]
                    if len(rows):
                        groups[role, fold][dsid].append(rows)
    print(f"Indexed the Higgs record counts from {len(jobs)} frozen MC shards", flush=True)
    return groups


def align(data, sources, offsets):
    states = np.empty(len(data["dsid"]), dtype="i1")
    for dsid in np.unique(data["dsid"]):
        take = data["dsid"] == dsid
        stop = offsets[dsid] + int(take.sum())
        rows = sources[dsid][offsets[dsid]:stop]
        np.testing.assert_array_equal(rows["event"], data["eventNumber"][take])
        np.testing.assert_array_equal(rows["pt"], data["pt"][take, :3])
        states[take] = rows["state"]
        offsets[dsid] = stop
    return states


def gate_pairs(a, states, counts):
    known, same = a["higgsPairKnown"], a["higgsPairSame"]
    present = states > 0
    if np.any(same & ~known) or np.any(same[~present]):
        raise ValueError("Unexpected positive or inconsistent Higgs truth")
    active = np.any(known != 0, axis=1)
    counts.update(rows=len(states), absent=int((states == 0).sum()), unknown=int((states < 0).sum()),
                  single_higgs=int((states == 1).sum()), double_higgs=int((states == 2).sum()),
                  higgs_events_before=int(active.sum()), higgs_events_after=int((active & present).sum()),
                  wz_higgs_events_after=int((a["isWZ"].astype(bool) & active & present).sum()),
                  higgs_positive_pairs=int(POPCOUNT[same].sum()), higgs_pairs_before=int(POPCOUNT[known].sum()))
    known[~present] = 0
    counts["higgs_pairs_after"] += int(POPCOUNT[known].sum())
    valid = a["valid"].astype(bool) & ~a["isMET"].astype(bool)
    for parent in ("W", "Z", "Top"):
        ids = np.where(valid, a["relation" + parent], -1)
        supervised = (ids >= 0).sum(1) >= 2
        if np.any(supervised & ~(ids > 0).any(1)):
            raise ValueError(f"{parent} pair supervision without evidence of a parent")
        counts[parent + "_events"] += int(supervised.sum())


def update_hashes(checks, arrays):
    for key, value in arrays.items():
        checks[key].update(np.ascontiguousarray(value).view("u1"))


def build_split(role, rem, pieces, stage, baseline):
    name = f"{ROLES[role]}_{rem}.root"
    table = {d: np.concatenate(rows) for d, rows in pieces.items()}
    pieces.clear()
    offsets, counts = Counter(), Counter()
    expected, actual = defaultdict(hashlib.sha256), defaultdict(hashlib.sha256)
    with open_root(REFERENCE / "dataset" / name) as src, uproot.recreate(stage / name, compression=uproot.LZ4(4)) as dest:
        tree = src["tree"]
        schema = tree.typenames()
        writer = None
        for a in tree.iterate(step_size=32000, library="np"):
            if np.any(a["eventNumber"] % 3 != rem) or np.any(a["in_wzcr"] != (role == "control")):
                raise ValueError(f"Misassigned rows in {name}")
            gate_pairs(a, align(a, table, offsets), counts)
            update_hashes(expected, a)
            if writer is None:
                writer = dest.mktree("tree", {k: np.dtype((v.dtype, v.shape[1:])) for k, v in a.items()})
            writer.extend(a)
        if any(offsets[d] != len(v) for d, v in table.items()) or counts["rows"] != tree.num_entries:
            raise ValueError(f"Mismatch in row coverage for {name}")
    for key, raw in (("rows", "events"), ("absent", "record_H_count/0"), ("single_higgs", "record_H_count/1"),
                     ("double_higgs", "record_H_count/2")):
        if counts[key] != baseline.get(raw, 0):
            raise ValueError(f"{name} Higgs record audit differs on {key}")
    with open_root(stage / name) as check:
        t = check["tree"]
        if t.typenames() != schema or t.num_entries != counts["rows"]:
            raise ValueError(f"Published schema or row count changed for {name}")
        for a in t.iterate(step_size=32000, library="np"):
            update_hashes(actual, a)
    if expected.keys() != actual.keys() or any(expected[k].digest() != actual[k].digest() for k in expected):
        raise ValueError(f"Output bytes do not reproduce source and requested label changes in {name}")
    print(name, dict(counts), flush=True)
    return {"counts": dict(counts), "sha256_by_branch": {k: d.hexdigest() for k, d in actual.items()}}


def main():
    target = RUN / "dataset"
    if target.exists():
        audit = json.loads((target / "audit.json").read_text())
        if audit.get("reference") != REFERENCE.name or set(audit.get("splits", {})) != {f"{p}_{r}" for p in ROLES for r in range(3)}:
            raise ValueError(f"Unrecognized existing publication at {target}")
        print("Existing completed dataset", target, flush=True)
        return
    RUN.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_bytes())
    inputs = presence_index(plan)
    source_audit = json.loads((HIGGS / "dataset/audit.json").read_text())["splits"]
    with TemporaryDirectory(dir=RUN, prefix=".publish-") as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        splits = {}
        for rem in (0, 2, 1):
            for role in ROLES:
                key = f"{role}_{rem}"
                splits[key] = build_split(role, rem, inputs.pop((role, rem)), stage, source_audit[key]["counts"])
        if inputs:
            raise ValueError("Unused source rows")
        fold1 = splits["base_0"]["counts"]
        required = dict(higgs_events_after=643081, wz_higgs_events_after=0, higgs_positive_pairs=1661748,
                        W_events=18150, Z_events=740422, Top_events=145465)
        if any(fold1[k] != value for k, value in required.items()):
            raise ValueError(f"Fold 1 disagrees with audited supervision: {fold1}")
        summary = json.loads((REFERENCE / "dataset/summary.json").read_bytes())
        summary.update(parent_presence_reference=REFERENCE.name, higgs_record_source=HIGGS.name)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (stage / "audit.json").write_text(json.dumps(dict(reference=REFERENCE.name, higgs_source=HIGGS.name,
            changed_branch="higgsPairKnown", splits=splits), indent=2) + "\n")
        stage.rename(target)


if __name__ == "__main__":
    main()
