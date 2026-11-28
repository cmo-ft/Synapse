"""Read the two EasyJet lepton parent masks in the original selected-object order."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import awkward as ak
import numpy as np
import uproot

from paths import ANCESTRY_RUN, MATRIX_RUN, PHYSICS_RUN, PLAN, RUN, TREE_RUN

# A positive value is the one-based position of a single EasyJet bit.
NO_MATCH, MULTIPLE = 0, -1
ROLES = ("base", "control")
RECORD = np.dtype([("event", "u8"), ("pt", "f4", (3,)), ("higgs", "i1"), ("mask", "i1", (3,))])


def mc_jobs(plan):
    return sorted((j for j in plan["jobs"] if j["kind"] == "weights" and not plan["samples"][j["sample"]]["is_data"]),
                  key=lambda j: (j["sample"], j["id"]))


def source_records(tree):
    records, needed = {}, {}
    for role in ROLES:
        if role + "_id" not in tree:
            continue
        events = tree[role + "_id"]
        a = np.zeros(len(events), dtype=RECORD)
        a["event"], a["pt"], a["higgs"] = events, tree[role + "_pt"], tree[role + "_state"]
        records[role] = a
        rows = np.flatnonzero(a["higgs"] != 0)
        if rows.size:
            anc = tree[role + "_ancestry"][rows, :3]
            bits = tree[role + "_known"][rows, :2]
            missing = (anc < 0).any(1) | ((bits[:, 0] & 6) != 6) | ((bits[:, 1] & 4) != 4)
            rows = rows[missing]
        needed[role] = rows
    return records, needed


def check_keys(source, role, records):
    np.testing.assert_array_equal(source[role + "_id"], records["event"])
    np.testing.assert_array_equal(source[role + "_pt"], records["pt"])


def cached_masks(job, records, flag_path):
    filename = flag_path.name
    with np.load(flag_path) as flags, np.load(MATRIX_RUN / "parent_shards" / filename) as ids:
        for source in (flags, ids):
            if str(source["sample"]) != job["sample"] or int(source["version"]) != 1:
                raise ValueError(f"Incorrect mask source for {filename}")
        for role, a in records.items():
            check_keys(flags, role, a)
            check_keys(ids, role, a)
            positive = (flags[role + "_bits"][:, :3] & 1) != 0
            instance = ids[role + "_ids"][:, 0, :3]
            if np.any((instance > 0) & ~positive):
                raise ValueError(f"Unique mask ID without a positive mask in {filename}/{role}")
            a["mask"] = np.where(instance > 0, instance, np.where(positive, MULTIPLE, NO_MATCH))


def encode(bits):
    bits = int(bits)
    if bits == 0:
        return NO_MATCH
    return bits.bit_length() if bits.bit_count() == 1 else MULTIPLE


def raw_tracks(event):
    tracks = []
    for prefix, name, pdg in (("el", "Electron", 11), ("mu", "Muon", 13)):
        fields = ("pt_NOSYS", "eta", "phi", "charge", "isAnalysis" + name + "_NOSYS", "parentHiggsParentsMask")
        for pt, eta, phi, charge, selected, mask in zip(*(event[prefix + "_" + k] for k in fields), strict=True):
            if selected:
                tracks.append((pt, eta, phi, -pdg * round(charge), encode(mask)))
    return tracks


def align_tracks(tracks, wanted):
    if len(tracks) != 3:
        return None
    chosen = []
    for pt, eta, phi, pdg in wanted:
        matches = [k for k, (x, e, f, code, _) in enumerate(tracks)
                   if x == pt and code == pdg and abs(e - eta) < 1e-4
                   and abs(math.remainder(f - phi, 2 * math.pi)) < 1e-4]
        if len(matches) != 1:
            return None
        chosen.append(matches[0])
    if len(set(chosen)) != 3:
        return None
    return [tracks[k][-1] for k in chosen]


def read_supplement(job, records, needed):
    wanted = defaultdict(list)
    source = PHYSICS_RUN / "shards" / f"{job['id']:06}.root"
    with uproot.open(source, handler=uproot.source.file.MemmapSource) as root:
        for role, rows in needed.items():
            if not len(rows):
                continue
            a = root[role].arrays(["eventNumber", "pt", "eta", "phi", "pdg"], library="np")
            np.testing.assert_array_equal(a["eventNumber"], records[role]["event"])
            np.testing.assert_array_equal(a["pt"][:, :3], records[role]["pt"])
            for row in rows:
                tracks = list(zip(*(a[key][row, :3].tolist() for key in ("pt", "eta", "phi", "pdg"))))
                wanted[int(a["eventNumber"][row])].append((role, int(row), tracks))
    numbers = np.fromiter(wanted, dtype="u8")
    found = set()
    columns = ["eventNumber"]
    for prefix, name in (("el", "Electron"), ("mu", "Muon")):
        columns.extend(prefix + "_" + k for k in ("pt_NOSYS", "eta", "phi", "charge",
                       "isAnalysis" + name + "_NOSYS", "parentHiggsParentsMask"))
    for path in job["files"]:
        with uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None) as root:
            tree = root["AnalysisMiniTree"]
            hits = np.isin(tree["eventNumber"].array(library="np"), numbers)
            for start in range(0, len(hits), 32000):
                select = hits[start:start + 32000]
                if not select.any():
                    continue
                a = tree.arrays(columns, entry_start=start, entry_stop=start + len(select))
                for event in ak.to_list(a[select]):
                    tracks = raw_tracks(event)
                    for role, row, expected in wanted[event["eventNumber"]]:
                        mask = align_tracks(tracks, expected)
                        if mask is None:
                            continue
                        if (role, row) in found:
                            raise ValueError(f"Ambiguous EasyJet event in {job['sample']}/{event['eventNumber']}")
                        found.add((role, row))
                        records[role]["mask"][row] = mask
    if len(found) != sum(map(len, needed.values())):
        raise ValueError(f"Incomplete raw mask coverage in {job['sample']}: {len(found)} rows matched")


def supplemental_masks(job, records, needed):
    folder = RUN / "supplemental_masks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{job['id']:06}.npz"
    signature = hashlib.sha256(json.dumps(job["files"]).encode()).hexdigest()
    if path.is_file():
        with np.load(path) as data:
            if str(data["source"]) != signature or str(data["sample"]) != job["sample"] or int(data["version"]) != 1:
                raise ValueError(f"Incompatible supplemental mask file {path}")
            for role, a in records.items():
                rows = needed[role]
                np.testing.assert_array_equal(data[role + "_id"], a["event"][rows])
                np.testing.assert_array_equal(data[role + "_pt"], a["pt"][rows])
                a["mask"][rows] = data[role + "_mask"]
        return
    read_supplement(job, records, needed)
    values = dict(version=1, sample=job["sample"], source=signature)
    for role, a in records.items():
        rows = needed[role]
        for key, field in (("id", "event"), ("pt", "pt"), ("mask", "mask")):
            values[role + "_" + key] = a[field][rows]
    with NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp, **values)
    Path(tmp.name).replace(path)
    print(job["sample"], "matched supplemental rows", sum(map(len, needed.values())), flush=True)


def mask_index(plan):
    index = defaultdict(lambda: defaultdict(list))
    coverage = Counter()
    for job in mc_jobs(plan):
        filename = f"{job['id']:06}.npz"
        with np.load(TREE_RUN / "shards" / filename) as tree:
            if str(tree["sample"]) != job["sample"] or int(tree["version"]) != 1:
                raise ValueError(f"Incorrect decay-tree shard {filename}")
            records, needed = source_records(tree)
        required = sum(map(len, needed.values()))
        flag_path = ANCESTRY_RUN / "raw_labels" / filename
        if required and flag_path.is_file():
            cached_masks(job, records, flag_path)
            coverage["cached_rows_requiring_check"] += required
        elif required:
            supplemental_masks(job, records, needed)
            coverage["new_mask_rows_requiring_check"] += required
            coverage["new_mask_files"] += len(job["files"])
        coverage["source_jobs"] += 1
        dsid = int(plan["samples"][job["sample"]]["dsid"])
        for role, a in records.items():
            for rem in range(3):
                rows = a[a["event"] % 3 == rem]
                if len(rows):
                    index[role, rem][dsid].append(rows)
    return index, dict(coverage)


if __name__ == "__main__":
    plan = json.loads(PLAN.read_bytes())
    for job in mc_jobs(plan):
        filename = f"{job['id']:06}.npz"
        if (ANCESTRY_RUN / "raw_labels" / filename).is_file():
            continue
        with np.load(TREE_RUN / "shards" / filename) as tree:
            records, needed = source_records(tree)
        if any(map(len, needed.values())):
            supplemental_masks(job, records, needed)
