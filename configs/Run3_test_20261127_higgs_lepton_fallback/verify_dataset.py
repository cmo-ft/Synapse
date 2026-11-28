"""Compare every ROOT branch and every entry against the tree-only folds."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import uproot

from paths import RUN, TREE_RUN

CHANGED = {"ancestryHiggs", "higgsPairKnown", "higgsPairSame"}
I, J = np.triu_indices(12, 1)
BITS = (1 << J).astype("i2")
LPAIRS = ((I < 3) & (J < 3))


def weighted_counts(known, positive, w):
    n = known.sum(1)
    use = n != 0
    return np.array([np.sum(w[use] * (positive[use].sum(1) / n[use])), w[use].sum()])


def check_file(name, reference_counts, factors):
    report = Counter()
    sums = defaultdict(lambda: np.zeros(2, dtype="f8"))
    digest = hashlib.sha256()
    with uproot.open(TREE_RUN / "dataset" / name, handler=uproot.source.file.MemmapSource, array_cache=None) as old, \
            uproot.open(RUN / "dataset" / name, handler=uproot.source.file.MemmapSource, array_cache=None) as new:
        ot, nt = old["tree"], new["tree"]
        if ot.keys() != nt.keys() or ot.typenames() != nt.typenames() or ot.num_entries != nt.num_entries:
            raise AssertionError(f"Schema, branch order or row count changed in {name}")
        same_keys = [k for k in ot.keys() if k not in CHANGED]
        for start in range(0, nt.num_entries, 20000):
            kwargs = dict(entry_start=start, entry_stop=min(start+20000, nt.num_entries), library="np")
            a, b = ot.arrays(**kwargs), nt.arrays(**kwargs)
            for k in same_keys:
                aa, bb = np.ascontiguousarray(a[k]).view("u1"), np.ascontiguousarray(b[k]).view("u1")
                if not np.array_equal(aa, bb):
                    raise AssertionError(f"Carried-over branch changed: {name}/{k}, starting row {start}")
                digest.update(bb)
            valid = a["valid"].astype(bool) & ~a["isMET"].astype(bool)
            lep = valid & np.isin(np.abs(a["pdg"]), (11, 13))
            lep[:, 3:] = False
            aa, ab = a["ancestryHiggs"], b["ancestryHiggs"]
            added = aa != ab
            if np.any(added & ~((aa == -1) & (ab == 1) & lep)):
                raise AssertionError("Unpermitted ancestry transition")
            ya, yb = [], []
            for d, y in ((a, ya), (b, yb)):
                known, same = d["higgsPairKnown"], d["higgsPairSame"]
                if np.any(same & ~known):
                    raise AssertionError("Positive Higgs pair with missing known bit")
                for i in range(12):
                    if np.any(known[:, i] & ((1 << (i + 1)) - 1)) or np.any(known[:, i] >> 12):
                        raise AssertionError("Pair bit outside upper triangle")
                y.extend(((known[:, I] & BITS) != 0, (same[:, I] & BITS) != 0))
                if np.any(y[0] & ~(valid[:, I] & valid[:, J])):
                    raise AssertionError("A masked slot has a supervised pair")
            known_a, same_a = ya
            known_b, same_b = yb
            fresh = known_b & ~known_a
            if np.any(known_a & (~known_b | (same_a != same_b))) or np.any(fresh & ~(LPAIRS & lep[:, I] & lep[:, J])):
                raise AssertionError("Higgs pair changed outside an unknown lepton pair")
            if np.any(a["wAssignment"].astype(bool)[:, None] & fresh):
                raise AssertionError("A fully resolved four-W lepton pair was newly labelled")
            report.update(events=len(aa), ancestry_added=int(added.sum()), ll_same_added=int((fresh & same_b).sum()),
                          ll_different_added=int((fresh & ~same_b).sum()),
                          events_changed=int((added.any(1) | fresh.any(1)).sum()))
            if factors is not None:
                w = np.abs(a["weight"].astype("f8")) * factors[a["class_id"]]
                for prefix, anc, known, same in (("tree", aa, *ya), ("fallback", ab, *yb)):
                    sums[prefix + "_ancestry"] += weighted_counts((anc >= 0) & valid, (anc == 1) & valid, w)
                    sums[prefix + "_pair"] += weighted_counts(known, same, w)
    for k, v in report.items():
        if reference_counts[k] != v:
            raise AssertionError(f"{name}/{k} independent count mismatch")
    return dict(counts=dict(report), branches_compared_bit_for_bit=len(same_keys),
                carried_branch_payload_sha256=digest.hexdigest(),
                effective_positive_fraction={k: float(v[0]/v[1]) for k, v in sums.items()})


def main():
    audit = json.loads((RUN / "dataset/audit.json").read_bytes())
    summary = json.loads((RUN / "dataset/summary.json").read_bytes())
    old_summary = json.loads((TREE_RUN / "dataset/summary.json").read_bytes())
    if {k:v for k,v in summary.items() if not k.startswith("higgs_lepton_fallback_")} != old_summary:
        raise AssertionError("Original sample and split catalog changed")
    files = {}
    for role, prefix in (("base", "3fold"), ("control", "wzcr")):
        for rem in range(3):
            sow = np.array(summary["sum_abs_weights_by_remainder"][rem])
            name = f"{prefix}_{rem}.root"
            files[name] = check_file(name, audit["splits"][f"{role}_{rem}"]["total"],
                                     sow[1] / sow if role == "base" else None)
            print(name, files[name]["counts"], files[name]["effective_positive_fraction"], flush=True)
    result = dict(reference=TREE_RUN.name, files=files,
                  summed_entries=sum(x["counts"]["events"] for x in files.values()),
                  root_bytes=sum((RUN/"dataset"/name).stat().st_size for name in files))
    dest = RUN / "dataset_integrity.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(dest)


if __name__ == "__main__":
    main()
