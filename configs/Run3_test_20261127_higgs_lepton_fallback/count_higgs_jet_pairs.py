"""Count unweighted same-Higgs jet pairs in the two published training datasets."""
import csv
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import uproot

from paths import ANCESTRY_RUN, RUN


JET_SLOTS = (3, 4, 5, 6, 7, 8, 10, 11)
LEFT, RIGHT = np.asarray([(i, j) for i in JET_SLOTS for j in JET_SLOTS if i < j]).T


def count_file(old_file, new_file, catalog):
    from process_catalog import GROUPS, process_codes

    names = ("HH", *GROUPS.values())
    result = {name: {version: Counter() for version in ("previous", "current")} for name in names}
    options = dict(handler=uproot.source.file.MemmapSource, array_cache=None)
    with ExitStack() as stack:
        old, new = [stack.enter_context(uproot.open(p, **options))["tree"] for p in (old_file, new_file)]
        if old.num_entries != new.num_entries:
            raise ValueError(f"Different event counts in {old_file.name}")
        group = process_codes(old["dsid"].array(library="np"), catalog) + 1
        for start in range(0, old.num_entries, 64000):
            end = min(start+64000, old.num_entries)
            def read(tree, fields):
                return tree.arrays(fields, entry_start=start, entry_stop=end, library="np")
            a = read(old, ["dsid", "eventNumber", "valid", "isMET", "wAssignment", "relationW", "relationHiggs"])
            b = read(new, ["dsid", "eventNumber", "higgsPairKnown", "higgsPairSame"])
            if any(not np.array_equal(a[k], b[k]) for k in ("dsid", "eventNumber")):
                raise ValueError(f"Event identities differ in {old_file.name}")
            jets = a["valid"].astype(bool) & ~a["isMET"].astype(bool)
            eligible = jets[:, LEFT] & jets[:, RIGHT]
            w = a["relationW"]
            wjj = eligible & a["wAssignment"].astype(bool)[:, None] & (w[:, LEFT] > 0) & (w[:, LEFT] == w[:, RIGHT])
            ids = a["relationHiggs"]
            old_known = eligible & (ids[:, LEFT] >= 0) & (ids[:, RIGHT] >= 0)
            old_same = old_known & (ids[:, LEFT] > 0) & (ids[:, LEFT] == ids[:, RIGHT])
            new_known = eligible & (((b["higgsPairKnown"][:, LEFT] >> RIGHT) & 1) != 0)
            new_same = new_known & (((b["higgsPairSame"][:, LEFT] >> RIGHT) & 1) != 0)
            groups = group[start:end]
            for version, known, same in (("previous", old_known, old_same), ("current", new_known, new_same)):
                values = {
                    "known_pairs": known.sum(1),
                    "same_pairs": same.sum(1),
                    "events_with_same_pair": same.any(1),
                    "verified_W_jj_pairs": wjj.sum(1),
                    "verified_W_jj_labelled_same_H": (wjj & same).sum(1),
                }
                for key, per_event in values.items():
                    counts = np.bincount(groups, weights=per_event, minlength=len(names))
                    for name, n in zip(names, counts):
                        result[name][version][key] += int(n)
    return result


def main():
    import sys
    sys.path.insert(0, str(RUN/"frozen/plots"))
    rows = []
    for prefix, sample in (("3fold", "SR+ExtConvCR"), ("wzcr", "WZCR")):
        for remainder in range(3):
            filename = f"{prefix}_{remainder}.root"
            counts = count_file(ANCESTRY_RUN/"dataset"/filename, RUN/"dataset"/filename,
                                ANCESTRY_RUN/"dataset/summary.json")
            for process, models in counts.items():
                for version, data in models.items():
                    rows.append(dict(sample=sample, remainder=remainder, process=process, version=version, **data))
    dest = RUN/"higgs_jet_pair_counts.csv"
    temp = dest.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(dest)


if __name__ == "__main__":
    main()
