"""Classify groups of three or more objects sharing an old Higgs ID in SR + ExtConvCR."""
import csv
import json
from itertools import combinations

import numpy as np
import uproot

from paths import ANCESTRY_RUN, TREE_RUN, RUN, PLAN

SLOTS = ("l1","l2","l3","j1","j2","j3","j4","j5","j6","MET","fwd1","fwd2")
OPTIONS = dict(handler=uproot.source.file.MemmapSource, array_cache=None)


def group_rows(remainder, samples):
    name = f"dataset/3fold_{remainder}.root"
    with uproot.open(ANCESTRY_RUN/name, **OPTIONS) as source, uproot.open(TREE_RUN/name, **OPTIONS) as truth:
        old, ref = source["tree"], truth["tree"]
        if old.num_entries!=ref.num_entries:
            raise ValueError("Event counts differ")
        for start in range(0,old.num_entries,64000):
            stop = min(start+64000,old.num_entries)
            a = old.arrays(["eventNumber","dsid","relationHiggs","valid","isMET","wAssignment"],
                           entry_start=start, entry_stop=stop, library="np")
            ids = np.where(a["valid"].astype(bool) & ~a["isMET"].astype(bool),a["relationHiggs"],-1)
            counts = np.stack([(ids==p).sum(1) for p in (1,2)],axis=1)
            selected = np.flatnonzero(counts.max(1)>=3)
            if not len(selected):
                continue
            t = ref.arrays(["eventNumber","dsid","higgsPairKnown","higgsPairSame","relationHiggs"],
                           entry_start=start, entry_stop=stop, library="np")
            for k in ("eventNumber","dsid"):
                np.testing.assert_array_equal(t[k][selected],a[k][selected])
            for row in selected:
                dsid = int(a["dsid"][row])
                for p in np.flatnonzero(counts[row]>=3)+1:
                    group = np.flatnonzero(ids[row]==p)
                    pairs = list(combinations(group,2))
                    labels = ["unknown" if not ((int(t["higgsPairKnown"][row,i])>>int(j))&1)
                              else ("same" if (int(t["higgsPairSame"][row,i])>>int(j))&1 else "different")
                              for i,j in pairs]
                    outcome = ("inconsistent" if "different" in labels else
                               ("confirmed" if all(x=="same" for x in labels) else "unresolved"))
                    nlep = int((group<3).sum())
                    yield dict(remainder=remainder, entry=start+row, dsid=dsid, sample=samples[dsid]["description"],
                               process=samples[dsid]["process"], event=int(a["eventNumber"][row]),
                               old_parent=int(p), group_size=len(group), largest_group=int(counts[row].max()),
                               objects="l"*nlep+"j"*(len(group)-nlep), slots=" ".join(SLOTS[i] for i in group),
                               tree_agreement=outcome, tree_same=labels.count("same"), tree_different=labels.count("different"),
                               tree_unknown=labels.count("unknown"), tree_parent_ids=" ".join(str(x) for x in t["relationHiggs"][row,group]),
                               fully_matched_4W=int(a["wAssignment"][row]))


def main():
    plan = json.loads(PLAN.read_bytes())
    samples = {int(s["dsid"]):s for s in plan["samples"].values() if not s["is_data"]}
    dest = RUN/"old_higgs_three_object_groups.csv"
    temp = dest.with_suffix(".csv.tmp")
    with temp.open("w",newline="") as stream:
        writer = None
        for remainder in range(3):
            for row in group_rows(remainder,samples):
                if writer is None:
                    writer = csv.DictWriter(stream,fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
    temp.replace(dest)


if __name__=="__main__":
    main()
