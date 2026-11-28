"""Check the materialized ancestry targets, including exceptions to geometric parent IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import uproot


def check(path):
    parents = ("Higgs", "Top", "W", "Z")
    columns = ["valid", "isMET", "isWZ", "isPrompt", "partOrigin", "dsid"]
    columns += [prefix + parent for parent in parents for prefix in ("ancestry", "relation")]
    counts, overrides = Counter(), Counter()
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as source:
        for data in source["tree"].iterate(columns, step_size=48000, library="np"):
            counts["events"] += len(data["dsid"])
            counts["WZ_process_events"] += int(data["isWZ"].sum())
            valid = data["valid"].astype(bool) & ~data["isMET"].astype(bool)
            for parent in parents:
                ancestry = data["ancestry" + parent]
                if np.any(ancestry[~valid] != -1) or not np.isin(ancestry[valid], (-1, 0, 1)).all():
                    raise ValueError(f"Invalid {parent} targets in {path}")
                positive = (data["relation" + parent] > 0) & valid
                counts[parent + "_positive_parent_IDs"] += int(positive.sum())
                rows, slots = np.where(positive & (ancestry != 1))
                counts[parent + "_positive_ID_overrides"] += len(rows)
                if not len(rows):
                    continue
                dsid, origin = data["dsid"][rows], data["partOrigin"][rows, slots]
                associated_w = np.isin(dsid, (601501, 601502)) & (origin == 4)
                top = np.isin(dsid, (602637, 602638, 604224)) & (origin == 2) & (data["ancestryTop"][rows, slots] == 1)
                verified = (parent == "Higgs") & (associated_w | top) & (slots < 3)
                verified &= data["isPrompt"][rows, slots].astype(bool) & (ancestry[rows, slots] == 0)
                if not verified.all():
                    bad = np.flatnonzero(~verified)[:5]
                    raise ValueError(f"Unexpected {parent} labels in {path} for {list(zip(dsid[bad], slots[bad], origin[bad]))}")
                overrides["associated_W_in_WH_HZZ4l"] += int(associated_w.sum())
                overrides["top_in_ttH"] += int(top.sum())
    return {"counts": counts, "independent_non_Higgs_leptons_with_positive_geometric_ID": overrides}


def main(run):
    result = {}
    for remainder in (2, 0, 1):
        for role in ("3fold", "wzcr"):
            name = f"{role}_{remainder}"
            result[name] = check(run / "dataset" / (name + ".root"))
            print(name, result[name], flush=True)
    path = run / "truth_consistency.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
