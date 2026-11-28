"""Publish the six lepton-fallback folds atomically after matching every source row."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot

from higgs_labels import resolve
from mask_sources import RECORD, mask_index
from paths import PLAN, RUN, TREE_RUN


def align(a, sources, offsets):
    out = np.empty(len(a["dsid"]), dtype=RECORD)
    for d in np.unique(a["dsid"]):
        take = a["dsid"] == d
        start = offsets[d]
        stop = start + int(take.sum())
        source = sources[d][start:stop]
        np.testing.assert_array_equal(source["event"], a["eventNumber"][take])
        np.testing.assert_array_equal(source["pt"], a["pt"][take, :3])
        out[take] = source
        offsets[d] = stop
    return out


def counts(rows, masks, proposed, accepted, equal, origins):
    p, v = proposed[rows], accepted[rows]
    m, same = masks[rows], equal[rows]
    lost = p & ~v
    return Counter(events=int(rows.sum()), events_changed=int(v.any(1).sum()),
                   ancestry_events=int(v[:, :3].any(1).sum()), ancestry_added=int(v[:, :3].sum()),
                   ancestry_single_bit=int((v[:, :3] & (m > 0)).sum()),
                   ancestry_multibit=int((v[:, :3] & (m < 0)).sum()),
                   ancestry_other_origin=int((v[:, :3] & (origins[rows, :3] != 1)).sum()),
                   ll_same_added=int((v[:, 3:] & same).sum()), ll_different_added=int((v[:, 3:] & ~same).sum()),
                   mixed_conflict_events=int(lost.any(1).sum()), ancestry_withheld=int(lost[:, :3].sum()),
                   ll_same_withheld=int((lost[:, 3:] & same).sum()),
                   ll_different_withheld=int((lost[:, 3:] & ~same).sum()))


def build(role, rem, parts, process, stage):
    prefix = "3fold" if role == "base" else "wzcr"
    name = f"{prefix}_{rem}.root"
    sources = {d: np.concatenate(v) for d, v in parts.items()}
    parts.clear()
    offsets, total = Counter(), Counter()
    groups = defaultdict(Counter)
    examples = []
    with uproot.open(TREE_RUN / "dataset" / name, handler=uproot.source.file.MemmapSource, array_cache=None) as root, \
            uproot.recreate(stage / name, compression=uproot.LZ4(4)) as out:
        tree = root["tree"]
        writer = None
        for a in tree.iterate(step_size=20000, library="np"):
            if np.any(a["eventNumber"] % 3 != rem) or np.any(a["in_wzcr"] != (role == "control")):
                raise ValueError(f"Incorrect split or region in {name}")
            src = align(a, sources, offsets)
            result, proposed, accepted, equal = resolve(a, src["mask"], src["higgs"])
            disputed = (proposed != accepted).any(1)
            for row in np.flatnonzero(disputed)[:max(0, 6 - len(examples))]:
                examples.append(dict(dsid=int(a["dsid"][row]), event=int(a["eventNumber"][row]),
                                     nH=int(src["higgs"][row]), mask=src["mask"][row].tolist(),
                                     tree_ids=a["relationHiggs"][row].tolist(),
                                     old_ancestry=a["ancestryHiggs"][row].tolist(),
                                     tree_known=a["higgsPairKnown"][row].tolist(), tree_same=a["higgsPairSame"][row].tolist(),
                                     proposed=proposed[row].tolist(), accepted=accepted[row].tolist()))
            for d in np.unique(a["dsid"]):
                c = counts(a["dsid"] == d, src["mask"], proposed, accepted, equal, a["partOrigin"])
                groups[process[int(d)]].update(c)
                total.update(c)
            for key, value in zip(("ancestryHiggs", "higgsPairKnown", "higgsPairSame"), result, strict=True):
                a[key] = value
            if writer is None:
                writer = out.mktree("tree", {k: np.dtype((v.dtype, v.shape[1:])) for k, v in a.items()})
            writer.extend(a)
        if total["events"] != tree.num_entries or any(offsets[d] != len(s) for d, s in sources.items()):
            raise ValueError(f"Incomplete mask coverage in {name}")
    print(name, dict(total), flush=True)
    return dict(total=total, processes=groups, mixed_conflict_examples=examples)


def main():
    dest = RUN / "dataset"
    if dest.exists():
        if (dest / "audit.json").is_file():
            print("Already published", dest)
            return
        raise RuntimeError(f"Incomplete destination at {dest}")
    plan = json.loads(PLAN.read_bytes())
    process = {int(s["dsid"]): ("HH" if s["is_signal"] else s["process"]) for s in plan["samples"].values()
               if not s["is_data"]}
    RUN.mkdir(parents=True, exist_ok=True)
    index, coverage = mask_index(plan)
    print("Indexed all masks", coverage, flush=True)
    with TemporaryDirectory(dir=RUN, prefix=".publish-") as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        splits = {}
        for rem in (0, 2, 1):
            for role in ("base", "control"):
                splits[f"{role}_{rem}"] = build(role, rem, index.pop((role, rem)), process, stage)
        if index:
            raise ValueError("Unused mask source split")
        summary = json.loads((TREE_RUN / "dataset/summary.json").read_bytes())
        summary["higgs_lepton_fallback_version"] = 1
        summary["higgs_lepton_fallback_reference"] = TREE_RUN.name
        audit = dict(reference=TREE_RUN.name, mask_source_coverage=coverage, splits=splits,
                     vote_order=["l1", "l2", "l3", "l1l2", "l1l3", "l2l3"])
        for name, value in (("summary", summary), ("audit", audit)):
            (stage / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
        stage.rename(dest)
    print("Published", dest, flush=True)


if __name__ == "__main__":
    main()
