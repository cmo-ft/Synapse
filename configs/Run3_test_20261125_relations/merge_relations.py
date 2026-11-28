"""Publish impact-parameter-free folds with MC relation targets, retaining the original row order."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot

from prepare_relations import HERE, PREVIOUS, PLAN, PAIRS, PARENTS, VERSION, WZ, open_root, raw_mode


def assign(a, ids, pair, control, top_process, hh):
    n = len(pair)
    pdg, prompt = a["pdg"][:, :3], a["isPrompt"][:, :3].astype(bool)
    sfos = np.stack([pdg[:, i] == -pdg[:, j] for i, j in PAIRS], axis=1)
    wz = np.isin(a["dsid"], WZ)
    strict = (pair >= 0) & wz & control & (a["zPair"] < 0) & prompt.all(1) & (sfos.sum(1) == 2)
    if np.any(strict & ~sfos[np.arange(n), pair.clip(0, 2)]):
        raise ValueError("The mass assignment is not an SFOS pair in CAF order")
    z = a["zPair"].copy()
    z[strict] = pair[strict]
    unique = wz & prompt.all(1) & (sfos.sum(1) == 1)
    if np.any(unique & (z != sfos.argmax(1))):
        raise ValueError(
            "Frozen WZ flavour assignment disagrees with the unique reconstructed SFOS pair"
        )
    zz = (a["dsid"] == 701040) & (pair >= 0) & prompt.all(1)
    semi = a["dsid"] == 701105
    both_prompt = np.stack([prompt[:, i] & prompt[:, j] for i, j in PAIRS], axis=1)
    selected_semi = semi & (prompt.sum(1) == 2) & ((sfos & both_prompt).sum(1) == 1)
    chosen = np.where(zz, pair, np.where(selected_semi, (sfos & both_prompt).argmax(1), z))
    resolved = strict | unique | zz | selected_semi
    ids[resolved, 3, :3] = np.where(zz[resolved, None], 2, 0)
    for k, (i, j) in enumerate(PAIRS):
        use = resolved & (chosen == k)
        ids[use, 3, i] = ids[use, 3, j] = 1
    # A nonprompt or missing e/mu in semileptonic WZ carries no established parent ID.
    ids[:, 3, :3] = np.where(selected_semi[:, None] & ~prompt, -1, ids[:, 3, :3])

    # The fifth channel encodes only lepton-pair truth here. The loss labels every valid
    # pair involving a non-lepton False independently of these IDs.
    tt = np.full((n, 12), -1, np.int8)
    tt[hh | wz | (a["dsid"] == 701040), :3] = 0
    top = ids[:, 1, :3]
    evidence = np.stack(
        [
            prompt[:, i]
            & prompt[:, j]
            & (pdg[:, i] * pdg[:, j] < 0)
            & (top[:, i] > 0)
            & (top[:, j] > 0)
            & (top[:, i] != top[:, j])
            for i, j in PAIRS
        ],
        axis=1,
    )
    known = top_process & (evidence.sum(1) == 1)
    for k, (i, j) in enumerate(PAIRS):
        use = known & evidence[:, k]
        third = 3 - i - j
        tt[use, i] = tt[use, j] = 1
        tt[use & ~prompt[:, third], third] = 0
    for r, name in enumerate(PARENTS):
        a["relation" + name] = ids[:, r]
    a.update(zPair=z, zMassFallback=strict, relationTTbar=tt)


def read_labels(jobs, role, remainder):
    blocks = defaultdict(list)
    for j in jobs:
        f = HERE / "parent_shards" / f"{j['id']:06}.npz"
        with np.load(f) as src:
            if int(src["version"]) != VERSION or str(src["sample"]) != j["sample"]:
                raise ValueError(f)
            if role + "_id" not in src:
                continue
            ids = src[role + "_id"]
            use = ids % 3 == remainder
            if use.any():
                blocks[int(j["sample"].split("_")[0])].append(
                    (ids[use], *(src[role + "_" + key][use] for key in ("pt", "ids", "pair")))
                )
    return {dsid: tuple(np.concatenate(col) for col in zip(*parts)) for dsid, parts in blocks.items()}


def build_fold(label, remainder, jobs, top_ids, stage, rootdir):
    control = label == "wzcr"
    table = read_labels(jobs, "control" if control else "base", remainder)
    pos = Counter()
    summary = Counter()
    name = f"{label}_{remainder}.root"
    with open_root(PREVIOUS / "dataset" / name) as src, uproot.recreate(
        stage / name, compression=uproot.LZ4(4)
    ) as dest:
        tree = src["tree"]
        names = [k for k in tree.keys() if k not in ("impactD0", "impactZ0")]
        output = None
        for a in tree.iterate(names, step_size=20000, library="np"):
            dsid = a["dsid"]
            n = len(dsid)
            if np.any(a["eventNumber"] % 3 != remainder) or np.any(a["in_wzcr"] != control):
                raise ValueError("Reference fold or region mismatch")
            ids = np.full((n, 4, 12), -1, np.int8)
            pair = np.full(n, -1, np.int8)
            for d in np.unique(dsid):
                if d not in table:
                    continue
                used = dsid == d
                n_used = int(used.sum())
                sl = slice(pos[d], pos[d] + n_used)
                pos[d] += n_used
                event, pts, rel, index = table[d]
                np.testing.assert_array_equal(event[sl], a["eventNumber"][used])
                np.testing.assert_array_equal(pts[sl], a["pt"][used, :3])
                ids[used] = rel[sl]
                pair[used] = index[sl]
            assign(a, ids, pair, control, np.isin(dsid, top_ids), a["label_signal"].astype(bool))
            if output is None:
                output = dest.mktree("tree", {k: np.dtype((v.dtype, v.shape[1:])) for k, v in a.items()})
            output.extend(a)
            summary["events"] += n
            summary["mass_fallback"] += int(a["zMassFallback"].sum())
            summary["wz_assigned"] += int(
                (np.isin(dsid, WZ) & (a["zPair"] >= 0) & (a["zPair"] < 3)).sum()
            )
            ii, jj = np.triu_indices(12, 1)
            live = a["valid"].astype(bool)
            summary["TTbar_other_negative_pairs"] += int((live[:, ii] & live[:, jj] & (jj >= 3)).sum())
            for rel in (*PARENTS, "TTbar"):
                v = a["relation" + rel]
                u, w = v[:, ii], v[:, jj]
                yes = (u > 0) & (u == w)
                summary[rel + "_positive_pairs"] += int(yes.sum())
                summary[rel + "_negative_pairs"] += int(((u >= 0) & (w >= 0) & ~yes).sum())
            if control and np.any(a["zMassFallback"] & ~np.isin(dsid, WZ)):
                raise ValueError("Unexpected mass fallback process")
        if summary["events"] != tree.num_entries or any(pos[d] != len(v[0]) for d, v in table.items()):
            raise ValueError(f"Incomplete source coverage in {name}")
    if not control and summary["mass_fallback"]:
        raise ValueError("A base event received a mass fallback")
    if control and remainder == 2:
        audit = json.loads((rootdir / "wzcr_mass_gain.json").read_bytes())["total"]
        if (summary["mass_fallback"], summary["wz_assigned"]) != (
            audit["additional_pairs"],
            audit["combined_pairs"],
        ):
            raise ValueError(f"Disagreement with the independent WZCR audit: {summary}")
    print(name, dict(summary), flush=True)
    return dict(summary)


def main():
    target = HERE / "dataset"
    if target.exists():
        if not (target / "relation_counts.json").is_file():
            raise RuntimeError("Incomplete existing relation dataset")
        return
    plan = json.loads(PLAN.read_bytes())
    jobs = sorted(
        (j for j in plan["jobs"] if j["kind"] == "weights" and raw_mode(plan["samples"][j["sample"]])),
        key=lambda j: (j["sample"], j["id"]),
    )
    missing = [j["id"] for j in jobs if not (HERE / "parent_shards" / f"{j['id']:06}.npz").is_file()]
    if missing:
        raise RuntimeError(f"Waiting for {len(missing)} parent shards: {missing}")
    top_ids = sorted(
        {int(s["dsid"]) for s in plan["samples"].values() if s["process"] == "top" and not s["is_data"]}
    )
    with TemporaryDirectory(dir=HERE, prefix=".relations-merge-") as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        counts = {}
        for label in ("wzcr", "3fold"):
            for r in (2, 1, 0):
                counts[f"{label}_{r}"] = build_fold(label, r, jobs, top_ids, stage, HERE)
        (stage / "relation_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        summary = json.loads((PREVIOUS / "dataset/summary.json").read_bytes())
        summary["relation_target_version"] = VERSION
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(target)


if __name__ == "__main__":
    main()
