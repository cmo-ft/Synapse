"""Compare EasyJet Higgs masks with decay-tree object and pair labels."""
import json
from collections import Counter, defaultdict

import numpy as np
import uproot

from paths import BASELINE, PHYSICS, PLAN, RELATIONS, RUN


def compare_pairs(mask_ids, valid, packed_known, packed_same, no_higgs):
    """Count each physical pair once when both objects have a one-bit EasyJet mask."""
    i, j = np.triu_indices(mask_ids.shape[1], 1)
    physical = (i != 9) & (j != 9)  # MET occupies slot 9.
    i, j = i[physical], j[physical]
    pair_type = (i >= 3).astype(int) + (j >= 3).astype(int)
    usable = valid & (mask_ids > 0)
    row, pair = np.nonzero(usable[:, i] & usable[:, j])
    left, right = i[pair], j[pair]
    ej_same = mask_ids[row, left] == mask_ids[row, right]
    known = ((packed_known[row, left] >> right) & 1).astype(bool)
    same = (packed_same[row, left] >> right) & 1
    tree = np.where(no_higgs[row], 0, np.where(known, same, -1))
    code = pair_type[pair] * 6 + (tree + 1) * 2 + ej_same
    table = np.bincount(code, minlength=18).reshape(3, 3, 2)
    return {
        name: {f"tree_{source}_mask_{prediction}": int(table[k, a, b])
               for a, source in enumerate(("unknown", "different", "same"))
               for b, prediction in enumerate(("different", "same"))}
        for k, name in enumerate(("lepton_lepton", "lepton_jet", "jet_jet"))
    }


def add_counts(counts, tree, flagged, tree_id, mask_id, origin, pairs):
    counts["events"] += len(tree)
    for name, value in (("unknown", -1), ("negative", 0), ("positive", 1)):
        selection = tree == value
        counts["tree_" + name] += int(selection.sum())
        counts["mask_positive_tree_" + name] += int((selection & flagged).sum())
        counts["mask_positive_origin_H_tree_" + name] += int((selection & flagged & (origin == 1)).sum())
    common = (tree_id > 0) & (mask_id > 0)
    counts["both_unique_Higgs_ID"] += int(common.sum())
    counts["same_unique_Higgs_ID"] += int((common & (tree_id == mask_id)).sum())
    # Seeing both Higgs bosons in the same event distinguishes a global swap from a local disagreement.
    sees_both = (common & (tree_id == 1)).any(axis=1) & (common & (tree_id == 2)).any(axis=1)
    direct_order = ((tree_id == mask_id) | ~common).all(axis=1)
    opposite_order = ((3 - tree_id == mask_id) | ~common).all(axis=1)
    counts["events_both_Higgs_observed"] += int(sees_both.sum())
    counts["events_same_order"] += int((sees_both & direct_order).sum())
    counts["events_swapped_order"] += int((sees_both & opposite_order).sum())
    counts["events_inconsistent_order"] += int((sees_both & ~direct_order & ~opposite_order).sum())
    i, j = np.triu_indices(3, 1)
    eligible = (mask_id[:, i] > 0) & (mask_id[:, j] > 0) & (pairs >= 0)
    geometric_same = mask_id[:, i] == mask_id[:, j]
    for name, value in (("different", 0), ("same", 1)):
        selection = eligible & (pairs == value)
        counts["pairs_tree_" + name] += int(selection.sum())
        counts["pairs_agree_tree_" + name] += int((selection & (geometric_same == value)).sum())


def compare_shard(job, groups, result, jet_result, pair_result):
    name = f"{job['id']:06}"
    with (np.load(BASELINE / "raw_labels" / (name + ".npz")) as raw,
          np.load(RELATIONS / "parent_shards" / (name + ".npz")) as previous,
          np.load(RUN / "shards" / (name + ".npz")) as new,
          uproot.open(PHYSICS / "shards" / (name + ".root"), handler=uproot.source.file.MemmapSource) as source):
        if "base_id" not in raw:
            return
        for data in (raw, previous, new):
            if str(data["sample"]) != job["sample"]:
                raise ValueError(f"Sample mismatch in {name}")
        event = new["base_id"]
        lepton_pt = new["base_pt"]
        a = source["base"].arrays(["eventNumber", "partOrigin", "valid", "pt"], library="np")
        for candidate in (raw["base_id"], previous["base_id"], a["eventNumber"]):
            np.testing.assert_array_equal(event, candidate)
        for candidate in (raw["base_pt"], previous["base_pt"], a["pt"][:, :3]):
            np.testing.assert_array_equal(lepton_pt, candidate)
        if not a["valid"][:, :3].all():
            raise ValueError("Every selected event must have three physical lepton candidates")
        origin = a["partOrigin"][:, :3]
        flags_all = (raw["base_bits"] & 1) != 0
        mask_ids_all = previous["base_ids"][:, 0]
        if np.any((mask_ids_all > 0) & ~flags_all):
            raise ValueError("A unique EasyJet Higgs ID must have a positive mask")
        tree_all = new["base_ancestry"]
        ids_all = new["base_ids"]
        no_higgs = new["base_state"] == 0
        tree_all[no_higgs] = 0
        ids_all[no_higgs] = 0
        i, j = np.triu_indices(3, 1)
        packed_known, packed_same = new["base_known"], new["base_same"]
        known = ((packed_known[:, i] >> j) & 1).astype(bool)
        same = (packed_same[:, i] >> j) & 1
        pairs = np.where(known, same, -1)
        pairs[no_higgs] = 0
        jet_valid = a["valid"].astype(bool)
        jet_valid[:, :3] = False
        jet_valid[:, 9] = False
        for split, rows in (("all_remainders", slice(None)), ("fold1_training", event % 3 == 0)):
            selection = [values[rows, :3] for values in (tree_all, flags_all, ids_all, mask_ids_all)]
            shard_counts = Counter()
            add_counts(shard_counts, *selection, origin[rows], pairs[rows])
            all_pairs = compare_pairs(mask_ids_all[rows], a["valid"][rows].astype(bool),
                                      packed_known[rows], packed_same[rows], no_higgs[rows])
            jets = Counter()
            for label, value in (("positive", 1), ("negative", 0), ("unknown", -1)):
                selected = jet_valid[rows] & (tree_all[rows] == value)
                jets["tree_" + label] = int(selected.sum())
                jets["mask_positive_tree_" + label] = int((selected & flags_all[rows]).sum())
            for group in groups:
                result[split][group].update(shard_counts)
                jet_result[split][group].update(jets)
                for name, pair_counts in all_pairs.items():
                    pair_result[split][group][name].update(pair_counts)


def main():
    plan = json.loads(PLAN.read_bytes())
    counts = defaultdict(lambda: defaultdict(Counter))
    jets = defaultdict(lambda: defaultdict(Counter))
    pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for job in plan["jobs"]:
        sample = plan["samples"][job["sample"]]
        if job["kind"] != "weights" or sample["is_data"]:
            continue
        if sample["is_signal"]:
            groups = ["HH"]
        elif sample["process"] in ("singleH", "top"):
            groups = [sample["process"]]
        else:
            continue
        if int(sample["dsid"]) in (604305, 604306):
            groups.append("inclusive_WH")
        compare_shard(job, groups, counts, jets, pairs)
    for split, groups in counts.items():
        for group, c in groups.items():
            ll = pairs[split][group]["lepton_lepton"]
            for answer in ("same", "different"):
                assert ll[f"tree_{answer}_mask_{answer}"] == c["pairs_agree_tree_" + answer]
                assert sum(ll[f"tree_{answer}_mask_{v}"] for v in ("same", "different")) == c["pairs_tree_" + answer]
    data = dict(selection="Three reconstructed leptons per SR or ExtConvCR MC event, unweighted. WH is a subset of singleH.",
                mask="Positive if EasyJet parentHiggsParentsMask is nonzero. Zero gives no positive evidence.",
                counts=counts, valid_jet_counts=jets, unique_mask_pair_counts=pairs)
    destination = RUN / "mask_agreement.json"
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
