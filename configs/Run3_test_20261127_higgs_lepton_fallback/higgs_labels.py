"""Resolve binary lepton ancestry and same-Higgs pairs against the frozen decay-tree evidence."""
import numpy as np

LEPTON_PAIRS = ((0, 1), (0, 2), (1, 2))


def incompatible(count, ids, ancestry, known, same):
    """Test if all known labels admit an assignment to the recorded zero, one or two Higgs bosons."""
    if not np.isin(count, (0, 1, 2)).all():
        raise ValueError("Unexpected Higgs record multiplicity")
    n, slots = ancestry.shape
    active = (ancestry == 1) | (ids > 0) | (same != 0)
    for j in range(1, slots):
        active[:, j] |= ((same[:, :j] & (1 << j)) != 0).any(1)
    bad = (active & (ancestry == 0)).any(1) | ((count == 0) & active.any(1))
    # A component ID with a sign gives membership in one of the two Higgs bosons.
    # Fixed tree IDs share one component. With one Higgs, every positive must share it.
    component = np.broadcast_to(np.arange(1, slots + 1, dtype="i2"), (n, slots)).copy()
    anchor = slots + 1
    component[ids == 1] = anchor
    component[ids == 2] = -anchor
    component[(count == 1)[:, None] & active] = anchor
    for i in range(slots - 1):
        for j in range(i + 1, slots):
            bit = 1 << j
            rows = np.flatnonzero(active[:, i] & active[:, j] & ((known[:, i] & bit) != 0) & ~bad)
            if not len(rows):
                continue
            left, right = component[rows, i], component[rows, j]
            parity = np.where(same[rows, i] & bit, 1, -1)
            flip = np.sign(left) * np.sign(right) * parity
            joined = np.abs(left) == np.abs(right)
            bad[rows[joined & (flip < 0)]] = True
            merge = ~joined
            if merge.any():
                rr = rows[merge]
                values = component[rr]
                take = np.abs(values) == np.abs(right[merge, None])
                replacement = np.sign(values) * (np.abs(left[merge]) * flip[merge])[:, None]
                component[rr] = np.where(take, replacement, values)
    return bad


def apply_votes(ancestry, known, same, votes, equal):
    ancestry, known, same = ancestry.copy(), known.copy(), same.copy()
    ancestry[:, :3] = np.where(votes[:, :3], 1, ancestry[:, :3])
    for k, (i, j) in enumerate(LEPTON_PAIRS):
        use = votes[:, k + 3]
        known[use, i] |= 1 << j
        same[use & equal[:, k], i] |= 1 << j
    return ancestry, known, same


def keep_consistent(count, ids, ancestry, known, same, proposed, equal):
    """Keep tree labels first, then as many binary fallbacks and then pair fallbacks as possible.

    If two equally supported solutions disagree, leave the disputed additions unknown.
    Each row passed here has already failed with the full set of proposed fallbacks.
    """
    if incompatible(count, ids, ancestry, known, same).any():
        raise ValueError("A tree-only event has internally contradictory Higgs labels")
    best = np.zeros(len(ids), dtype=int)
    agreed = np.zeros_like(proposed)
    for subset in range(1, 64):
        candidate = ((subset >> np.arange(6)) & 1).astype(bool)
        rank = 4 * int(candidate[:3].sum()) + int(candidate[3:].sum())
        rows = np.flatnonzero((~candidate | proposed).all(1) & (best <= rank))
        if not len(rows):
            continue
        votes = np.broadcast_to(candidate, (len(rows), 6))
        y = apply_votes(ancestry[rows], known[rows], same[rows], votes, equal[rows])
        rows = rows[~incompatible(count[rows], ids[rows], *y)]
        tie = rows[best[rows] == rank]
        win = rows[best[rows] < rank]
        agreed[tie] &= candidate
        agreed[win] = candidate
        best[win] = rank
    return agreed


def resolve(a, mask, count):
    ancestry, known, same = (a[k] for k in ("ancestryHiggs", "higgsPairKnown", "higgsPairSame"))
    ids = a["relationHiggs"]
    lep = a["valid"][:, :3].astype(bool) & np.isin(np.abs(a["pdg"][:, :3]), (11, 13))
    proposed = np.zeros((len(mask), 6), dtype=bool)
    proposed[:, :3] = lep & (ancestry[:, :3] == -1) & (mask != 0)
    equal = np.zeros((len(mask), 3), dtype=bool)
    for k, (i, j) in enumerate(LEPTON_PAIRS):
        proposed[:, 3 + k] = (lep[:, i] & lep[:, j] & (mask[:, i] > 0) & (mask[:, j] > 0)
                               & ((known[:, i] & (1 << j)) == 0))
        equal[:, k] = mask[:, i] == mask[:, j]
    accepted = proposed.copy()
    rows = np.flatnonzero(proposed.any(1))
    if len(rows):
        y = apply_votes(ancestry[rows], known[rows], same[rows], accepted[rows], equal[rows])
        conflict = rows[incompatible(count[rows], ids[rows], *y)]
        if len(conflict):
            q = conflict
            accepted[q] = keep_consistent(count[q], ids[q], ancestry[q], known[q], same[q], proposed[q], equal[q])
            checked = apply_votes(ancestry[q], known[q], same[q], accepted[q], equal[q])
            if incompatible(count[q], ids[q], *checked).any():
                raise ValueError("Higgs labels still conflict after checking the proposed additions")
    result = apply_votes(ancestry, known, same, accepted, equal)
    return result, proposed, accepted, equal
