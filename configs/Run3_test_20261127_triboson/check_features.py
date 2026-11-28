"""Cross-check candidate selection for invalid objects, an absent jet and all slot permutations."""
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import uproot

from hypothesis import FEATURES, INPUTS, features
from prepare_dataset import scalar_reference


def check(a):
    before = {name: arr.copy() for name, arr in a.items()}
    f = features(a)
    calculated = np.column_stack([f[k] for k in FEATURES])
    expected = [scalar_reference({k: arr[n] for k, arr in a.items()}) for n in range(len(calculated))]
    np.testing.assert_allclose(calculated, expected, rtol=2e-5, atol=3e-4)
    for name in a:
        np.testing.assert_array_equal(before[name], a[name])
    return calculated


def main(source):
    with uproot.open(source, handler=uproot.source.file.MemmapSource) as root:
        a = root["tree"].arrays(INPUTS, entry_stop=48, library="np")
    # A wide variety of assignments from real four-vectors, plus explicit boundary cases.
    a["valid"][:4] = False
    a["valid"][1:4, :3] = True
    a["valid"][2, 3] = True
    a["valid"][3, 3:5] = True
    a["isJet"][2:4, 3:5] = True
    a["charge"][:4, :3] = [-1, 1, 1]
    baseline = check(a)
    np.testing.assert_array_equal(baseline[:4, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(baseline[:4, 2], [0, 0, 0, 1])
    order = np.arange(a["pt"].shape[1])
    for perm in permutations(range(3)):
        order[:3] = perm
        order[3:9] = [7, 4, 8, 6, 3, 5]
        relabelled = {k: x[:, order] for k, x in a.items()}
        np.testing.assert_allclose(check(relabelled), baseline, rtol=2e-5, atol=3e-4)
    print(f"{len(baseline)} events checked against scalar enumeration and six slot permutations.", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
