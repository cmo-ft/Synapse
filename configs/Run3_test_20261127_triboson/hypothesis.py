"""Reconstructed visible masses for the two Higgs candidates in HH → 4W → 3ℓ + jets."""
from itertools import combinations

import numpy as np


INPUTS = ("pt", "eta", "phi", "e", "charge", "valid", "isJet")
FEATURES = ("hh_one_jet_valid", "hh_one_jet_bound", "hh_two_jet_valid", "hh_two_jet_bound",
            "hh_two_jet_mass", "hh_two_jet_compatible_fraction", "hh_hadronic_side_visible_mass",
            "hh_leptonic_side_visible_mass", "hh_leptonic_side_dr")


def mass(v):
    return np.sqrt(np.maximum(0, v[:, 0]**2 - np.square(v[:, 1:]).sum(1)))


def features(data):
    """Use the three reconstructed leptons and up to six central jets. Return masses in GeV."""
    pt, eta, phi, energy = [np.asarray(data[k][:, :9], dtype="f8") for k in ("pt","eta","phi","e")]
    pt = pt * .001
    energy = energy * .001
    valid = data["valid"][:, :9].astype(bool)
    jet = valid & data["isJet"][:, :9].astype(bool)
    charge = data["charge"][:, :3]
    v = np.stack((energy, pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)), axis=-1)
    n = len(pt)
    # Columns in best are the mass bound, semileptonic visible mass, complementary
    # dilepton mass, dilepton opening angle and dijet mass.
    best = {size: np.full((n,5),np.inf,dtype="f8") for size in (1,2)}
    trials = np.zeros(n, dtype=np.int16)
    compatible = np.zeros_like(trials)
    all_leptons = valid[:, :3].all(1)
    for left, right in combinations(range(3), 2):
        lepton = 3-left-right
        os = all_leptons & (charge[:,left]*charge[:,right] < 0)
        pair = v[:,left]+v[:,right]
        mll = mass(pair)
        dphi = np.remainder(phi[:,left]-phi[:,right]+np.pi, 2*np.pi)-np.pi
        dr = np.hypot(eta[:,left]-eta[:,right],dphi)
        for size in (1,2):
            for indices in combinations(range(3,9),size):
                eligible = os & jet[:,indices[0]]
                if size == 2:
                    eligible = eligible & jet[:,indices[1]]
                rows = np.flatnonzero(eligible)
                if not len(rows):
                    continue
                hadrons = v[rows, indices[0]]
                if size == 2:
                    hadrons = hadrons + v[rows, indices[1]]
                mlhad = mass(hadrons + v[rows,lepton])
                bound = np.maximum(mlhad, mll[rows])
                if size == 2:
                    trials[rows] += 1
                    compatible[rows] += bound <= 125.
                prior = best[size][rows]
                # When the complementary dilepton fixes the bound, use the smaller visible
                # semileptonic mass to distinguish candidates. Remaining ties keep slot order.
                choose = (bound < prior[:,0]) | ((bound==prior[:,0]) & (mlhad<prior[:,1]))
                if choose.any():
                    keep = rows[choose]
                    mjj = mass(hadrons[choose]) if size==2 else np.zeros(len(keep))
                    best[size][keep] = np.column_stack((bound[choose],mlhad[choose],mll[keep],dr[keep],mjj))
    one,two = best[1],best[2]
    has1,has2 = np.isfinite(one[:,0]),np.isfinite(two[:,0])
    selected = np.where(has2[:,None],two,one)
    out = (has1,np.where(has1,one[:,0],0),has2,np.where(has2,two[:,0],0),
           np.where(has2,two[:,4],0),compatible/np.maximum(trials,1),
           np.where(has1,selected[:,1],0),np.where(has1,selected[:,2],0),np.where(has1,selected[:,3],0))
    result = {k:arr.astype("f4") for k,arr in zip(FEATURES,out,strict=True)}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite reconstructed hypothesis observable")
    return result
