#pragma once
#include "physics_objects.hxx"
#include <limits>

namespace synrelations {
using h3::Leptons;
using h3::RVec;
using Slots = std::array<int, 12>;
constexpr std::array<std::array<int, 2>, 3> pairs{{{0, 1}, {0, 2}, {1, 2}}};
inline Slots unknown() {
  Slots ids;
  ids.fill(-1);
  return ids;
}

template <class V> int instance(const V &masks, size_t index) {
  if (index >= masks.size())
    return -1;
  const int m = masks[index];
  if (m <= 0 || (m & (m - 1)))
    return -1;
  return 1 + __builtin_ctz(static_cast<unsigned>(m));
}

template <class E, class M, class J>
Slots maskIds(const Leptons &lep, const h3::Jets &jets, const RVec<float> &pt,
              const RVec<float> &eta, const RVec<char> &selected,
              const E &electrons, const M &muons, const J &jetMasks) {
  Slots result = unknown();
  for (size_t i = 0; i < 3; ++i)
    result[i] = std::abs(lep[i].pdg) == 11 ? instance(electrons, lep[i].index)
                                           : instance(muons, lep[i].index);
  for (size_t i = 0; i < std::min(size_t(6), jets.centralIndex.size()); ++i)
    result[3 + i] = instance(jetMasks, jets.centralIndex[i]);
  std::vector<size_t> forward;
  for (size_t i = 0; i < selected.size(); ++i)
    if (selected[i] && std::abs(eta[i]) >= 2.5)
      forward.push_back(i);
  std::stable_sort(forward.begin(), forward.end(),
                   [&](auto i, auto j) { return pt[i] > pt[j]; });
  for (size_t i = 0; i < std::min(size_t(2), forward.size()); ++i)
    result[10 + i] = instance(jetMasks, forward[i]);
  return result;
}

struct TruthLepton {
  int pdg;
  double px, py, pz, e, eta, phi;
};
template <class ID, class V>
std::vector<TruthLepton> truthLeptons(const ID &pdg, const V &px, const V &py,
                                      const V &pz, const V &e) {
  std::vector<TruthLepton> result;
  for (size_t i = 0; i < pdg.size(); ++i) {
    const double pt = std::hypot(double(px.at(i)), double(py.at(i)));
    result.push_back({pdg[i], px[i], py[i], pz.at(i), e.at(i),
                      pt > 0 ? std::asinh(pz[i] / pt)
                             : std::numeric_limits<double>::quiet_NaN(),
                      std::atan2(double(py[i]), double(px[i]))});
  }
  return result;
}
inline double deltaR(double eta, double phi, double otherEta, double otherPhi) {
  return std::hypot(eta - otherEta, std::remainder(phi - otherPhi, 2 * M_PI));
}

// The returned index is in CAF lepton order. A negative value leaves the label
// masked.
inline int continuumZ(const Leptons &reco, std::vector<TruthLepton> truth,
                      const std::vector<TruthLepton> &muons, bool zz) {
  truth.insert(truth.end(), muons.begin(), muons.end());
  if (truth.size() != (zz ? 4u : 3u))
    return -1;
  std::array<int, 3> match{-1, -1, -1};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < truth.size(); ++j) {
      const auto &t = truth[j];
      if (t.pdg == reco[i].pdg &&
          deltaR(t.eta, t.phi, reco[i].p4.Eta(), reco[i].p4.Phi()) < 0.1) {
        if (match[i] >= 0)
          return -1;
        match[i] = j;
      }
    }
    if (match[i] < 0 || std::find(match.begin(), match.begin() + i, match[i]) !=
                            match.begin() + i)
      return -1;
  }
  if (zz) {
    std::vector<int> ids;
    for (const auto &t : truth)
      ids.push_back(t.pdg);
    std::sort(ids.begin(), ids.end());
    if (ids != std::vector<int>{-13, -11, 11, 13})
      return -1;
  }
  int near = -1, distant = 0, nSFOS = 0;
  for (int k = 0; k < 3; ++k) {
    const auto &a = truth[match[pairs[k][0]]];
    const auto &b = truth[match[pairs[k][1]]];
    if (a.pdg != -b.pdg)
      continue;
    if (zz)
      return k; // Exactly one SFOS pair among the three observed daughters.
    ++nSFOS;
    const double e = a.e + b.e, x = a.px + b.px, y = a.py + b.py,
                 z = a.pz + b.pz;
    const double m2 = e * e - x * x - y * y - z * z;
    if (!std::isfinite(m2))
      return -1;
    const double distance =
        std::abs(std::sqrt(std::max(0., m2)) * 0.001 - 91.2);
    if (distance < 5.)
      near = k;
    if (distance > 15.)
      ++distant;
  }
  return nSFOS == 2 && near >= 0 && distant == 1 ? near : -1;
}

struct Decays {
  Slots w = unknown(), z = unknown();
};
struct Daughter {
  int pdg, boson;
  double eta, phi;
};
inline Slots matchDaughters(const std::vector<Daughter> &daughters,
                            const Leptons &lep, const h3::Jets &jets,
                            const RVec<float> &eta, const RVec<float> &phi,
                            bool allowMissing) {
  Slots ids = unknown();
  int matched = 0;
  for (const auto &d : daughters) {
    int candidate = -1, count = 0;
    if (std::abs(d.pdg) <= 5) {
      for (size_t i = 0; i < jets.centralIndex.size(); ++i) {
        auto j = jets.centralIndex[i];
        if (deltaR(d.eta, d.phi, eta[j], phi[j]) < 0.2) {
          candidate = i < 6 ? int(3 + i) : -1;
          ++count;
        }
      }
    } else {
      for (size_t i = 0; i < 3; ++i)
        if (d.pdg == lep[i].pdg &&
            deltaR(d.eta, d.phi, lep[i].p4.Eta(), lep[i].p4.Phi()) < 0.1) {
          candidate = i;
          ++count;
        }
    }
    if (!count && allowMissing)
      continue;
    if (count != 1 || candidate < 0 || ids[candidate] != -1)
      return unknown();
    ids[candidate] = d.boson;
    ++matched;
  }
  return matched == (allowMissing ? 3 : 5) ? ids : unknown();
}

template <class C, class I, class V>
Decays bosons(const Leptons &lep, const h3::Jets &jets, const RVec<float> &eta,
              const RVec<float> &phi, const C &c1, const I &id1, const V &eta1,
              const V &phi1, const C &c2, const I &id2, const V &eta2,
              const V &phi2) {
  Decays out;
  std::vector<Daughter> wd;
  int ww = 0, nCharged = 0, zz = 0;
  for (int h = 0; h < 2; ++h) {
    const auto &children = h == 0 ? c1 : c2;
    const auto &ids = h == 0 ? id1 : id2;
    const auto &etas = h == 0 ? eta1 : eta2;
    const auto &phis = h == 0 ? phi1 : phi2;
    if (children.size() != 2 || ids.size() != 2 || etas.size() != 2 ||
        phis.size() != 2)
      continue;
    const bool isWW =
        std::abs(children[0]) == 24 && children[0] == -children[1];
    const bool isZZ = children[0] == 23 && children[1] == 23;
    if (!isWW && !isZZ)
      continue;
    std::vector<Daughter> zd;
    if (isWW)
      ++ww;
    if (isZZ)
      ++zz;
    for (int j = 0; j < 2; ++j) {
      if (ids[j].size() != 2 || etas[j].size() != 2 || phis[j].size() != 2)
        continue;
      for (int k = 0; k < 2; ++k) {
        const int p = ids[j][k], other = ids[j][1 - k], ap = std::abs(p),
                  ao = std::abs(other);
        Daughter d{p, 2 * h + j + 1, etas[j][k], phis[j][k]};
        if (isZZ && (ap == 11 || ap == 13) && p == -other)
          zd.push_back(d);
        if (!isWW)
          continue;
        const int w = children[j];
        if ((ap == 11 || ap == 13) && p * w < 0 && ao == ap + 1 &&
            other * w > 0) {
          wd.push_back(d);
          ++nCharged;
        }
        const bool up = (ap == 2 || ap == 4) && p * w > 0;
        const bool down = (ap == 1 || ap == 3 || ap == 5) && p * w < 0;
        if ((up && (ao == 1 || ao == 3 || ao == 5) && other * w < 0) ||
            (down && (ao == 2 || ao == 4) && other * w > 0))
          wd.push_back(d);
      }
    }
    if (isZZ && zd.size() == 4)
      out.z = matchDaughters(zd, lep, jets, eta, phi, true);
  }
  if (zz != 1)
    out.z = unknown();
  if (ww == 2 && nCharged == 3 && wd.size() == 5)
    out.w = matchDaughters(wd, lep, jets, eta, phi, false);
  return out;
}
} // namespace synrelations
