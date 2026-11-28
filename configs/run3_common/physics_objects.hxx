#pragma once

#include <array>
#include <unordered_set>
#include "physics.hxx"

namespace synphysics {
using h3::RVec;
inline std::unordered_set<ULong64_t> wanted;
template<class V> void selectEvents(const V &ids) {
  wanted.clear();
  wanted.insert(ids.begin(), ids.end());
}

inline h3::Leptons selectedLeptons(const RVec<float>& pt, const RVec<float>& eta,
    const RVec<float>& phi, const RVec<float>& q, const RVec<char>& selected, int flavor) {
  const auto zero = h3::zeros<int>(pt.size());
  return h3::leptons(pt, eta, phi, q, selected, selected, selected, selected,
                     zero, zero, zero, flavor);
}

struct Inputs {
  std::array<float, 3> lepPt{};
  std::array<int, 3> higgs{}, z{};
  std::array<int, 6> quantile{};
  std::array<float, 8> forward{};
  int nForward = 0;
};

template<class Q, class H1, class H2, class Z1, class Z2>
Inputs inputs(const h3::Leptons& lep, const h3::Jets& jets,
              const Q& tags, const H1& elH, const H2& muH, const Z1& elZ, const Z2& muZ) {
  Inputs out;
  for (size_t i = 0; i < 3; ++i) {
    const auto& l = lep.at(i);
    out.lepPt[i] = l.p4.Pt();
    const auto index = static_cast<size_t>(l.index);
    if (std::abs(l.pdg) == 11) {
      out.higgs[i] = index < elH.size() ? static_cast<int>(elH[index]) : -1;
      out.z[i] = index < elZ.size() ? static_cast<int>(elZ[index]) : -1;
    } else {
      out.higgs[i] = index < muH.size() ? static_cast<int>(muH[index]) : -1;
      out.z[i] = index < muZ.size() ? static_cast<int>(muZ[index]) : -1;
    }
  }
  out.quantile.fill(-1);
  for (size_t i = 0; i < std::min(out.quantile.size(), jets.centralIndex.size()); ++i)
    out.quantile[i] = tags.at(jets.centralIndex[i]);
  out.nForward = jets.forward.size();
  for (size_t i = 0; i < std::min(size_t(2), jets.forward.size()); ++i) {
    const auto& p = jets.forward[i];
    out.forward[4*i] = p.Pt(); out.forward[4*i+1] = p.Eta();
    out.forward[4*i+2] = p.Phi(); out.forward[4*i+3] = p.E();
  }
  return out;
}
}  // namespace synphysics
