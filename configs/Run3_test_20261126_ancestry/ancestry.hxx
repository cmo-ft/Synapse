#pragma once
#include "relation_targets.hxx"
#include <array>
#include <vector>

namespace ancestry {
using synrelations::Slots;
struct Object { int slot, pdg; double eta, phi; };
struct Daughter { int pdg, boson; double eta, phi; };

// Bits 0 and 1 carry positive geometric H/top matches. Bits 2 and 3 carry
// explicit H -> W/Z -> daughter matches, independent of those decorations.
template<class H, class T> int positiveMasks(const H &h, const T &t, size_t i) {
  return (i < h.size() && h[i] > 0 ? 1 : 0) | (i < t.size() && t[i] > 0 ? 2 : 0);
}

template<class E, class M, class J, class C, class I, class V>
Slots labels(const h3::Leptons &leptons, const h3::Jets &jets,
    const ROOT::VecOps::RVec<float> &pt, const ROOT::VecOps::RVec<float> &eta,
    const ROOT::VecOps::RVec<float> &phi, const ROOT::VecOps::RVec<char> &selected,
    const E &eh, const E &et, const M &mh, const M &mt, const J &jh, const J &jt,
    const C &c1, const I &id1, const V &eta1, const V &phi1,
    const C &c2, const I &id2, const V &eta2, const V &phi2) {
  Slots bits{};
  std::vector<Object> objects;
  for (int i=0; i<3; ++i) {
    const auto &l=leptons[i];
    bits[i]=std::abs(l.pdg)==11 ? positiveMasks(eh,et,l.index) : positiveMasks(mh,mt,l.index);
    objects.push_back({i,l.pdg,l.p4.Eta(),l.p4.Phi()});
  }
  std::vector<size_t> forward;
  for (size_t j=0; j<selected.size(); ++j)
    if (selected[j] && std::abs(eta[j])>=2.5) forward.push_back(j);
  std::stable_sort(forward.begin(),forward.end(),[&](size_t a,size_t b){return pt[a]>pt[b];});
  auto addJet=[&](size_t j,int slot) {
    if (slot>=0) bits[slot]=positiveMasks(jh,jt,j);
    objects.push_back({slot,0,eta[j],phi[j]});
  };
  for (size_t i=0; i<jets.centralIndex.size(); ++i)
    addJet(jets.centralIndex[i],i<6 ? int(3+i) : -1);
  for (size_t i=0; i<forward.size(); ++i) addJet(forward[i],i<2 ? int(10+i) : -1);

  std::vector<Daughter> daughters;
  for (int h=0; h<2; ++h) {
    const auto &children=h==0 ? c1 : c2;
    const auto &ids=h==0 ? id1 : id2;
    const auto &etas=h==0 ? eta1 : eta2;
    const auto &phis=h==0 ? phi1 : phi2;
    if (children.size()!=2 || ids.size()!=2 || etas.size()!=2 || phis.size()!=2) continue;
    for (size_t k=0; k<2; ++k) {
      const int boson=std::abs(children[k]);
      if ((boson!=23 && boson!=24) || ids[k].size()!=2 || etas[k].size()!=2 || phis[k].size()!=2) continue;
      for (size_t l=0; l<2; ++l) {
        const int id=ids[k][l], other=ids[k][1-l], a=std::abs(id), b=std::abs(other);
        const bool lepton=a==11 || a==13;
        const bool quark=a>=1 && a<=5;
        const bool chargedW=boson==24 && ((lepton && b==a+1 && id*children[k]<0 && other*children[k]>0)
          || (quark && b>=1 && b<=5 && ((a%2)==0) != ((b%2)==0) &&
              ((a%2)==0 ? id*children[k]>0 && other*children[k]<0 : id*children[k]<0 && other*children[k]>0)));
        if ((boson==23 && (lepton || quark) && id==-other) || chargedW)
          daughters.push_back({id,boson,etas[k][l],phis[k][l]});
      }
    }
  }
  std::vector<int> hits(objects.size()), matches(daughters.size(),-1);
  for (size_t k=0; k<daughters.size(); ++k) {
    const auto &d=daughters[k];
    int count=0;
    for (size_t j=0; j<objects.size(); ++j) {
      const auto &o=objects[j];
      const bool jet=std::abs(d.pdg)<=5;
      if ((jet ? o.pdg==0 : o.pdg==d.pdg) &&
          synrelations::deltaR(d.eta,d.phi,o.eta,o.phi)<(jet ? 0.2 : 0.1)) {
        ++hits[j]; ++count; matches[k]=int(j);
      }
    }
    if (count!=1) matches[k]=-1;
  }
  for (size_t k=0; k<daughters.size(); ++k) {
    const int j=matches[k];
    if (j>=0 && hits[j]==1 && objects[j].slot>=0)
      bits[objects[j].slot] |= daughters[k].boson==24 ? 4 : 8;
  }
  return bits;
}
}
