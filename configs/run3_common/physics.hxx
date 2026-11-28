#ifndef HHML3L_PHYSICS_HXX
#define HHML3L_PHYSICS_HXX
#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <Math/VectorUtil.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace h3 {
using P4 = ROOT::Math::PtEtaPhiMVector;
using ROOT::VecOps::RVec;
struct Lepton {
  P4 p4;
  int index=0, pdg=0, charge=0, truthType=0, origin=0, mother=0;
  bool loose=false, tight=false, tightLoose=false;
};
using Leptons = std::vector<Lepton>;
struct Jets {
  std::vector<P4> central, forward;
  int nb=0;
  std::vector<size_t> centralIndex;
};
inline double dr(const P4 &a,const P4 &b) {return ROOT::Math::VectorUtil::DeltaR(a,b);}
template<class T> RVec<T> zeros(size_t n) {return RVec<T>(n,0);}
inline Leptons leptons(const RVec<float>& pt,const RVec<float>& eta,const RVec<float>& phi,
 const RVec<float>& charge,const RVec<char>& selected,const RVec<char>& loose,
 const RVec<char>& tight,const RVec<char>& tightLoose,const RVec<int>& type,
 const RVec<int>& origin,const RVec<int>& mother,int flavor) {
  Leptons out;
  for(size_t i=0;i<selected.size();++i) if(selected[i]) {
    Lepton l;
    l.p4=P4(pt.at(i),eta.at(i),phi.at(i),0.); // CAF uses massless leptons.
    l.index=i;l.charge=charge.at(i);l.pdg=-flavor*l.charge;
    l.loose=loose.at(i);l.tight=tight.at(i);l.tightLoose=tightLoose.at(i);
    l.truthType=type.at(i);l.origin=origin.at(i);l.mother=mother.at(i);
    out.push_back(l);
  }
  return out;
}
inline Leptons order(Leptons el,const Leptons& mu) {
  el.insert(el.end(),mu.begin(),mu.end());
  std::stable_sort(el.begin(),el.end(),[](const Lepton&a,const Lepton&b){return a.p4.Pt()>b.p4.Pt();});
  if(el.size()==3) {
    int q=el[0].charge+el[1].charge+el[2].charge;
    auto it=std::find_if(el.begin(),el.end(),[q](const Lepton&l){return l.charge*q<0;});
    if(it!=el.end()) {
      std::iter_swap(el.begin(),it);
      if(dr(el[0].p4,el[1].p4)>dr(el[0].p4,el[2].p4))std::swap(el[1],el[2]);
    }
  }
  return el;
}
inline Jets jets(const RVec<float>&pt,const RVec<float>&eta,const RVec<float>&phi,
 const RVec<float>&mass,const RVec<char>&selected,const RVec<char>&btag,
 const Leptons&l,bool useMass,bool electronCorrection) {
  Jets out;
  std::vector<std::pair<P4, size_t>> central;
  for(size_t i=0;i<selected.size();++i) if(selected[i]) {
    P4 p(pt.at(i),eta.at(i),phi.at(i),useMass?mass.at(i):0.);
    if(std::abs(eta.at(i))>=2.5) {out.forward.push_back(p);continue;}
    out.nb+=bool(btag.at(i));
    if(electronCorrection) {
      // CAF applies the first matched electron in descending pT order.
      const Lepton* match=nullptr;
      for(const auto& e:l) if(std::abs(e.pdg)==11 && dr(p,e.p4)<0.325 &&
         (!match || e.p4.Pt()>match->p4.Pt()))match=&e;
      if(match && p.Pt()>0)p*=std::max(0.,p.Pt()-1.2*match->p4.Pt())/p.Pt();
    }
    central.emplace_back(p, i);
  }
  auto cmp=[](const P4&a,const P4&b){return a.Pt()>b.Pt();};
  std::stable_sort(central.begin(), central.end(),
                   [](const auto &a, const auto &b){ return a.first.Pt() > b.first.Pt(); });
  out.central.reserve(central.size());
  out.centralIndex.reserve(central.size());
  for (const auto &[p4, index] : central) {
    out.central.push_back(p4);
    out.centralIndex.push_back(index);
  }
  std::stable_sort(out.forward.begin(),out.forward.end(),cmp);
  return out;
}
inline bool qmis(const Lepton&l){return std::abs(l.pdg)==11 && l.mother!=22 && l.mother*l.pdg<0;}
inline bool prompt(const Lepton&l){return (std::abs(l.pdg)==11 && l.truthType==2 && !qmis(l)) || (std::abs(l.pdg)==13 && l.truthType==6);}
inline bool conv(const Lepton&l){return !prompt(l)&&!qmis(l)&&l.origin==5&&l.truthType==4;}
inline bool hf(const Lepton&l){int o=l.origin;return o==26||o==29||o==33||o==25||o==27||o==32;}
inline bool lf(const Lepton&l){int o=l.origin;return o==42||o==23||o==24||o==30||o==31||o==34||o==35;}
// Bit mask preserves CAF's possible HF_e/HF_mu (and LF_e/LF_mu) overlap.
inline unsigned categories(const Leptons&l) {
  if(l.size()!=3)return 0;
  if(prompt(l[0])&&prompt(l[1])&&prompt(l[2]))return 1;
  if(qmis(l[0])||qmis(l[1])||qmis(l[2]))return 2;
  if(conv(l[1])||conv(l[2]))return 4;
  unsigned h=0;
  for(int i=1;i<3;++i) if(hf(l[i]))h|=std::abs(l[i].pdg)==11?8:16;
  if(h)return h;
  if(hf(l[0])||hf(l[1])||hf(l[2]))return 32;
  for(int i=1;i<3;++i) if(lf(l[i]))h|=std::abs(l[i].pdg)==11?64:128;
  if(h)return h;
  if(lf(l[0])||lf(l[1])||lf(l[2]))return 256;
  return 512;
}
inline double sf(const Leptons&l,const RVec<float>&el0,const RVec<float>&el12,
 const RVec<float>&mu0,const RVec<float>&mu12) {
  double w=1;
  for(size_t i=0;i<l.size();++i) {
    const auto&v=std::abs(l[i].pdg)==11?(i==0?el0:el12):(i==0?mu0:mu12);
    w*=v.at(l[i].index);
  }
  return w;
}
inline bool tight(const Leptons&l){return l.size()==3&&l[0].loose&&l[1].tight&&l[2].tight;}
inline bool loose(const Leptons&l){return l.size()==3&&l[0].loose&&l[1].tightLoose&&l[2].tightLoose;}
inline bool lowmass(const Leptons&l){return l.size()==3 &&
 (l[0].pdg!=-l[1].pdg||(l[0].p4+l[1].p4).M()>12000.) &&
 (l[0].pdg!=-l[2].pdg||(l[0].p4+l[2].p4).M()>12000.);}
inline bool zveto(const Leptons&l){return l.size()==3 &&
 !(l[0].pdg==-l[1].pdg&&std::abs((l[0].p4+l[1].p4).M()-91200.)<10000.) &&
 !(l[0].pdg==-l[2].pdg&&std::abs((l[0].p4+l[2].p4).M()-91200.)<10000.);}
inline P4 sum(const Leptons&l){P4 s;for(const auto&p:l)s+=p.p4;return s;}
inline P4 sum(const std::vector<P4>&v){P4 s;for(const auto&p:v)s+=p;return s;}
inline std::vector<float> xgbInputs(const Leptons&l,const Jets&j) {
  if(l.size()!=3)return std::vector<float>(27,0.f);
  auto mass=[](const P4&a,const P4&b){return float((a+b).M());};
  std::vector<float> v={float(l[0].pdg==-l[1].pdg),float(l[0].pdg==-l[2].pdg),
    float(l[0].pdg),float(l[1].pdg),float(l[2].pdg)};
  for(auto ij:{std::pair<int,int>{0,1},{0,2},{1,2}})v.push_back(mass(l[ij.first].p4,l[ij.second].p4));
  for(int k=0;k<2;++k)for(int i=0;i<3;++i)v.push_back(k<int(j.central.size())?mass(l[i].p4,j.central[k]):0.f);
  v.push_back(j.central.size()>1?mass(j.central[0],j.central[1]):0.f);
  v.push_back(sum(l).M());v.push_back((sum(l)+sum(j.central)).M());
  for(auto ij:{std::pair<int,int>{0,1},{0,2},{1,2}})v.push_back(dr(l[ij.first].p4,l[ij.second].p4));
  for(int k=0;k<2;++k)for(int i=0;i<3;++i)v.push_back(k<int(j.central.size())?dr(l[i].p4,j.central[k]):0.f);
  v.push_back(j.central.size()>1?dr(j.central[0],j.central[1]):0.f);
  return v;
}
inline double finiteWeight(double w){if(!std::isfinite(w))throw std::runtime_error("Non-finite event weight");return w;}
}
#endif
