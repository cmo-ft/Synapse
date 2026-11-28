#pragma once
#include "relation_targets.hxx"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace higgsdecay {
using Slots = synrelations::Slots;
using h3::RVec;
constexpr int none = 1; // Option bits: nonmember, H1, H2.
constexpr int both = 6;

struct Object { int slot, kind; double eta, phi; bool prompt; };
struct Daughter { int parent, kind; double eta, phi; };
struct Result {
  Slots ids{}, ancestry{}, known{}, same{};
  int conservative = 0;
  Result() { ids.fill(-1); ancestry.fill(-1); }
};
struct Inventory {
  std::vector<Daughter> daughters;
  std::array<int,5> open{};
};

inline int kind(int pdg) {
  if (pdg == 11) return 1;
  if (pdg == -11) return 2;
  if (pdg == 13) return 3;
  if (pdg == -13) return 4;
  return 0; // quark or gluon for the hard-decay jet target
}
inline bool parton(int pdg) {
  return (std::abs(pdg) >= 1 && std::abs(pdg) <= 5) || pdg == 21;
}
inline void openAll(Inventory& inv, int parent) {
  for (auto& mask : inv.open) mask |= parent;
}
inline void openTau(Inventory& inv, int parent, int tau) {
  const int sign = tau > 0 ? 1 : -1;
  inv.open[kind(sign*11)] |= parent;
  inv.open[kind(sign*13)] |= parent;
  inv.open[0] |= parent;
}
inline bool bosonPair(int boson, int a, int b) {
  if (boson == 23) return a == -b && ((std::abs(a) >= 1 && std::abs(a) <= 5) ||
                                         (std::abs(a) >= 11 && std::abs(a) <= 16));
  const int up = a*boson > 0 ? a : b, down = a*boson > 0 ? b : a;
  if (up*boson <= 0 || down*boson >= 0) return false;
  return ((std::abs(up)==2 || std::abs(up)==4) && (std::abs(down)==1 || std::abs(down)==3 || std::abs(down)==5)) ||
         ((std::abs(down)==11 || std::abs(down)==13 || std::abs(down)==15) && std::abs(up)==std::abs(down)+1);
}

template<class C, class V, class I, class G>
void readHiggs(Inventory& inv, int parent, int hpdg,
               const C& children, const V& ce, const V& cp,
               const I& ids, const G& eta, const G& phi) {
  if (hpdg == 0) return;
  if (hpdg != 25 || children.empty() || ce.size()!=children.size() || cp.size()!=children.size()) {
    openAll(inv,parent);
    return;
  }
  if (children.size()<2) openAll(inv,parent);
  auto add = [&](int pdg, double e, double p) {
    if (!std::isfinite(e) || !std::isfinite(p)) inv.open[kind(pdg)] |= parent;
    else inv.daughters.push_back({parent,kind(pdg),e,p});
  };
  for (size_t j=0;j<children.size();++j) {
    const int child=children[j], a=std::abs(child);
    if (kind(child) || parton(child)) { add(child,ce[j],cp[j]); continue; }
    if (a==12 || a==14 || a==16 || a==22) continue;
    if (a!=24 && a!=23 && a!=15) { openAll(inv,parent); continue; }
    const bool available=j<ids.size() && j<eta.size() && j<phi.size() &&
                         ids[j].size()==eta[j].size() && ids[j].size()==phi[j].size();
    if (a==15) {
      // A direct H -> tau can have a stored charged granddaughter. No third generation is inferred.
      bool completeLeptonic=false;
      if (available && ids[j].size()==3) {
        for (size_t i=0;i<3;++i) {
          const int p=ids[j][i], f=std::abs(p);
          if ((f!=11 && f!=13) || p*child<=0) continue;
          const int sign=child>0 ? 1 : -1;
          if (std::find(ids[j].begin(),ids[j].end(),16*sign)!=ids[j].end() &&
              std::find(ids[j].begin(),ids[j].end(),-(f+1)*sign)!=ids[j].end()) {
            add(p,eta[j][i],phi[j][i]);
            completeLeptonic=true;
          }
        }
      }
      if (!completeLeptonic) openTau(inv,parent,child);
      continue;
    }
    if (!available || ids[j].size()!=2 || !bosonPair(child,ids[j][0],ids[j][1])) {
      openAll(inv,parent);
      continue;
    }
    for (size_t i=0;i<2;++i) {
      const int p=ids[j][i];
      if (kind(p) || parton(p)) add(p,eta[j][i],phi[j][i]);
      else if (std::abs(p)==15) openTau(inv,parent,p);
    }
  }
}

inline std::vector<Object> objects(const h3::Leptons& lep, const h3::Jets& jets,
                                   const RVec<float>& pt,const RVec<float>& eta,
                                   const RVec<float>& phi,const RVec<char>& selected) {
  std::vector<Object> out;
  for (int i=0;i<3;++i)
    out.push_back({i,kind(lep[i].pdg),lep[i].p4.Eta(),lep[i].p4.Phi(),h3::prompt(lep[i])});
  auto add = [&](size_t j,int slot) { out.push_back({slot,0,eta[j],phi[j],true}); };
  for (size_t k=0;k<jets.centralIndex.size();++k)
    add(jets.centralIndex[k],k<6 ? int(3+k) : -1);
  std::vector<size_t> forward;
  for (size_t j=0;j<selected.size();++j)
    if (selected[j] && std::abs(eta[j])>=2.5) forward.push_back(j);
  std::stable_sort(forward.begin(),forward.end(),[&](size_t a,size_t b){return pt[a]>pt[b];});
  for (size_t k=0;k<forward.size();++k) add(forward[k],k<2 ? int(10+k) : -1);
  return out;
}

// Keep consensus over maximum-cardinality one-to-one daughter assignments. Jets
// beyond the stored slots participate to avoid a spurious unique match.
inline Result match(const Inventory& inv,const std::vector<Object>& reco,int higgs) {
  const auto& d=inv.daughters;
  std::vector<std::vector<int>> candidates(d.size());
  for (size_t i=0;i<d.size();++i)
    for (size_t j=0;j<reco.size();++j)
      if (d[i].kind==reco[j].kind &&
          synrelations::deltaR(d[i].eta,d[i].phi,reco[j].eta,reco[j].phi)<(d[i].kind ? 0.1 : 0.2))
        candidates[i].push_back(j);
  std::vector<int> owner(reco.size(),-1);
  std::vector<bool> assigned(d.size(),false);
  int best=-1, visits=0;
  bool overflow=false;
  Slots unionOptions{}, possibleSame{}, possibleDifferent{};
  auto record = [&](int matched) {
    if (matched<best) return;
    if (matched>best) {
      best=matched; unionOptions.fill(0); possibleSame.fill(0); possibleDifferent.fill(0);
    }
    auto loose=inv.open;
    for (size_t i=0;i<d.size();++i) if (!assigned[i]) loose[d[i].kind] |= d[i].parent;
    Slots options{};
    for (size_t j=0;j<reco.size();++j) {
      const auto& o=reco[j];
      if (o.slot<0) continue;
      int m=owner[j]>=0 ? d[owner[j]].parent : (none | loose[o.kind]);
      if (owner[j]<0 && !o.prompt) m |= higgs;
      options[o.slot]=m;
      unionOptions[o.slot] |= m;
    }
    for (int i=0;i<12;++i) for (int j=i+1;j<12;++j) {
      const int a=options[i], b=options[j];
      if (!a || !b) continue;
      if (a & b & both) possibleSame[i] |= 1<<j;
      if (!(a==b && (a==2 || a==4))) possibleDifferent[i] |= 1<<j;
    }
  };
  auto search = [&](auto&& self,size_t i,int n) -> void {
    if (overflow || n+int(d.size()-i)<best) return;
    if (++visits>8192) { overflow=true; return; }
    if (i==d.size()) { record(n); return; }
    for (int j:candidates[i]) if (owner[j]<0) {
      owner[j]=int(i); assigned[i]=true;
      self(self,i+1,n+1);
      owner[j]=-1; assigned[i]=false;
    }
    self(self,i+1,n);
  };
  search(search,0,0);
  Result result;
  if (overflow) { result.conservative=1; return result; }
  for (int i=0;i<12;++i) {
    const int m=unionOptions[i];
    if (!m) continue;
    if (m==none) result.ids[i]=result.ancestry[i]=0;
    if (!(m & none)) {
      result.ancestry[i]=1;
      if (m==2 || m==4) result.ids[i]=m==2 ? 1 : 2;
    }
    result.known[i]=possibleSame[i]^possibleDifferent[i];
    result.same[i]=possibleSame[i]&result.known[i];
  }
  return result;
}

template<class C, class V, class I, class G>
Result labels(const h3::Leptons& lep,const h3::Jets& jets,const RVec<float>& pt,
              const RVec<float>& eta,const RVec<float>& phi,const RVec<char>& selected,
              int h1,const C& c1,const V& ce1,const V& cp1,const I& g1,const G& ge1,const G& gp1,
              int h2,const C& c2,const V& ce2,const V& cp2,const I& g2,const G& ge2,const G& gp2) {
  if ((h1!=0 && h1!=25) || (h2!=0 && h2!=25)) return Result{};
  Inventory inv;
  readHiggs(inv,2,h1,c1,ce1,cp1,g1,ge1,gp1);
  readHiggs(inv,4,h2,c2,ce2,cp2,g2,ge2,gp2);
  return match(inv,objects(lep,jets,pt,eta,phi,selected),(h1==25 ? 2:0) | (h2==25 ? 4:0));
}
} // namespace higgsdecay
