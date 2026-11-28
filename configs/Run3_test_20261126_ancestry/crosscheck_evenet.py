"""Apply EveNet files 1 and 2 to identical reconstructed events in each fold-1 split."""
import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import uproot


FEATURES = ("eventNumber","dsid","pt","eta","phi","e","charge","pdg","valid","HT_jet","m_all","m_lll","n_jet")
MODELS = (2, 1)
SPLITS = ("validation", "test")


def build_inputs(a):
    # This is the three-lepton, massless-jet, zero-tau CAF model input convention.
    n=len(a["pt"])
    part=np.zeros((n,11,7),dtype="f4")
    mask=np.zeros((n,11),dtype=bool)
    slots=np.r_[3:9,10:12]
    jet_pt=a["pt"][:,slots]
    rank=np.argsort(-jet_pt,axis=1,kind="stable")
    order=np.take_along_axis(np.broadcast_to(slots,rank.shape),rank,axis=1)
    for k,field in enumerate(("e","pt","eta","phi")):
        part[:,:3,k]=a[field][:,:3]
        part[:,3:,k]=np.take_along_axis(a[field],order,axis=1)
    part[:,:,:2]*=np.float32(.001)
    mask[:,:3]=a["valid"][:,:3]
    mask[:,3:]=np.take_along_axis(a["valid"],order,axis=1)
    part[:,:3,5]=np.where(np.abs(a["pdg"][:,:3])==13,2,1)
    part[:,:3,6]=a["charge"][:,:3]
    part[~mask]=0
    scale=np.float32(.001)
    met=a["pt"][:,9]*scale
    lep=a["pt"][:,:3].sum(axis=1,dtype="f8")
    ht=(lep+a["HT_jet"]).astype("f4")*scale+met
    global_feats=np.column_stack((met,a["phi"][:,9],np.full(n,3),np.zeros(n),a["n_jet"],ht,
                                  lep.astype("f4")*scale,a["m_all"]*scale,a["m_lll"]*scale,np.zeros(n))).astype("f4")
    return {"particle_features":part,"particle_mask":mask,"global_features":global_feats}


def cuts(folder):
    with (folder/"evaluation/validation/background_rejection/working_points.csv").open() as f:
        rows=list(csv.DictReader(f))
    def lookup(method,eff):
        return float(next(r for r in rows if r["normalization"]=="from_preselection"
                          and r["category"]=="All backgrounds" and r["method"]==method
                          and float(r["target_signal_eff"])==eff)["score_min"])
    return lookup("Synapse",.30), lookup("EveNet",.20)


def collect(run, folder, split, rem, syn_cut, eve_cut):
    sys.path.insert(0,str(run/"frozen/plots"))
    from process_catalog import GROUPS,process_codes
    groups=list(GROUPS.values())
    names=("singleH","VVV")
    with uproot.open(folder/split/(split+"_scores.root")) as score_file:
        a=score_file["tree"].arrays(["eventNumber","dsid","weight","evenet_valid","evenet_pS","evenet_pConv",
                                    "p_signal","p_extconv_bkg","n_jet"],library="np")
    g=process_codes(a["dsid"],run/"dataset/summary.json")
    base=(a["weight"]>0)&(a["n_jet"]>=1)&a["evenet_valid"].astype(bool)
    use=base & (a["p_signal"]>=syn_cut) & (a["p_extconv_bkg"]<.8) & np.isin(g,[groups.index(n) for n in names])
    event=a["eventNumber"]
    if not np.all(event%3==rem):
        raise ValueError("Unexpected event-number remainder")
    evpass=(a["evenet_pS"]>=eve_cut)&(a["evenet_pConv"]<.8)
    coverage={}
    for name in names:
        passed=(g==groups.index(name))&base&evpass
        coverage[name]=a["weight"][passed&use].sum(dtype="f8")/a["weight"][passed].sum(dtype="f8")
    arrays=[]
    with uproot.open(run/"dataset"/f"3fold_{rem}.root") as root:
        tree=root["tree"]
        if tree.num_entries!=len(event):
            raise ValueError("The applied sample and the reconstructed input file differ in length")
        for start in range(0,len(event),24000):
            sl=slice(start,start+24000)
            pick=use[sl]
            if not pick.any(): continue
            chunk=tree.arrays(FEATURES,entry_start=start,entry_stop=start+24000,library="np")
            for key in ("eventNumber","dsid","n_jet"):
                np.testing.assert_array_equal(chunk[key],a[key][sl])
            arrays.append(build_inputs({k:v[pick] for k,v in chunk.items()}))
    feeds={k:np.concatenate([chunk[k] for chunk in arrays]) for k in arrays[0]}
    sample={k:a[k][use] for k in ("weight","dsid","eventNumber","evenet_pS","evenet_pConv")}
    sample["group"]=np.array(groups)[g[use]]
    return feeds,sample,coverage


def paired_fraction(weight,keys,passed,rng):
    ix=np.unique(keys,return_inverse=True)[1].ravel()
    sums=np.bincount(ix,weights=weight)
    passing={i:np.bincount(ix,weights=weight*p,minlength=len(sums)) for i,p in passed.items()}
    ratios=[]
    for _ in range(1500):
        multi=rng.poisson(1.,len(sums))
        den=passing[2]@multi
        if den>0: ratios.append((passing[1]@multi)/den)
    total=sums.sum()
    frac={i:v.sum()/total for i,v in passing.items()}
    return dict(events=len(weight),file2=frac[2],file1=frac[1],file1_over_file2=frac[1]/frac[2],
                ratio_mc_interval95=np.quantile(ratios,[.025,.975]).tolist())


def main(run,models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    uproot.open.defaults["handler"]=uproot.source.file.MemmapSource
    folder=run/"output/fold_1"
    syn_cut,eve_cut=cuts(folder)
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    sessions={i:ort.InferenceSession(str(models/f"evenet_lite_fold{i}.onnx"),sess_options=options,
                                     providers=["CPUExecutionProvider"]) for i in MODELS}
    result=dict(synapse_cohort_cut=syn_cut,evenet_probe_cut=eve_cut,
                cohort="All positive-weight, jet-preselected EveNet-valid single-H and VVV events in the same Synapse SR above the fixed validation 30% HH cut.",
                quantity="Conditional weighted fraction with EveNet HH score above its validation 20% cut and pConv < 0.8, using an identical threshold for both files.",
                splits={})
    for split,rem in zip(SPLITS,MODELS):
        feeds,a,coverage=collect(run,folder,split,rem,syn_cut,eve_cut)
        print(f"{split}: using {len(a['weight'])} events",flush=True)
        predictions={i:np.empty((len(a["weight"]),3),dtype="f4") for i in MODELS}
        # The published ONNX input signature fixes the batch dimension at one.
        for j in range(len(a["weight"])):
            row={k:v[j:j+1] for k,v in feeds.items()}
            for i,session in sessions.items():
                predictions[i][j]=session.run(["class_probabilities"],row)[0][0]
            if j==199:
                err=np.abs(predictions[rem][:200,0]-a["evenet_pS"][:200])
                print("First 200 native HH score residuals",np.quantile(err,[.5,.99,1]),flush=True)
                if np.median(err)>.0001 or np.quantile(err,.99)>.003:
                    raise ValueError("Reconstructed inputs do not reproduce the original EveNet scores")
        delta=np.abs(predictions[rem][:,[0,2]]-np.column_stack((a["evenet_pS"],a["evenet_pConv"])))
        passed={i:(q[:,0]>=eve_cut)&(q[:,2]<.8) for i,q in predictions.items()}
        recorded=(a["evenet_pS"]>=eve_cut)&(a["evenet_pConv"]<.8)
        differences=int(np.count_nonzero(passed[rem]!=recorded))
        if np.quantile(delta,.99)>.003 or differences>max(1,.002*len(recorded)):
            raise ValueError("Native inference closure failed")
        report=dict(nominal_file=rem,absolute_score_difference_quantiles=np.quantile(delta,[.5,.99,1]).tolist(),
                    wp_decision_disagreements=differences, samples={})
        for cat in ("singleH","VVV"):
            use=a["group"]==cat
            wt=a["weight"][use].astype("f8")
            keys=np.rec.fromarrays((a["dsid"][use],a["eventNumber"][use]),names="dsid,event")
            stats=paired_fraction(wt,keys,{i:p[use] for i,p in passed.items()},np.random.default_rng(271126+rem))
            stats["coverage_of_native_EveNet_passing_yield"]=coverage[cat]
            report["samples"][cat]=stats
        print(split,report,flush=True)
        result["splits"][split]=report
    out=folder/"evaluation/validation_test"
    out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        stage=Path(tmp)
        (stage/"evenet_same_event_check.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        fig,axes=plt.subplots(1,2,figsize=(9,4.6))
        for ax,cat in zip(axes,("singleH","VVV")):
            for split,col,off in (("validation","#266D9C",-.16),("test","#C15923",.16)):
                stats=result["splits"][split]["samples"][cat]
                ax.bar(np.array([0.,1.])+off,[stats["file2"],stats["file1"]],.30,color=col,label=split.capitalize())
            ax.set_xticks([0,1],["EveNet file 2","EveNet file 1"])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title("Single H" if cat=="singleH" else "VVV",loc="left")
            ax.spines[["right","top"]].set_visible(False)
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Fraction passing the EveNet cut")
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc="upper right",frameon=False,bbox_to_anchor=(.98,.88))
        fig.suptitle("Two EveNet files applied to identical events within each split",x=.07,ha="left",fontsize=15)
        fig.subplots_adjust(left=.09,right=.97,top=.72,bottom=.2,wspace=.23)
        fig.text(.09,.035,"Events pass the same Synapse 30% HH cut. Both EveNet files use the same validation 20% HH threshold.",fontsize=8)
        fig.savefig(stage/"evenet_same_event_check.pdf")
        plt.close(fig)
        for p in stage.iterdir(): p.replace(out/p.name)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--models",type=Path,required=True)
    args=p.parse_args()
    main(args.run.resolve(),args.models.resolve())
