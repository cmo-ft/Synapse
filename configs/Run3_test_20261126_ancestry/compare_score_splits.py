"""Compare single-H and VVV on the validation and test samples of one trained fold."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from working_point_composition import METHODS, SPLITS, hh_cuts, read_split, wait_for_product


CATEGORIES = ("singleH", "VVV")
NAMES = {"singleH": "Single H", "VVV": "VVV"}
STYLE = {"Synapse": "#20639B", "EveNet": "#B94F19", "XGBoost": "#21815F"}
EFF = np.arange(10, 61, dtype=float)/100
I20 = 10


def load(run, fold, split):
    a, codes, names, regions, scores = read_split(run, fold, split)
    keep = (a["weight"] > 0) & np.isin(codes, [-1, *[names.index(k) for k in CATEGORIES]])
    return dict(weight=a["weight"][keep].astype("f8"), dsid=a["dsid"][keep], event=a["eventNumber"][keep],
                hh=codes[keep] == -1, bkg={k: codes[keep] == names.index(k) for k in CATEGORIES},
                regions={m: regions[m][keep] for m in METHODS}, scores={m: scores[m][keep] for m in METHODS})


def cluster_index(d, mask):
    key = np.rec.fromarrays((d["dsid"][mask], d["event"][mask]), names="dsid,event")
    return np.unique(key, return_inverse=True)[1].ravel()


def signal_bootstrap(d, rng, n):
    hh = d["hh"]
    w = d["weight"][hh]
    ix = cluster_index(d, hh)
    orders, scores = {}, {}
    for m in METHODS:
        scores[m] = d["scores"][m][hh]
        use = np.flatnonzero(d["regions"][m][hh])
        orders[m] = use[np.argsort(-scores[m][use], kind="stable")]
    cuts = {m: np.empty(n+1) for m in METHODS}
    for k in range(n+1):
        wk = w if k == 0 else w*rng.poisson(1., ix.max()+1)[ix]
        target = .2*wk.sum()
        for m, order in orders.items():
            j = np.searchsorted(wk[order].cumsum(), target)
            if j >= len(order):
                raise ValueError(f"Cannot reach 20% HH efficiency in replica {k}")
            cuts[m][k] = scores[m][order[j]]
    return cuts


def rejection_bootstrap(d, cat, cuts, val_cuts, rng):
    mask = d["bkg"][cat]
    w = d["weight"][mask]
    ix = cluster_index(d, mask)
    cluster_weight = np.bincount(ix, weights=w)
    scores = {m: np.where(d["regions"][m][mask], d["scores"][m][mask], -np.inf) for m in METHODS}
    tail = np.logical_or.reduce([scores[m] >= min(cuts[m].min(), val_cuts[m].min()) for m in METHODS])
    scores = {m: s[tail] for m,s in scores.items()}
    n = len(cuts[METHODS[0]])
    result = {mode: {m: np.empty(n) for m in METHODS} for mode in ("own", "val_cut")}
    tail_weight, tail_ix = w[tail], ix[tail]
    for k in range(n):
        if k == 0:
            total, tw = w.sum(), tail_weight
        else:
            multiplicity = rng.poisson(1., len(cluster_weight))
            total, tw = cluster_weight @ multiplicity, tail_weight*multiplicity[tail_ix]
        for mode, thresholds in (("own", cuts), ("val_cut", val_cuts)):
            for m, score in scores.items():
                kept = tw[score >= thresholds[m][k]].sum()
                result[mode][m][k] = total/kept if kept > 0 else np.nan
    return result


def summary(x):
    x = np.asarray(x)
    samples = x[1:][np.isfinite(x[1:])]
    if len(samples) < .99*(len(x)-1):
        raise ValueError("Insufficient finite bootstrap replicas")
    q = np.quantile(samples, [.025,.16,.84,.975])
    return dict(nominal=float(x[0]), low95=q[0], low68=q[1], high68=q[2], high95=q[3])


def tail_rows(d, split, cat, cuts, rows, by_dsid):
    bg = d["bkg"][cat]
    w = d["weight"]
    ix = cluster_index(d, bg)
    for mode, c in cuts.items():
        for m in METHODS:
            passed = d["regions"][m] & (d["scores"][m] >= c[m])
            b = bg & passed
            positive = w[b]
            cl = np.bincount(ix[passed[bg]], weights=positive)
            cl = np.sort(cl[cl > 0])[::-1]
            rows.append(dict(split=split, background=cat, method=m, thresholds=mode, cut=float(c[m]),
                             hh_eff=w[passed & d["hh"]].sum()/w[d["hh"]].sum(), preselection=w[bg].sum(),
                             passing_weight=positive.sum(), passing_rows=int(b.sum()), passing_event_keys=len(cl),
                             effective_passing_keys=positive.sum()**2 / np.sum(cl**2),
                             largest_key_fraction=cl[0]/positive.sum(), top3_key_fraction=cl[:3].sum()/positive.sum()))
            ids, id_indices = np.unique(d["dsid"][bg], return_inverse=True)
            den = np.bincount(id_indices, weights=w[bg])
            selected = passed[bg]
            sums = np.bincount(id_indices[selected], weights=positive, minlength=len(ids))
            counts = np.bincount(id_indices[selected], minlength=len(ids))
            for dsid, sw, kept, n in zip(ids, den, sums, counts):
                by_dsid.append(dict(split=split, background=cat, method=m, thresholds=mode, dsid=int(dsid),
                                    denominator=sw, passing=kept, passing_fraction=kept/positive.sum(), passing_rows=int(n)))


def reference_checks(run, fold, split, values):
    path = run / "output" / f"fold_{fold}/evaluation" / split / "background_rejection/working_points.csv"
    with path.open() as stream:
        reference = {(r["category"], r["method"]): r for r in csv.DictReader(stream)
                     if r["normalization"] == "from_preselection" and r["category"] in CATEGORIES
                     and float(r["target_signal_eff"]) == .2 and r["method"] in METHODS}
    for cat in CATEGORIES:
        for m in METHODS:
            ref = reference[cat,m]
            np.testing.assert_allclose(values[cat]["own"][m][0],float(ref["rejection"]),rtol=1e-9)


def scan(d, cuts):
    curves = {}
    for name, mask in d["bkg"].items():
        denom = d["weight"][mask].sum()
        curves[name] = {}
        for m in METHODS:
            rows = np.flatnonzero(mask & d["regions"][m])
            s = d["scores"][m][rows]
            order = np.argsort(-s, kind="stable")
            tail_sum = np.r_[0., np.cumsum(d["weight"][rows][order])]
            j = np.searchsorted(-s[order], -cuts[m], side="right")
            curves[name][m] = denom / tail_sum[j]
    return curves


def rejection_figure(curves, ratio, fold, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.7,7.4), sharex=True,
                             gridspec_kw={"height_ratios":[2.2,1.]})
    for column,cat in enumerate(CATEGORIES):
        top, bottom = axes[:,column]
        for m in METHODS:
            val, test = (curves[split][cat][m] for split in SPLITS)
            top.plot(EFF, val, color=STYLE[m], lw=2, linestyle="--")
            top.plot(EFF, test, color=STYLE[m], lw=2)
            bottom.plot(EFF, test/val, color=STYLE[m], lw=1.8)
            ci = ratio[cat][m]
            # Offset intervals slightly in x so three intervals at 20% remain visible.
            x = .2 + (METHODS.index(m)-1)*.004
            bottom.vlines(x,ci["low95"],ci["high95"],color=STYLE[m],lw=1)
            bottom.vlines(x,ci["low68"],ci["high68"],color=STYLE[m],lw=3)
            bottom.scatter(x,ci["nominal"],s=22,color=STYLE[m],zorder=4)
        for ax in (top,bottom):
            ax.grid(alpha=.18)
            ax.axvline(.2,color=".65",lw=1,ls=":",zorder=0)
            ax.spines[["right","top"]].set_visible(False)
        top.set_yscale("log")
        top.set_title(NAMES[cat],loc="left",fontsize=13)
        bottom.axhline(1,color=".35",ls="--",lw=1)
        if cat=="singleH":
            bottom.set_yscale("log")
            ticks = [.25,.5,1,2,4,8,16]
            lo,hi = bottom.get_ylim()
            bottom.set_yticks([v for v in ticks if lo<=v<=hi],labels=[str(v) for v in ticks if lo<=v<=hi])
        bottom.xaxis.set_major_formatter(PercentFormatter(1))
        bottom.set_xlabel("HH efficiency from jet preselection")
    axes[0,0].set_ylabel("Background rejection")
    axes[1,0].set_ylabel("Test / validation")
    handles=[Line2D([],[],color=STYLE[m],lw=2,label=m) for m in METHODS]
    handles += [Line2D([],[],color=".2",lw=2,ls=style,label=split.capitalize()) for split,style in zip(SPLITS,("--","-"))]
    fig.suptitle(f"Fold {fold}  ·  Background rejection on validation and test",x=.075,ha="left",fontsize=16)
    fig.legend(handles=handles,ncol=5,frameon=False,loc="upper left",bbox_to_anchor=(.065,.939))
    fig.subplots_adjust(left=.085,right=.982,top=.83,bottom=.14,wspace=.2,hspace=.14)
    fig.text(.085,.027,"Each split sets its own HH threshold. At 20%, thick and thin bars give 68% and 95% MC bootstrap intervals for the ratio.",fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def shapes_figure(data, val_cuts, fold, path):
    fig, axes=plt.subplots(2,3,figsize=(12.6,7.2),sharex=True)
    # Uniform in log odds to resolve the high-score tail; endpoints include under/overflow.
    edges=np.linspace(-8,8,49)
    centers=(edges[:-1]+edges[1:])/2
    def logodds(s):
        s=np.clip(np.asarray(s,dtype="f8"),1e-12,1-1e-12)
        return np.clip(np.log(s)-np.log1p(-s),edges[0],edges[-1])
    for col,m in enumerate(METHODS):
        for row,cat in enumerate(CATEGORIES):
            ax=axes[row,col]
            for split,color,linestyle in (("validation","#266D9C","--"),("test","#C15923","-")):
                d=data[split]
                bg=d["bkg"][cat]
                mask=bg & d["regions"][m]
                sw=d["weight"][mask]/d["weight"][bg].sum()
                score=logodds(d["scores"][m][mask])
                h=np.histogram(score,bins=edges,weights=sw)[0]
                error=np.sqrt(np.histogram(score,bins=edges,weights=sw**2)[0])
                ax.stairs(h,edges,color=color,lw=1.6,ls=linestyle)
                upper=h+error
                lower=np.maximum(h-error,1e-7)
                ax.fill_between(centers,np.where(h>0,lower,np.nan),np.where(h>0,upper,np.nan),
                                step="mid",color=color,alpha=.13,linewidth=0)
            ax.axvline(float(logodds(val_cuts[m])),color=".28",lw=1,ls=":")
            ax.set_yscale("log")
            ax.set_ylim(1e-5,1)
            ax.set_xlim(edges[0],edges[-1])
            ax.grid(axis="y",alpha=.15)
            ax.spines[["right","top"]].set_visible(False)
            if row==0: ax.set_title(m,fontsize=12)
            if col==0: ax.set_ylabel(NAMES[cat]+"\nFraction of preselection / bin")
            if row==1:
                labels=np.array([.01,.1,.5,.9,.99,.999])
                ax.set_xticks(logodds(labels),[".01",".1",".5",".9",".99",".999"])
                ax.set_xlabel("HH score")
    handles=[Line2D([],[],color=c,ls=l,lw=2,label=n) for c,l,n in
             (("#266D9C","--","Validation"),("#C15923","-","Test"),(".28",":","Validation 20% HH cut"))]
    fig.suptitle(f"Fold {fold}  ·  Background HH-score distributions",x=.07,ha="left",fontsize=16)
    fig.legend(handles=handles,ncol=3,frameon=False,loc="upper left",bbox_to_anchor=(.06,.938))
    fig.subplots_adjust(left=.095,right=.985,top=.82,bottom=.14,wspace=.20,hspace=.20)
    fig.text(.095,.027,"Positive-weight MC in each classifier’s signal region. The common jet-preselection denominator includes vetoed events. Shading shows per-bin √Σw².",fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def single_h_sources(rows, tail, fold, path):
    labels = {604305: "W⁻H inclusive", 604306: "W⁺H inclusive", 604304: "qqZH inclusive",
              602637: "ttH dileptonic", 602638: "ttH semileptonic", 603327: "ggH → μμ"}
    colors = ("#4477AA", "#74ADD1", "#228833", "#AA3377", "#CC6677", "#CCBB44", "#9298A4")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5.7))
    max_total = 0.
    for ax,split in zip(axes,SPLITS):
        bottom = np.zeros(len(METHODS))
        for (dsid,label),color in zip([*labels.items(), (None, "Other single H")], colors):
            values = []
            for method in METHODS:
                pool = [r for r in rows if r["background"]=="singleH" and r["thresholds"]=="own"
                        and r["split"]==split and r["method"]==method]
                values.append(sum(float(r["passing"]) for r in pool if
                                  (int(r["dsid"])==dsid if dsid is not None else int(r["dsid"]) not in labels)))
            ax.bar(METHODS,values,bottom=bottom,color=color,label=label,width=.62,edgecolor="white",linewidth=.4)
            bottom += values
        max_total = max(max_total, float(bottom.max()))
        for i,method in enumerate(METHODS):
            row = next(r for r in tail if r["background"]=="singleH" and r["thresholds"]=="own"
                       and r["split"]==split and r["method"]==method)
            ax.annotate(f"N_eff {float(row['effective_passing_keys']):.1f}",(i,bottom[i]),
                        textcoords="offset points",xytext=(0,5),ha="center",fontsize=9)
        ax.set_title(split.capitalize(),loc="left")
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylim(0,max_total*1.21)
    axes[0].set_ylabel("Passing single-H yield · positive nominal weights")
    fig.suptitle(f"Fold {fold}  ·  Single-H events passing the 20% HH working point",x=.07,ha="left",fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,frameon=False,loc="upper left",bbox_to_anchor=(.06,.92))
    fig.subplots_adjust(left=.09,right=.98,top=.74,bottom=.15,wspace=.12)
    fig.text(.09,.037,"N_eff = (Σw)² / Σw², grouping repeated (DSID, eventNumber) before squaring. Working points are calibrated on each split.",fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main(run, fold, replicas, seed):
    data={split:load(run,fold,split) for split in SPLITS}
    random=iter(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6))
    fitted, curves, boot, tail, dsids={}, {}, {}, [], []
    for split,d in data.items():
        w, hh = d["weight"],d["hh"]
        grid=hh_cuts(w,hh,d["regions"],d["scores"],EFF)
        curves[split]=scan(d,grid)
        fitted[split]=signal_bootstrap(d,next(random),replicas)
        for m in METHODS:
            np.testing.assert_equal(grid[m][I20], fitted[split][m][0])
        cuts=fitted[split]
        val=fitted["validation"]
        boot[split]={}
        for cat in CATEGORIES:
            boot[split][cat]=rejection_bootstrap(d,cat,cuts,val,next(random))
            tail_rows(d,split,cat,{"own":{m:cuts[m][0] for m in METHODS},
                                    "val_cut":{m:val[m][0] for m in METHODS}},tail,dsids)
        reference_checks(run,fold,split,boot[split])
    intervals=[]
    ratios={}
    for cat in CATEGORIES:
        ratios[cat]={}
        for mode in ("own","val_cut"):
            for m in METHODS:
                v,t = (boot[split][cat][mode][m] for split in SPLITS)
                for what,x in (("validation",v),("test",t),("test_over_validation",t/v)):
                    intervals.append(dict(background=cat,thresholds=mode,method=m,quantity=what,**summary(x)))
                if mode=="own":
                    ratios[cat][m]=summary(t/v)
            for alt in ("EveNet","XGBoost"):
                v,t=(boot[split][cat][mode]["Synapse"]/boot[split][cat][mode][alt] for split in SPLITS)
                for what,x in (("validation",v),("test",t),("test_over_validation",t/v)):
                    intervals.append(dict(background=cat,thresholds=mode,method="Synapse / "+alt,quantity=what,**summary(x)))
    dest=run/"output"/f"fold_{fold}/evaluation/validation_test"
    dest.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest) as tmp:
        out=Path(tmp)
        rejection_figure(curves,ratios,fold,out/"rejection_validation_test.pdf")
        shapes_figure(data,{m:fitted["validation"][m][0] for m in METHODS},fold,out/"score_shapes_validation_test.pdf")
        single_h_sources(dsids,tail,fold,out/"single_h_passing_sources.pdf")
        write_csv(out/"r20_mc_intervals.csv",intervals)
        write_csv(out/"r20_passing_events.csv",tail)
        write_csv(out/"r20_by_dsid.csv",dsids)
        (out/"settings.json").write_text(json.dumps(dict(fold=fold,bootstrap_replicas=replicas,seed=seed,
            split_remainders=[(fold+1)%3,fold],weight="positive nominal MC weights",
            sampling="Poisson(1) per (DSID,eventNumber), shared across classifiers. The two samples are resampled independently. HH threshold and background denominator are recomputed per replica. Validation thresholds are shared across samples in val_cut.",
            xgboost_region="EveNet SR"),indent=2)+"\n")
        for p in out.iterdir():
            p.replace(dest/p.name)
    print(dest,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1,2),required=True)
    parser.add_argument("--replicas",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=27112026)
    parser.add_argument("--wait-for-benchmark",action="store_true")
    args=parser.parse_args()
    if args.wait_for_benchmark:
        folder = args.run / "output" / f"fold_{args.fold}"
        for split in SPLITS:
            wait_for_product(folder / "evaluation" / split / "background_rejection/working_points.csv", folder)
    if args.replicas < 100: parser.error("Use at least 100 replicas")
    main(args.run.resolve(),args.fold,args.replicas,args.seed)
