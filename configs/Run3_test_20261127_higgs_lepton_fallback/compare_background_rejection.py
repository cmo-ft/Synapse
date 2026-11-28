"""Add the previous Synapse to the published current Synapse, EveNet and XGBoost rejection plot."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import uproot

from paths import ANCESTRY_RUN, REPO, RUN

sys.path.insert(0, str(REPO/"hhml_eval"))
sys.path.insert(0, str(RUN/"frozen/plots"))
import plot_background_rejection as benchmark

METHODS = ("Current", "EveNet", "XGBoost", "Previous")
LABELS = {"Previous":"Previous NN","Current":"Current NN","EveNet":"EveNet","XGBoost":"XGBoost"}
STYLES = {"Current":benchmark.STYLES["Synapse"], "EveNet":benchmark.STYLES["EveNet"],
          "XGBoost":benchmark.STYLES["XGBoost"], "Previous":("#843b92",":")}
GROUPS = ("All backgrounds",*benchmark.GROUPS.values())
DISPLAY = dict(benchmark.TITLES, **{"All backgrounds":"Inclusive"})
DEST = RUN/"rejection_comparison"


def epoch(run,fold):
    files=list((run/f"output/fold_{fold}/fit/checkpoints").glob("BEST-*.ckpt"))
    if len(files)!=1:
        raise ValueError(f"Expected one selected checkpoint for fold {fold} in {run}")
    return 1+int(re.search(r"epoch=epoch=(\d+)",files[0].name).group(1))


def previous_curves(fold, split, expected):
    """Use the published previous NN curve, including its original ordering of tied scores."""
    source = ANCESTRY_RUN/f"output/fold_{fold}/evaluation/{split}/background_rejection/roc_curves.csv"
    curves = {key: [] for key in expected if key[2] == "Previous"}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "Synapse":
                key = (row["normalization"], row["category"], "Previous")
                curves[key].append((float(row["signal_eff"]), float(row["background_eff"])))
    if not all(curves.values()):
        raise ValueError(f"Previous NN curves are incomplete in {source}")
    return {key: np.asarray(points).T for key, points in curves.items()}


def supply_points(fold,split):
    folder=f"output/fold_{fold}/{split}"
    source=folder+f"/{split}_scores.root"
    current,groups=benchmark.load_scores(RUN/source, RUN/"dataset/summary.json")
    identity=("eventNumber","dsid","weight","class_id","n_jet","evenet_valid","evenet_pS","evenet_pConv","xgb_score")
    with uproot.open(ANCESTRY_RUN/source,handler=uproot.source.file.MemmapSource,array_cache=None) as old:
        previous=old["tree"].arrays([*identity,"p_signal","p_extconv_bkg"],library="np")
    if any(not np.array_equal(previous[k],current[k],equal_nan=True) for k in identity):
        raise ValueError("Old and current scores do not contain the same events, weights and EveNet reference")
    rem=fold if split=="test" else (fold+1)%3
    if not np.all(current["eventNumber"]%3==rem):
        raise ValueError("Wrong sample remainder")
    group,weight,regions,scores,_=benchmark.roc_inputs(current,groups)
    use=(current["n_jet"]>=1)&(current["weight"]>0)
    regions={"Previous":previous["p_extconv_bkg"][use]<.8, "Current":regions["Synapse"],
             "EveNet":regions["EveNet"], "XGBoost":regions["XGBoost"]}
    scores={"Previous":previous["p_signal"][use],"Current":scores["Synapse"],
            "EveNet":scores["EveNet"],"XGBoost":scores["XGBoost"]}
    categories,curves,points=benchmark.measure(group,weight,regions,scores,reference="Current")
    if tuple(categories)!=GROUPS:
        raise ValueError(f"Unexpected categories: {categories}")
    curves.update(previous_curves(fold, split, curves))
    return categories,curves,points,dict(scored_rows=len(current["weight"]),positive_weight_jet_preselection=len(weight),remainder=rem)


def validate(fold,split,points):
    by_key={(x["normalization"],x["category"],x["method"],x["target_signal_eff"]):x for x in points}
    tested=0
    for run,alias in ((ANCESTRY_RUN,"Previous"),(RUN,"Current")):
        path=run/f"output/fold_{fold}/evaluation/{split}/background_rejection/working_points.csv"
        with path.open(newline="") as stream:
            for old in csv.DictReader(stream):
                if old["method"] not in ("Synapse","EveNet","XGBoost"):
                    continue
                method=alias if old["method"]=="Synapse" else old["method"]
                key=old["normalization"],old["category"],method,float(old["target_signal_eff"])
                for name in ("rejection","rejection_mc_stat","score_min","signal_eff"):
                    got=by_key[key][name]
                    if not np.isclose(float(old[name]),got,atol=1e-12,rtol=1e-10,equal_nan=True):
                        raise ValueError(f"Mismatch to published {run.name} fold {fold} {split} {key}, {name}")
                if run == RUN:
                    for new_key,old_key in (("rejection_ratio_to_current","rejection_ratio_to_synapse"),
                                            ("ratio_mc_stat","ratio_mc_stat")):
                        if not np.isclose(by_key[key][new_key],float(old[old_key]),atol=1e-12,rtol=1e-10,equal_nan=True):
                            raise ValueError(f"Original ratio or paired uncertainty changed in {key}")
                tested+=1
    return tested


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def format_rejection(value):
    return f"{value:,.0f}" if np.isfinite(value) else "0 pass"


def summary_pages(all_points,epochs,stage):
    by_key={(x["split"],x["fold"],x["category"],x["method"]):x for x in all_points
            if x["normalization"]=="from_preselection" and x["target_signal_eff"]==.20}
    compact=[]
    plain=["Background rejection at 20% inclusive HH efficiency. Larger values give better rejection.",
           f"Fold 0  previous epoch {epochs[0][0]}, current epoch {epochs[0][1]}.",
           f"Fold 1  previous epoch {epochs[1][0]}, current epoch {epochs[1][1]}.",""]
    with PdfPages(stage/"rejection_at_20pct.pdf") as pdf:
        for split in ("test","validation"):
            matrix=np.array([[by_key[split,fold,cat,method]["rejection"] for fold in (0,1) for method in METHODS]
                             for cat in GROUPS])
            plain.extend((split.capitalize(), "                                      Fold 0                                    Fold 1",
                          f"{'Background':17s}"+" ".join(f"{m:>10s}" for m in METHODS*2)))
            for cat,values in zip(GROUPS,matrix):
                row=dict(split=split,background=cat)
                for f in (0,1):
                    for j,m in enumerate(METHODS):
                        point=by_key[split,f,cat,m]
                        row[f"fold_{f}_{m.lower()}"]=values[len(METHODS)*f+j]
                        row[f"fold_{f}_{m.lower()}_mc_stat"]=point["rejection_mc_stat"]
                compact.append(row)
                plain.append(f"{DISPLAY.get(cat,cat):17s}"+" ".join(f"{format_rejection(v):>10s}" for v in values))
            plain.append("")
            fig=plt.figure(figsize=(14.1,6.8))
            ax=fig.add_axes([.065,.14,.91,.66])
            ax.set_xlim(0,10.2)
            ax.set_ylim(-.15,9.4)
            ax.axis("off")
            xpos=(1.92,2.92,3.92,4.95,6.53,7.53,8.53,9.56)
            for f,center in enumerate((3.43,8.05)):
                ax.text(center,8.92,f"Fold {f}",ha="center",fontsize=15,weight="bold")
                ax.text(center,8.42,f"Previous epoch {epochs[f][0]} · Current epoch {epochs[f][1]}",ha="center",fontsize=9,color=".3")
            ax.text(.08,7.85,"Background",ha="left",va="center",weight="bold",fontsize=11)
            for x,method in zip(xpos,METHODS*2):
                ax.text(x,7.85,method,ha="center",va="center",color=STYLES[method][0],weight="bold",fontsize=11)
            for i,(category,values) in enumerate(zip(GROUPS,matrix)):
                y=7.12-i*.92
                if not i%2:
                    ax.axhspan(y-.42,y+.42,color="#eff3f6",linewidth=0)
                ax.text(.08,y,DISPLAY.get(category,category),ha="left",va="center",fontsize=12)
                for x,value in zip(xpos,values):
                    ax.text(x,y,format_rejection(value),ha="center",va="center",fontsize=12)
            ax.axvline(5.72,ymin=.06,ymax=.91,color=".78",lw=.8)
            fig.text(.07,.91,f"Background rejection at 20% HH efficiency  ·  {split.capitalize()}",fontsize=21,weight="semibold")
            fig.text(.07,.855,"Matched signal efficiency in each sample. Larger rejection means fewer background events pass.",fontsize=11)
            fig.text(.07,.095,"HH efficiency and rejection use positive nominal MC weights and the common ≥1-jet sample before each model’s conversion veto.",fontsize=9.5)
            fig.text(.07,.062,"0 pass means zero positive-weight background events passed in this finite sample. Per-process uncertainties appear in the CSV.",fontsize=9.5)
            pdf.savefig(fig)
            plt.close(fig)
    plain.extend(("Positive nominal weights and a common jet preselection. Both NNs use their own conversion veto; XGBoost uses the EveNet signal region.",
                  "0 pass denotes no passing positive-weight background. See rejection_at_20pct.csv for MC uncertainties.",""))
    (stage/"rejection_at_20pct.txt").write_text("\n".join(plain))
    write_csv(stage/"rejection_at_20pct.csv",compact)


def main():
    uproot.open.defaults["handler"]=uproot.source.file.MemmapSource
    DEST.mkdir(parents=True,exist_ok=True)
    epochs={f:(epoch(ANCESTRY_RUN,f),epoch(RUN,f)) for f in (0,1)}
    all_points,checks=[],{}
    with TemporaryDirectory(dir=DEST,prefix=".staging-") as tmp:
        stage=Path(tmp)
        pdfs=[]
        for split in ("test","validation"):
            for fold in (0,1):
                folder=stage/f"fold_{fold}_{split}"
                folder.mkdir()
                cats,curves,points,meta=supply_points(fold,split)
                meta["matched_existing_working_points"]=validate(fold,split,points)
                checks[f"fold_{fold}_{split}"]=meta
                label=f"Training fold {fold} · {split} · Previous epoch {epochs[fold][0]} · Current epoch {epochs[fold][1]}"
                with plt.rc_context({"legend.handlelength": 1.6, "legend.handletextpad": 0.6}):
                    benchmark.plot(cats, curves, points, label, folder, styles=STYLES,
                                   labels=LABELS, reference="Current")
                all_points.extend(dict(fold=fold,split=split,**p) for p in points)
                pdfs.append(folder/"rejection_by_background.pdf")
                print(label,meta,flush=True)
        summary_pages(all_points,epochs,stage)
        write_csv(stage/"all_working_points.csv",all_points)
        (stage/"comparison_checks.json").write_text(json.dumps(dict(previous=ANCESTRY_RUN.name,current=RUN.name,
            methods=list(METHODS), ratio_reference="Current NN", xgboost_signal_region="EveNet",
            previous_curve_source="26 November published Synapse ROC points",
            selected_epochs=epochs,checks=checks),indent=2)+"\n")
        subprocess.run(["pdfunite",*[str(p) for p in pdfs],str(stage/"rejection_at_20pct.pdf"),
                        str(stage/"background_rejection_comparison.pdf")],check=True)
        for timed in stage.iterdir():
            if timed.is_dir():
                (DEST/timed.name).mkdir(exist_ok=True)
                for p in timed.iterdir():
                    p.replace(DEST/timed.name/p.name)
            else:
                timed.replace(DEST/timed.name)


if __name__=="__main__":
    main()
