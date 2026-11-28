"""Write the live fold 1 epoch history alongside the previous run on the same split."""
import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import time


METRICS = {"loss":"val_loss_epoch","auc":"val_AUC_epoch","r20":"val_Rejection20_epoch","z":"val_Significance20_30_50_epoch"}


def atomic(path,text):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def refresh(run, ref, fold):
    directory=run/"output"/f"fold_{fold}"
    rows=[]
    messages=[]
    for epoch in range(1,51):
        name=f"epoch_{epoch:03d}.json"
        path=directory/"fit/epochs"/name
        if not path.exists():
            break
        try:
            current=json.loads(path.read_text())
        except json.JSONDecodeError:
            break
        oldpath=ref/"output"/f"fold_{fold}"/"fit/epochs"/name
        old=json.loads(oldpath.read_text()) if oldpath.is_file() else {}
        row={"epoch":epoch,"completed_utc":current["completed"],"seconds":current["seconds"]}
        for key,field in METRICS.items():
            row[key]=current[field]
            row["reference_"+key]=old.get(field,"")
        rows.append(row)
        messages.append(f"{row['completed_utc']}  fold {fold} epoch {epoch:02d}/50  validation loss {row['loss']:.5f}  "
                        f"AUC {row['auc']:.5f}  R20 {row['r20']:.1f}  Z {row['z']:.6f}")
    if rows:
        stream=StringIO()
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
        atomic(run/"epochs.csv",stream.getvalue())
        atomic(run/"epoch_updates.log","\n".join(messages)+"\n")
    return len(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--fold",type=int,default=1)
    args=p.parse_args()
    directory=args.run/"output"/f"fold_{args.fold}"
    n=-1
    while True:
        if (directory/"pipeline.exit").is_file():
            refresh(args.run,args.reference,args.fold)
            break
        if n < 50 and (directory/"fit/epochs"/f"epoch_{max(1,n+1):03d}.json").is_file():
            n=refresh(args.run,args.reference,args.fold)
        time.sleep(20)
