"""Maintain the overnight fold 1 status and interpret the paired validation and test results."""
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def rows(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def job_state(job):
    states = {1: "queued", 2: "running", 3: "removing", 4: "completed", 5: "held", 6: "transferring output", 7: "suspended"}
    try:
        result = subprocess.run(["condor_q", job, "-json", "-attributes", "JobStatus,HoldReason,RemoteHost"],
                                capture_output=True, text=True, timeout=15, check=True)
        data = json.loads(result.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return "scheduler status unavailable"
    if not data:
        return "left the queue"
    job = data[0]
    return " ".join(str(x) for x in [states.get(job["JobStatus"], "unknown"), job.get("RemoteHost"), job.get("HoldReason")] if x)


def snapshot(run):
    folder = run / "output/fold_1"
    history = rows(run / "epochs.csv")
    level = "preparing the fold"
    if (run / "publication.exit").is_file() and (run / "publication.exit").read_text().strip() == "0":
        level = "dataset verified, waiting for worker"
    for name, label in (("workflow.log","fitting"), ("validation/validation_scores.root","applying and evaluating"),
                        ("evaluation/hhml_eval/summary.json","evaluating backgrounds and auxiliary tasks")):
        if (folder / name).is_file():
            level = label
    for file, success in (("exit_status.txt","comparing with the reference"), ("pipeline.exit","training and reference evaluation complete")):
        if (folder / file).is_file():
            code = (folder / file).read_text().strip()
            level = success if code == "0" else f"workflow exited with code {code} ({file})"
    if not history:
        return f"{level}. No complete validation epoch."
    last = history[-1]
    best = max(history, key=lambda x: float(x["z"]))
    return (f"{level}. Epoch {last['epoch']}/50, validation loss {float(last['loss']):.5f}, "
            f"AUC {float(last['auc']):.5f}, R20 {float(last['r20']):.1f}, three-bin Z {float(last['z']):.5f}. "
            f"Highest validation Z so far {float(best['z']):.5f} at epoch {best['epoch']}.")


def comparison(run, reference, title):
    directory = run / "output/fold_1/evaluation" / ("comparison_" + reference.name)
    if not (directory / "comparison.md").is_file():
        return []
    results = rows(directory / "results.csv")
    if not results:
        return []
    data = {(r["split"],r["model"]): r for r in results if r["cuts"] == "matched"}
    changes = {(r["split"],r["category"]): r for r in rows(directory/"signed_yield_at_20.csv") if r["cuts"] == "matched"}
    if len(data) != 8 or not changes:
        return []
    out = ["", "## " + title, "",
           "Rejection uses the matched 20% HH working point in each split. Higher R20 and Z are better.", "",
           "| Split | Model | Inclusive R20 | WZ R20 | VVV R20 | Single-H R20 | Three-bin Z | Trafo60 Z |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation","test"):
        for name in ("New Synapse","Previous Synapse","EveNet","XGBoost"):
            r = data[split,name]
            out.append(f"| {split} | {name} | {float(r['r20']):.1f} | {float(r['r20_WZ']):.1f} | "
                       f"{float(r['r20_VVV']):.1f} | {float(r['r20_singleH']):.1f} | {float(r['z_20_30_50']):.5f} | {float(r['trafo60']):.5f} |")
    out += ["", "Change in expected background yield for the new NN relative to the preceding NN. "
                "A negative change means fewer expected events pass. Uncertainties are paired MC statistical errors at fixed thresholds.", ""]
    for group in ("WZ","VVV","singleH","All backgrounds"):
        fields = [changes[split,group] for split in ("validation","test")]
        delta = [float(x["delta"]) for x in fields]
        errors = [float(x["paired_mc_error"]) for x in fields]
        detail = ("lower in both splits" if max(delta)<0 else "higher in both splits" if min(delta)>0 else "different directions across splits")
        if all(abs(d)<=s for d,s in zip(delta,errors)):
            detail += ", within one MC standard error in both"
        out.append(f"- {group} changed by {delta[0]:+.4f} ± {errors[0]:.4f} on validation and "
                   f"{delta[1]:+.4f} ± {errors[1]:.4f} on test ({detail}).")
    out += ["", f"Full results are in `{directory}`."]
    return out


def main(primary, extra, previous, job):
    target = primary / "overnight_status.md"
    tick = 0
    scheduler = "awaiting scheduler update"
    archived = None
    while True:
        completed = extra / "output/fold_1/pipeline.exit"
        if completed.is_file():
            scheduler = "workflow ended with exit code " + completed.read_text().strip()
        elif tick % 5 == 0:
            scheduler = job_state(job)
        text = ["# Overnight fold 1 status", "",
                f"- Parent-conditioned model on the terminal — {snapshot(primary)}",
                f"- Visible HH candidates on HTCondor — {snapshot(extra)}",
                f"- GPU job {job} — {scheduler}.",
                ""]
        for run, ref, label in ((primary,previous,"Parent-conditioned model against the preceding NN"),
                                (extra,previous,"Visible HH candidates against the preceding NN"),
                                (extra,primary,"Effect of adding the visible HH candidates")):
            text.extend(comparison(run,ref,label))
        body = "\n".join(text).rstrip()+"\n"
        if body != archived:
            tmp = target.with_suffix(".tmp")
            tmp.write_text(body+"\nUpdated "+datetime.now().astimezone().isoformat(timespec="seconds")+"\n")
            tmp.replace(target)
            archived = body
        if (primary/"comparisons.exit").is_file():
            break
        tick += 1
        time.sleep(60)


if __name__ == "__main__":
    main(*(Path(x).resolve() for x in sys.argv[1:4]), sys.argv[4])
