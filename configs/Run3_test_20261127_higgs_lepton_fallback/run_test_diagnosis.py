"""Run the paired diagnostics once a fold has published its final test score tree."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from paths import RUN


def main(fold):
    out = RUN / f"output/fold_{fold}"
    score_file = out / "test/test_scores.root"
    worker_status = out / "exit_status.txt"
    while not score_file.is_file():
        if worker_status.is_file():
            raise RuntimeError(f"Fold {fold} stopped with status {worker_status.read_text().strip()} before scoring")
        time.sleep(45)
    recipe = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    for name in ("compare_event_scores", "background_migrations", "compare_higgs_supervision", "probe_higgs_feedback", "compare_wz_recoil"):
        print(f"Fold {fold} · {name}", flush=True)
        subprocess.run([sys.executable, "-B", "-u", str(recipe/(name+".py")), "--fold", str(fold)],
                       env=env, check=True)
    print(f"Fold {fold} completed all paired diagnostics", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=(0, 1), type=int, required=True)
    main(parser.parse_args().fold)
