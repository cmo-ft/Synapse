# Fold 1 training statistics

The reference is the 36-event-input `Run3_test_20261127_parent_present` model. All fits use the same 2,297,211 validation events (eventNumber % 3 = 2), 50 validation checks, batch size 768, seed 42 and the reference physics objective.

| Worker argument | Training entries including WZCR | Approximate epochs over selected events | Optimizer updates |
|---|---:|---:|---:|
| `quarter` | 703,623 | 200 | 183,300 |
| `half` | 1,406,031 | 100 | 183,300 |
| Original, completed reference | 2,815,350 | 50 | 183,300 |
| `double` | 5,630,088 | 25 | 183,300 |
| `double_50epochs` | 5,630,088 | 50 | 366,550 |

`selection.py` assigns each (DSID, eventNumber) to a stable group before training. The quarter is wholly contained in the half. Training refits normalization and class factors on the selected rows. The `double` and `double_50epochs` samples add the original test fold, with its WZCR sample. Their performance comparison uses validation only.

Run `selection.py` with the parent_present fold files available. It writes masks and the selection manifest under `run_dir/Run3_test_20261128_training_statistics/selections/`. From the repository root, `condor_submit configs/Run3_test_20261128_training_statistics/train.sub` submits the three fixed-update jobs. The companion job uses `condor_submit DoubleFullEpochs=1 configs/Run3_test_20261128_training_statistics/train.sub`. The submit file specifies the CERN spool and GPU resources.

Each worker scores the common validation sample. `compare.py` checks event identities and weights and measures the rejection curve across HH efficiencies using the checkpoint with maximum three-bin significance. The primary working point is 20%. Each sample has `comparison/rejection_by_background.pdf`, `working_points.csv` and `validation.json`. The two doubled-sample schedules are compared directly under `double_50epochs/comparison_to_double/`.
