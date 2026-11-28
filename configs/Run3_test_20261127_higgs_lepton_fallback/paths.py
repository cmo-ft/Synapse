"""Reference datasets and destination for the Run 3 lepton mask fallback."""
from pathlib import Path

RECIPE = Path(__file__).resolve().parent
if RECIPE.parent.name == "frozen":
    RUN = RECIPE.parents[1]
    REPO = RUN.parents[1]
else:
    REPO = RECIPE.parents[1]
    RUN = REPO / "run_dir/Run3_test_20261127_higgs_lepton_fallback"

TREE_RUN = RUN.parent / "Run3_test_20261127_higgs_tree"
ANCESTRY_RUN = RUN.parent / "Run3_test_20261126_ancestry"
MATRIX_RUN = RUN.parent / "Run3_test_20261125_relations"
PHYSICS_RUN = RUN.parent / "Run3_test_20261124_physics"
PLAN = REPO.parents[1] / "caf/hhml3l0tau/work/nominal-run3-20261122/plan/plan.json"
