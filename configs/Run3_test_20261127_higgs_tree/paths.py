"""Inputs and destinations for the Run 3 Higgs decay-tree labels."""
from pathlib import Path

RECIPE = Path(__file__).resolve().parent
if RECIPE.parent.name == "frozen":
    RUN = RECIPE.parents[1]
    REPO = RUN.parents[1]
else:
    REPO = RECIPE.parents[1]
    RUN = REPO / "run_dir/Run3_test_20261127_higgs_tree"
BASELINE = RUN.parent / "Run3_test_20261126_ancestry"
PHYSICS = RUN.parent / "Run3_test_20261124_physics"
RELATIONS = RUN.parent / "Run3_test_20261125_relations"
PRODUCTION = REPO.parents[1] / "caf/hhml3l0tau/work/nominal-run3-20261122"
PLAN = PRODUCTION / "plan/plan.json"
