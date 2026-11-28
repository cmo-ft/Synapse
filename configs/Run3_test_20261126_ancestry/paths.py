"""Locations of the published Run 3 MC inputs and the new run's products."""
from pathlib import Path

RECIPE = Path(__file__).resolve().parent
REPO = RECIPE.parents[3] if RECIPE.parent.name == "frozen" else RECIPE.parents[1]
REFERENCE = REPO / "run_dir/Run3_test_20261125_relations"
PHYSICS = REFERENCE.parent / "Run3_test_20261124_physics"
PRODUCTION = REPO.parents[1] / "caf/hhml3l0tau/work/nominal-run3-20261122"
RUN = REFERENCE.parent / "Run3_test_20261126_ancestry"
PLAN = PRODUCTION / "plan/plan.json"
PARENTS = ("Higgs", "Top", "W", "Z")
