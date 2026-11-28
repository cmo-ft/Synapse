"""Freeze the HHML TransformTool bin edges from validation histograms using system PyROOT."""
import json
import os
import sys
from pathlib import Path


def transformtool_source():
    override = os.environ.get("HHML_TRANSFORMTOOL_DIR")
    candidates = [Path(override)] if override else [
        Path(__file__).resolve().parents[2] / "TransformTool",
        Path("/afs/cern.ch/user/c/cmo/private/workspace/bbtautau/WSMaker_HH_bbtautau-unblind/WSMakerCore/TransformTool"),
    ]
    for source in candidates:
        if all((source/file).is_file() for file in ("Root/HistoTransform.cxx", "TransformTool/HistoTransform.h")):
            return source
    raise FileNotFoundError("Set HHML_TRANSFORMTOOL_DIR to the checkout containing Root/HistoTransform.cxx and TransformTool/HistoTransform.h")


if __name__ == "__main__":
    import ROOT

    ROOT.gROOT.SetBatch(True)
    source = transformtool_source()
    ROOT.gInterpreter.AddIncludePath(str(source))
    if not ROOT.gInterpreter.Declare(f'#include "{source}/Root/HistoTransform.cxx"'):
        raise RuntimeError("Failed to load HistoTransform")

    transform = ROOT.HistoTransform()
    transform.trafoSixY = 10
    transform.trafoSixZ = 5
    transform.trafoSixtyMCLowBound = 3
    transform.trafoSixMCstatUpBound = 0.2
    transform.trafoSixtyIncludeS = False

    result = {}
    with ROOT.TFile.Open(sys.argv[1]) as f:
        for key in f.GetListOfKeys():
            name = key.GetName()
            bkg, sig = f.Get(f"{name}/background"), f.Get(f"{name}/signal")
            bins = transform.getRebinBins(bkg, sig, 60)
            last = bkg.GetNbinsX() + 1
            edges = sorted({bkg.GetBinLowEdge(max(1, min(i, last))) for i in bins})
            if len(edges) < 2 or abs(edges[0]) > 1e-10 or abs(edges[-1] - 1) > 1e-10:
                raise ValueError(f"Invalid validation binning for {name}: {edges}")
            edges[0], edges[-1] = 0.0, 1.0
            result[name] = edges
    Path(sys.argv[2]).write_text(json.dumps(result))
