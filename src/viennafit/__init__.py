"""ViennaFit package for semiconductor process parameter optimization."""

from importlib import import_module

__version__ = "2.0.0"

__all__ = [
    "Project",
    "Study",
    "Optimization",
    "LocalSensitivityStudy",
    "GlobalSensitivityStudy",
    "CustomEvaluator",
    "readPointsFromFile",
    "plotParameterProgression",
    "plotParameterPositions",
    "openInParaview",
    "annotate",
    "EarlyStoppingException",
    "implant",
]


def __getattr__(name):
    """Lazy top-level exports so pure-Python subpackages stay importable."""
    if name == "Project":
        from .fitProject import Project
        return Project
    if name == "Study":
        from .fitStudy import Study
        return Study
    if name == "Optimization":
        from .fitOptimization import Optimization
        return Optimization
    if name == "LocalSensitivityStudy":
        from .fitLocalSensitivityStudy import LocalSensitivityStudy
        return LocalSensitivityStudy
    if name == "GlobalSensitivityStudy":
        from .fitGlobalSensitivityStudy import GlobalSensitivityStudy
        return GlobalSensitivityStudy
    if name == "CustomEvaluator":
        from .fitCustomEvaluator import CustomEvaluator
        return CustomEvaluator
    if name in {
        "readPointsFromFile",
        "plotParameterProgression",
        "plotParameterPositions",
    }:
        from . import fitUtilities
        return getattr(fitUtilities, name)
    if name == "openInParaview":
        from .fitParaviewViewer import openInParaview
        return openInParaview
    if name == "annotate":
        from .fitAnnotator import annotate
        return annotate
    if name == "EarlyStoppingException":
        from .fitExceptions import EarlyStoppingException
        return EarlyStoppingException
    if name == "implant":
        return import_module(".implant", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
