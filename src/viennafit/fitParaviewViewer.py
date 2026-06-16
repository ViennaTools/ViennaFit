"""Module for opening VTP files in ParaView with annotations."""

import fnmatch
import glob
import os
import subprocess
import tempfile


def openInParaview(
    folder,
    patterns=None,
    translations=None,
    labels=True,
    labelAnchor="y",
    mode="2D",
    paraviewExecutable="paraview",
):
    """Open .vtp files in a folder in ParaView with filename annotations.

    Launches the ParaView GUI with a script that loads .vtp files found in
    the given folder. Each source is renamed to its filename stem in the Pipeline
    Browser, and each file gets a movable text annotation in the render view.

    Parameters
    ----------
    folder : str
        Path to the folder containing .vtp files.
    patterns : str or list of str, optional
        Glob pattern(s) to filter which files to open. If None, all .vtp files
        are opened. Examples: ``"*standard.vtp"``,
        ``["*standard.vtp", "*wider.vtp"]``.
    translations : dict, optional
        Dictionary mapping glob patterns to ``(x, y, z)`` translation offsets.
        Files matching a pattern are translated by the given amount using a
        ParaView Transform filter. Example::

            translations={
                "*standard.vtp": (0, 0, 0),
                "*wider.vtp": (100, 0, 0),
            }

    labels : bool, optional
        Whether to show a movable text annotation for each file. Defaults to True.
    labelAnchor : {'x', 'y', 'z', None}, optional
        Axis whose minimum is used as the world-space anchor for each file's text
        annotation. Defaults to ``'y'`` (bottom edge of geometry). Pass ``None``
        to use the file's translation offset instead.
    mode : {'2D', '3D'}, optional
        Interaction mode for the ParaView render view. Defaults to ``'2D'``.
    paraviewExecutable : str, optional
        Path to the ParaView executable. Defaults to "paraview".
    """
    folder = os.path.abspath(folder)

    if patterns is None:
        patterns = ["*.vtp"]
    elif isinstance(patterns, str):
        patterns = [patterns]

    vtpFiles = set()
    for pattern in patterns:
        vtpFiles.update(glob.glob(os.path.join(folder, pattern)))
    vtpFiles = sorted(vtpFiles)

    if not vtpFiles:
        print(f"No files matching {patterns} found in {folder}")
        return

    # Build a mapping from filepath to translation offset
    fileTranslations = {}
    if translations:
        for filepath in vtpFiles:
            baseName = os.path.basename(filepath)
            for pattern, offset in translations.items():
                if fnmatch.fnmatch(baseName, pattern):
                    fileTranslations[filepath] = tuple(offset)
                    break

    # Build lists for the ParaView script
    filepathsRepr = repr(vtpFiles)
    stems = [os.path.splitext(os.path.basename(f))[0] for f in vtpFiles]
    labelsRepr = repr(stems)
    translationsRepr = repr(fileTranslations)
    showLabels = labels

    useBoundsAnchor = labelAnchor in ("x", "y", "z")
    if useBoundsAnchor:
        axisMap = {"x": (0, 0), "y": (2, 1), "z": (4, 2)}
        boundsMinIdx, worldIdx = axisMap[labelAnchor]
    else:
        boundsMinIdx, worldIdx = 2, 1  # unused placeholders

    scriptContent = f"""\
from paraview.simple import *

files = {filepathsRepr}
labels = {labelsRepr}
translations = {translationsRepr}

view = GetActiveViewOrCreate('RenderView')
view.InteractionMode = '{mode}'

text_displays = []

for filepath, label in zip(files, labels):
    source = OpenDataFile(filepath)
    RenameSource(label, source)

    if filepath in translations:
        transform = Transform(Input=source)
        transform.Transform.Translate = list(translations[filepath])
        RenameSource(label + " (translated)", transform)
        Show(transform, view)
        Hide(source, view)
    else:
        Show(source, view)

    # Per-file movable text annotation
    if {showLabels}:
        text = Text(Text=label)
        textDisplay = Show(text, view)
        textDisplay.FontFamily = 'Arial'
        textDisplay.FontSize = 20
        textDisplay.Color = [0, 0, 0]
        textDisplay.WindowLocation = 'Any Location'
        if {useBoundsAnchor}:
            bounds = source.GetDataInformation().GetBounds()
            translation = translations.get(filepath, (0, 0, 0))
            world_pos = [
                (bounds[0] + bounds[1]) / 2 + translation[0],
                (bounds[2] + bounds[3]) / 2 + translation[1],
                (bounds[4] + bounds[5]) / 2 + translation[2],
            ]
            world_pos[{worldIdx}] = bounds[{boundsMinIdx}] + translation[{worldIdx}]
        else:
            world_pos = translations.get(filepath, (0, 0, 0))
        text_displays.append((textDisplay, world_pos))
        RenameSource(label + " (label)", text)

ResetCamera()
Render()

# Place each annotation at the screen position of its file's world-space origin
renderer = view.GetRenderer()
size = view.ViewSize
for textDisplay, world_pos in text_displays:
    renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
    renderer.WorldToDisplay()
    d = renderer.GetDisplayPoint()
    textDisplay.Position = [d[0] / size[0], d[1] / size[1]]

Render()
"""

    scriptFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_vtp_", delete=False
    )
    scriptFile.write(scriptContent)
    scriptFile.close()

    subprocess.Popen([paraviewExecutable, f"--script={scriptFile.name}"])
    print(f"Launched ParaView with {len(vtpFiles)} .vtp files from {folder}")


def _findProjectDir(startDir, maxLevels=6):
    """Walk up from startDir until finding the project root (contains domains/)."""
    d = os.path.abspath(startDir)
    for _ in range(maxLevels):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "domains")):
            return d
    raise FileNotFoundError(
        f"Could not find a project root with a 'domains/' subdirectory above {startDir}"
    )


def openBestInParaview(
    optimizationRunDir,
    labels=True,
    domainSpacing=200.0,
    paraviewExecutable="paraview",
):
    """Open the current best optimization result alongside target surfaces in ParaView.

    Reads ``progressBest.csv`` (or falls back to ``progressAll.csv``) to find
    the best evaluation number, then opens the corresponding VTP files from the
    ``progress/`` folder alongside the target surfaces from the project's
    ``domains/targetDomain/`` folder.

    For multi-domain runs each domain pair (target + simulated) is placed side
    by side along the X axis with ``domainSpacing`` between columns. Domains are
    sorted in natural order (W1 < W2 < … < W13). Target surfaces are shown in
    green with line width 3; simulated surfaces use default styling.

    Parameters
    ----------
    optimizationRunDir : str
        Path to the optimization run directory (contains ``progressBest.csv``
        and the ``progress/`` subfolder).
    labels : bool, optional
        Whether to show a domain-name annotation for each column. Defaults to True.
    domainSpacing : float, optional
        X distance between consecutive domain columns (same units as the geometry,
        typically nm). Defaults to 200.0.
    paraviewExecutable : str, optional
        Path to the ParaView executable. Defaults to ``"paraview"``.
    """
    import csv
    import re

    optimizationRunDir = os.path.abspath(optimizationRunDir)
    runName = os.path.basename(optimizationRunDir)

    # Locate best evaluation number
    bestEval = None
    bestCsv = os.path.join(optimizationRunDir, "progressBest.csv")
    if os.path.exists(bestCsv):
        with open(bestCsv, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            bestEval = int(rows[-1]["evaluationNumber"])

    if bestEval is None:
        allCsv = os.path.join(optimizationRunDir, "progressAll.csv")
        if os.path.exists(allCsv):
            with open(allCsv, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                bestRow = min(rows, key=lambda r: float(r["objectiveValue"]))
                bestEval = int(bestRow["evaluationNumber"])

    if bestEval is None:
        raise FileNotFoundError(
            f"No progressBest.csv or progressAll.csv found in {optimizationRunDir}"
        )

    # Glob best VTPs from progress/
    progressDir = os.path.join(optimizationRunDir, "progress")
    bestVtps = sorted(
        glob.glob(os.path.join(progressDir, f"{runName}-{bestEval:03d}-*.vtp"))
    )
    multiDomain = bool(bestVtps)
    if not multiDomain:
        bestVtps = sorted(
            glob.glob(os.path.join(progressDir, f"{runName}-{bestEval:03d}.vtp"))
        )

    # Glob target surfaces
    projectDir = _findProjectDir(optimizationRunDir)
    targetVtps = sorted(
        glob.glob(os.path.join(projectDir, "domains", "targetDomain", "*-surface.vtp"))
    )

    # If this is a fold run, show only calibration (train) domain targets
    foldInfoPath = os.path.join(
        os.path.dirname(os.path.dirname(optimizationRunDir)), "fold-info.json"
    )
    if os.path.exists(foldInfoPath):
        import json

        with open(foldInfoPath) as f:
            foldInfo = json.load(f)
        trainDomains = foldInfo.get("trainDomains", [])
        targetVtps = [
            v
            for v in targetVtps
            if any(f"-{d}-" in os.path.basename(v) for d in trainDomains)
        ]

    if not bestVtps and not targetVtps:
        print(f"No VTP files found for evaluation {bestEval:03d} in {progressDir}")
        return

    def _naturalKey(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

    if multiDomain:
        # Extract domain name: target -> *-targetDomain-{domain}-surface.vtp
        def _targetDomain(path):
            m = re.search(r"-targetDomain-(.+)-surface\.vtp$", path)
            return m.group(1) if m else None

        # Extract domain name: best -> {runName}-{eval:03d}-{domain}.vtp
        bestPrefix = f"{runName}-{bestEval:03d}-"

        def _bestDomain(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            return stem[len(bestPrefix) :] if stem.startswith(bestPrefix) else None

        targetByDomain = {_targetDomain(f): f for f in targetVtps if _targetDomain(f)}
        bestByDomain = {_bestDomain(f): f for f in bestVtps if _bestDomain(f)}

        allDomains = sorted(set(targetByDomain) | set(bestByDomain), key=_naturalKey)
        domainOffsets = {d: i * domainSpacing for i, d in enumerate(allDomains)}

        # Ordered (filepath, pipeline_label, x_offset) entries
        targetEntries = [
            (targetByDomain[d], f"{d} (target)", domainOffsets[d])
            for d in allDomains
            if d in targetByDomain
        ]
        bestEntries = [
            (bestByDomain[d], f"{d} (sim)", domainOffsets[d])
            for d in allDomains
            if d in bestByDomain
        ]
        # One text annotation per domain column, placed using target bounds
        labelEntries = [
            (targetByDomain.get(d) or bestByDomain.get(d), d, domainOffsets[d])
            for d in allDomains
        ]
    else:
        targetEntries = [
            (f, os.path.splitext(os.path.basename(f))[0], 0.0) for f in targetVtps
        ]
        bestEntries = [
            (f, os.path.splitext(os.path.basename(f))[0], 0.0) for f in bestVtps
        ]
        labelEntries = targetEntries or bestEntries

    showLabels = labels

    scriptContent = f"""\
from paraview.simple import *

target_entries = {repr(targetEntries)}
best_entries = {repr(bestEntries)}
label_entries = {repr(labelEntries)}

view = GetActiveViewOrCreate('RenderView')
view.InteractionMode = '2D'
view.AxesGrid.Visibility = 1

loaded_sources = {{}}  # filepath -> source, for label bounds

def _load_and_show(filepath, pipeline_name, x_offset):
    source = OpenDataFile(filepath)
    loaded_sources[filepath] = source
    RenameSource(pipeline_name, source)
    if x_offset:
        transform = Transform(Input=source)
        transform.Transform.Translate = [x_offset, 0.0, 0.0]
        RenameSource(pipeline_name + " (t)", transform)
        display = Show(transform, view)
        Hide(source, view)
    else:
        display = Show(source, view)
    return display

for filepath, label, x_offset in target_entries:
    display = _load_and_show(filepath, label, x_offset)
    display.AmbientColor = [0.0, 1.0, 0.0]
    display.DiffuseColor = [0.0, 1.0, 0.0]
    display.LineWidth = 3.0

for filepath, label, x_offset in best_entries:
    display = _load_and_show(filepath, label, x_offset)
    display.LineWidth = 3.0

import numpy as _np
from paraview import servermanager as _sm
from vtk.util.numpy_support import vtk_to_numpy as _vtk_to_numpy

def _label_pos(source, x_offset, band_fraction=0.05):
    dataset = _sm.Fetch(source)
    pts = _vtk_to_numpy(dataset.GetPoints().GetData())  # (N, 3)
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    band = (y_max - y_min) * band_fraction
    near_bottom = pts[pts[:, 1] <= y_min + band]
    x_right = near_bottom[:, 0].max()
    z_mid = (pts[:, 2].min() + pts[:, 2].max()) / 2.0
    return [x_right + x_offset, y_min, z_mid]

text_displays = []
if {showLabels}:
    for filepath, domain_label, x_offset in label_entries:
        source = loaded_sources.get(filepath)
        if source is None:
            continue
        text = Text(Text=domain_label)
        textDisplay = Show(text, view)
        textDisplay.FontFamily = 'Arial'
        textDisplay.FontSize = 20
        textDisplay.Color = [0, 0, 0]
        textDisplay.WindowLocation = 'Any Location'
        text_displays.append((textDisplay, _label_pos(source, x_offset)))
        RenameSource(domain_label + " (label)", text)

ResetCamera()
Render()

renderer = view.GetRenderer()
size = view.ViewSize
for textDisplay, world_pos in text_displays:
    renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
    renderer.WorldToDisplay()
    d = renderer.GetDisplayPoint()
    textDisplay.Position = [d[0] / size[0], d[1] / size[1]]

Render()
"""

    scriptFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_best_", delete=False
    )
    scriptFile.write(scriptContent)
    scriptFile.close()

    subprocess.Popen([paraviewExecutable, f"--script={scriptFile.name}"])
    nTarget = len(targetVtps)
    nBest = len(bestVtps)
    print(
        f"Launched ParaView: {nTarget} target surface(s) + {nBest} best surface(s)"
        f" (evaluation {bestEval:03d}) from {optimizationRunDir}"
    )


def openCustomEvaluationInParaview(
    customEvaluationDir,
    labels=True,
    paraviewExecutable="paraview",
):
    """Open custom evaluation results alongside target surfaces in ParaView.

    Loads all result VTP files from a custom evaluation directory and the
    corresponding target surfaces from the project's ``domains/targetDomain/``
    folder. Domains whose name contains ``wider`` are shifted 200 units in X
    so all domain variants are visible side-by-side.

    Target surfaces are shown in green with line width 3; simulated surfaces
    use default styling.

    Parameters
    ----------
    customEvaluationDir : str
        Path to the custom evaluation output directory (contains ``*-result-*.vtp``).
    labels : bool, optional
        Whether to show a movable text annotation for each file. Defaults to True.
    paraviewExecutable : str, optional
        Path to the ParaView executable. Defaults to ``"paraview"``.
    """
    customEvaluationDir = os.path.abspath(customEvaluationDir)

    resultVtps = sorted(glob.glob(os.path.join(customEvaluationDir, "*-result-*.vtp")))
    if not resultVtps:
        print(f"No result VTP files found in {customEvaluationDir}")
        return

    # Project dir is two levels up: customEvaluations/<evalName> -> project root
    projectDir = os.path.dirname(os.path.dirname(customEvaluationDir))
    targetVtps = sorted(
        glob.glob(os.path.join(projectDir, "domains", "targetDomain", "*-surface.vtp"))
    )

    resultStems = [os.path.splitext(os.path.basename(f))[0] for f in resultVtps]
    targetStems = [os.path.splitext(os.path.basename(f))[0] for f in targetVtps]
    showLabels = labels

    scriptContent = f"""\
from paraview.simple import *
import os as _os

target_files = {repr(targetVtps)}
target_labels = {repr(targetStems)}
result_files = {repr(resultVtps)}
result_labels = {repr(resultStems)}

view = GetActiveViewOrCreate('RenderView')
view.InteractionMode = '2D'
view.AxesGrid.Visibility = 1

text_displays = []

def _x_offset(filepath):
    return 200.0 if 'wider' in _os.path.basename(filepath) else 0.0

for filepath, label in zip(target_files, target_labels):
    source = OpenDataFile(filepath)
    RenameSource(label, source)
    x_off = _x_offset(filepath)
    if x_off:
        transform = Transform(Input=source)
        transform.Transform.Translate = [x_off, 0.0, 0.0]
        RenameSource(label + " (translated)", transform)
        display = Show(transform, view)
        Hide(source, view)
    else:
        display = Show(source, view)
    display.AmbientColor = [0.0, 1.0, 0.0]
    display.DiffuseColor = [0.0, 1.0, 0.0]
    display.LineWidth = 3.0
    if {showLabels}:
        text = Text(Text=label)
        textDisplay = Show(text, view)
        textDisplay.FontFamily = 'Arial'
        textDisplay.FontSize = 20
        textDisplay.Color = [0, 0, 0]
        textDisplay.WindowLocation = 'Any Location'
        bounds = source.GetDataInformation().GetBounds()
        text_displays.append((textDisplay, [(bounds[0] + bounds[1]) / 2 + x_off, bounds[2], (bounds[4] + bounds[5]) / 2]))
        RenameSource(label + " (label)", text)

for filepath, label in zip(result_files, result_labels):
    source = OpenDataFile(filepath)
    RenameSource(label, source)
    x_off = _x_offset(filepath)
    if x_off:
        transform = Transform(Input=source)
        transform.Transform.Translate = [x_off, 0.0, 0.0]
        RenameSource(label + " (translated)", transform)
        display = Show(transform, view)
        Hide(source, view)
    else:
        display = Show(source, view)
    display.LineWidth = 3.0
    if {showLabels}:
        text = Text(Text=label)
        textDisplay = Show(text, view)
        textDisplay.FontFamily = 'Arial'
        textDisplay.FontSize = 20
        textDisplay.Color = [0, 0, 0]
        textDisplay.WindowLocation = 'Any Location'
        bounds = source.GetDataInformation().GetBounds()
        text_displays.append((textDisplay, [(bounds[0] + bounds[1]) / 2 + x_off, bounds[2], (bounds[4] + bounds[5]) / 2]))
        RenameSource(label + " (label)", text)

ResetCamera()
Render()

renderer = view.GetRenderer()
size = view.ViewSize
for textDisplay, world_pos in text_displays:
    renderer.SetWorldPoint(world_pos[0], world_pos[1], world_pos[2], 1.0)
    renderer.WorldToDisplay()
    d = renderer.GetDisplayPoint()
    textDisplay.Position = [d[0] / size[0], d[1] / size[1]]

Render()
"""

    scriptFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_custom_eval_", delete=False
    )
    scriptFile.write(scriptContent)
    scriptFile.close()

    subprocess.Popen([paraviewExecutable, f"--script={scriptFile.name}"])
    print(
        f"Launched ParaView: {len(targetVtps)} target surface(s) + "
        f"{len(resultVtps)} result surface(s) from {customEvaluationDir}"
    )


def _viewBestCLI():
    import argparse

    parser = argparse.ArgumentParser(prog="viennafit-view-best")
    parser.add_argument(
        "optimizationRunDir", help="Path to the optimization run directory"
    )
    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        default=True,
        help="Hide domain annotations (shown by default)",
    )
    parser.add_argument(
        "--domain-spacing",
        dest="domainSpacing",
        type=float,
        default=500.0,
        help="X distance between domain columns in geometry units (default: 500.0)",
    )
    args = parser.parse_args()
    openBestInParaview(
        args.optimizationRunDir,
        labels=args.labels,
        domainSpacing=args.domainSpacing,
    )


def _viewCustomEvaluationCLI():
    import argparse

    parser = argparse.ArgumentParser(prog="viennafit-view-custom-eval")
    parser.add_argument(
        "customEvaluationDir", help="Path to the custom evaluation directory"
    )
    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        default=True,
        help="Hide filename annotations (shown by default)",
    )
    args = parser.parse_args()
    openCustomEvaluationInParaview(args.customEvaluationDir, labels=args.labels)
