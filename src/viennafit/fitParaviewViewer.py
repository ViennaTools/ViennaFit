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
    label_anchor="y",
    mode="2D",
    paraview_executable="paraview",
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
    label_anchor : {'x', 'y', 'z', None}, optional
        Axis whose minimum is used as the world-space anchor for each file's text
        annotation. Defaults to ``'y'`` (bottom edge of geometry). Pass ``None``
        to use the file's translation offset instead.
    mode : {'2D', '3D'}, optional
        Interaction mode for the ParaView render view. Defaults to ``'2D'``.
    paraview_executable : str, optional
        Path to the ParaView executable. Defaults to "paraview".
    """
    folder = os.path.abspath(folder)

    if patterns is None:
        patterns = ["*.vtp"]
    elif isinstance(patterns, str):
        patterns = [patterns]

    vtp_files = set()
    for pattern in patterns:
        vtp_files.update(glob.glob(os.path.join(folder, pattern)))
    vtp_files = sorted(vtp_files)

    if not vtp_files:
        print(f"No files matching {patterns} found in {folder}")
        return

    # Build a mapping from filepath to translation offset
    file_translations = {}
    if translations:
        for filepath in vtp_files:
            basename = os.path.basename(filepath)
            for pattern, offset in translations.items():
                if fnmatch.fnmatch(basename, pattern):
                    file_translations[filepath] = tuple(offset)
                    break

    # Build lists for the ParaView script
    filepaths_repr = repr(vtp_files)
    stems = [os.path.splitext(os.path.basename(f))[0] for f in vtp_files]
    labels_repr = repr(stems)
    translations_repr = repr(file_translations)
    show_labels = labels

    use_bounds_anchor = label_anchor in ("x", "y", "z")
    if use_bounds_anchor:
        _axis_map = {"x": (0, 0), "y": (2, 1), "z": (4, 2)}
        bounds_min_idx, world_idx = _axis_map[label_anchor]
    else:
        bounds_min_idx, world_idx = 2, 1  # unused placeholders

    script_content = f"""\
from paraview.simple import *

files = {filepaths_repr}
labels = {labels_repr}
translations = {translations_repr}

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
    if {show_labels}:
        text = Text(Text=label)
        textDisplay = Show(text, view)
        textDisplay.FontFamily = 'Arial'
        textDisplay.FontSize = 20
        textDisplay.Color = [0, 0, 0]
        textDisplay.WindowLocation = 'Any Location'
        if {use_bounds_anchor}:
            bounds = source.GetDataInformation().GetBounds()
            translation = translations.get(filepath, (0, 0, 0))
            world_pos = [
                (bounds[0] + bounds[1]) / 2 + translation[0],
                (bounds[2] + bounds[3]) / 2 + translation[1],
                (bounds[4] + bounds[5]) / 2 + translation[2],
            ]
            world_pos[{world_idx}] = bounds[{bounds_min_idx}] + translation[{world_idx}]
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

    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_vtp_", delete=False
    )
    script_file.write(script_content)
    script_file.close()

    subprocess.Popen([paraview_executable, f"--script={script_file.name}"])
    print(f"Launched ParaView with {len(vtp_files)} .vtp files from {folder}")


def _find_project_dir(start_dir, max_levels=6):
    """Walk up from start_dir until finding the project root (contains domains/)."""
    d = os.path.abspath(start_dir)
    for _ in range(max_levels):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "domains")):
            return d
    raise FileNotFoundError(
        f"Could not find a project root with a 'domains/' subdirectory above {start_dir}"
    )


def openBestInParaview(
    optimizationRunDir,
    labels=True,
    domainSpacing=200.0,
    paraview_executable="paraview",
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
    paraview_executable : str, optional
        Path to the ParaView executable. Defaults to ``"paraview"``.
    """
    import csv
    import re

    optimizationRunDir = os.path.abspath(optimizationRunDir)
    runName = os.path.basename(optimizationRunDir)

    # Locate best evaluation number
    best_eval = None
    best_csv = os.path.join(optimizationRunDir, "progressBest.csv")
    if os.path.exists(best_csv):
        with open(best_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best_eval = int(rows[-1]["evaluationNumber"])

    if best_eval is None:
        all_csv = os.path.join(optimizationRunDir, "progressAll.csv")
        if os.path.exists(all_csv):
            with open(all_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best_row = min(rows, key=lambda r: float(r["objectiveValue"]))
                best_eval = int(best_row["evaluationNumber"])

    if best_eval is None:
        raise FileNotFoundError(
            f"No progressBest.csv or progressAll.csv found in {optimizationRunDir}"
        )

    # Glob best VTPs from progress/
    progress_dir = os.path.join(optimizationRunDir, "progress")
    best_vtps = sorted(
        glob.glob(os.path.join(progress_dir, f"{runName}-{best_eval:03d}-*.vtp"))
    )
    multi_domain = bool(best_vtps)
    if not multi_domain:
        best_vtps = sorted(
            glob.glob(os.path.join(progress_dir, f"{runName}-{best_eval:03d}.vtp"))
        )

    # Glob target surfaces
    project_dir = _find_project_dir(optimizationRunDir)
    target_vtps = sorted(
        glob.glob(os.path.join(project_dir, "domains", "targetDomain", "*-surface.vtp"))
    )

    # If this is a fold run, show only calibration (train) domain targets
    fold_info_path = os.path.join(
        os.path.dirname(os.path.dirname(optimizationRunDir)), "fold-info.json"
    )
    if os.path.exists(fold_info_path):
        import json as _json

        with open(fold_info_path) as _f:
            _fold_info = _json.load(_f)
        train_domains = _fold_info.get("trainDomains", [])
        target_vtps = [
            v
            for v in target_vtps
            if any(f"-{d}-" in os.path.basename(v) for d in train_domains)
        ]

    if not best_vtps and not target_vtps:
        print(f"No VTP files found for evaluation {best_eval:03d} in {progress_dir}")
        return

    def _natural_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

    if multi_domain:
        # Extract domain name: target -> *-targetDomain-{domain}-surface.vtp
        def _target_domain(path):
            m = re.search(r"-targetDomain-(.+)-surface\.vtp$", path)
            return m.group(1) if m else None

        # Extract domain name: best -> {runName}-{eval:03d}-{domain}.vtp
        best_prefix = f"{runName}-{best_eval:03d}-"

        def _best_domain(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            return stem[len(best_prefix) :] if stem.startswith(best_prefix) else None

        target_by_domain = {
            _target_domain(f): f for f in target_vtps if _target_domain(f)
        }
        best_by_domain = {_best_domain(f): f for f in best_vtps if _best_domain(f)}

        all_domains = sorted(
            set(target_by_domain) | set(best_by_domain), key=_natural_key
        )
        domain_offsets = {d: i * domainSpacing for i, d in enumerate(all_domains)}

        # Ordered (filepath, pipeline_label, x_offset) entries
        target_entries = [
            (target_by_domain[d], f"{d} (target)", domain_offsets[d])
            for d in all_domains
            if d in target_by_domain
        ]
        best_entries = [
            (best_by_domain[d], f"{d} (sim)", domain_offsets[d])
            for d in all_domains
            if d in best_by_domain
        ]
        # One text annotation per domain column, placed using target bounds
        label_entries = [
            (target_by_domain.get(d) or best_by_domain.get(d), d, domain_offsets[d])
            for d in all_domains
        ]
    else:
        target_entries = [
            (f, os.path.splitext(os.path.basename(f))[0], 0.0) for f in target_vtps
        ]
        best_entries = [
            (f, os.path.splitext(os.path.basename(f))[0], 0.0) for f in best_vtps
        ]
        label_entries = target_entries or best_entries

    show_labels = labels

    script_content = f"""\
from paraview.simple import *

target_entries = {repr(target_entries)}
best_entries = {repr(best_entries)}
label_entries = {repr(label_entries)}

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
if {show_labels}:
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

    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_best_", delete=False
    )
    script_file.write(script_content)
    script_file.close()

    subprocess.Popen([paraview_executable, f"--script={script_file.name}"])
    n_target = len(target_vtps)
    n_best = len(best_vtps)
    print(
        f"Launched ParaView: {n_target} target surface(s) + {n_best} best surface(s)"
        f" (evaluation {best_eval:03d}) from {optimizationRunDir}"
    )


def openCustomEvaluationInParaview(
    customEvaluationDir,
    labels=True,
    paraview_executable="paraview",
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
    paraview_executable : str, optional
        Path to the ParaView executable. Defaults to ``"paraview"``.
    """
    customEvaluationDir = os.path.abspath(customEvaluationDir)

    result_vtps = sorted(glob.glob(os.path.join(customEvaluationDir, "*-result-*.vtp")))
    if not result_vtps:
        print(f"No result VTP files found in {customEvaluationDir}")
        return

    # Project dir is two levels up: customEvaluations/<evalName> -> project root
    project_dir = os.path.dirname(os.path.dirname(customEvaluationDir))
    target_vtps = sorted(
        glob.glob(os.path.join(project_dir, "domains", "targetDomain", "*-surface.vtp"))
    )

    result_stems = [os.path.splitext(os.path.basename(f))[0] for f in result_vtps]
    target_stems = [os.path.splitext(os.path.basename(f))[0] for f in target_vtps]
    show_labels = labels

    script_content = f"""\
from paraview.simple import *
import os as _os

target_files = {repr(target_vtps)}
target_labels = {repr(target_stems)}
result_files = {repr(result_vtps)}
result_labels = {repr(result_stems)}

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
    if {show_labels}:
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
    if {show_labels}:
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

    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_custom_eval_", delete=False
    )
    script_file.write(script_content)
    script_file.close()

    subprocess.Popen([paraview_executable, f"--script={script_file.name}"])
    print(
        f"Launched ParaView: {len(target_vtps)} target surface(s) + "
        f"{len(result_vtps)} result surface(s) from {customEvaluationDir}"
    )


def _viewBestCLI():
    import argparse

    parser = argparse.ArgumentParser(prog="viennafit-view-best")
    parser.add_argument(
        "optimization_run_dir", help="Path to the optimization run directory"
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
        dest="domain_spacing",
        type=float,
        default=500.0,
        help="X distance between domain columns in geometry units (default: 500.0)",
    )
    args = parser.parse_args()
    openBestInParaview(
        args.optimization_run_dir,
        labels=args.labels,
        domainSpacing=args.domain_spacing,
    )


def _viewCustomEvaluationCLI():
    import argparse

    parser = argparse.ArgumentParser(prog="viennafit-view-custom-eval")
    parser.add_argument(
        "custom_evaluation_dir", help="Path to the custom evaluation directory"
    )
    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        default=True,
        help="Hide filename annotations (shown by default)",
    )
    args = parser.parse_args()
    openCustomEvaluationInParaview(args.custom_evaluation_dir, labels=args.labels)
