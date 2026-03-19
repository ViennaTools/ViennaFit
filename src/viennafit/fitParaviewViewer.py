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


def openBestInParaview(
    optimizationRunDir,
    paraview_executable="paraview",
):
    """Open the current best optimization result alongside target surfaces in ParaView.

    Reads ``progressBest.csv`` (or falls back to ``progressAll.csv``) to find
    the best evaluation number, then opens the corresponding VTP files from the
    ``progress/`` folder alongside the target surfaces from the project's
    ``domains/targetDomain/`` folder.

    Target surfaces are shown in green with line width 3; best simulated
    surfaces use default styling.

    Parameters
    ----------
    optimizationRunDir : str
        Path to the optimization run directory (contains ``progressBest.csv``
        and the ``progress/`` subfolder).
    paraview_executable : str, optional
        Path to the ParaView executable. Defaults to ``"paraview"``.
    """
    import csv

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
    if not best_vtps:
        # Also try without domain suffix (single-surface case)
        best_vtps = sorted(
            glob.glob(os.path.join(progress_dir, f"{runName}-{best_eval:03d}.vtp"))
        )

    # Glob target surfaces
    project_dir = os.path.dirname(os.path.dirname(optimizationRunDir))
    target_vtps = sorted(
        glob.glob(os.path.join(project_dir, "domains", "targetDomain", "*-surface.vtp"))
    )

    if not best_vtps and not target_vtps:
        print(f"No VTP files found for evaluation {best_eval:03d} in {progress_dir}")
        return

    target_stems = [os.path.splitext(os.path.basename(f))[0] for f in target_vtps]
    best_stems = [os.path.splitext(os.path.basename(f))[0] for f in best_vtps]

    script_content = f"""\
from paraview.simple import *

target_files = {repr(target_vtps)}
target_labels = {repr(target_stems)}
best_files = {repr(best_vtps)}
best_labels = {repr(best_stems)}

view = GetActiveViewOrCreate('RenderView')
view.InteractionMode = '2D'

text_displays = []

import os as _os
for filepath, label in zip(target_files, target_labels):
    source = OpenDataFile(filepath)
    RenameSource(label, source)
    x_offset = 200.0 if 'wider' in _os.path.basename(filepath) else 0.0
    if x_offset:
        transform = Transform(Input=source)
        transform.Transform.Translate = [x_offset, 0.0, 0.0]
        RenameSource(label + " (translated)", transform)
        display = Show(transform, view)
        Hide(source, view)
    else:
        display = Show(source, view)
    display.AmbientColor = [0.0, 1.0, 0.0]
    display.DiffuseColor = [0.0, 1.0, 0.0]
    display.LineWidth = 3.0
    text = Text(Text=label)
    textDisplay = Show(text, view)
    textDisplay.FontFamily = 'Arial'
    textDisplay.FontSize = 20
    textDisplay.Color = [0, 0, 0]
    textDisplay.WindowLocation = 'Any Location'
    bounds = source.GetDataInformation().GetBounds()
    text_displays.append((textDisplay, [(bounds[0] + bounds[1]) / 2 + x_offset, bounds[2], (bounds[4] + bounds[5]) / 2]))
    RenameSource(label + " (label)", text)

for filepath, label in zip(best_files, best_labels):
    source = OpenDataFile(filepath)
    RenameSource(label, source)
    x_offset = 200.0 if 'wider' in _os.path.basename(filepath) else 0.0
    if x_offset:
        transform = Transform(Input=source)
        transform.Transform.Translate = [x_offset, 0.0, 0.0]
        RenameSource(label + " (translated)", transform)
        display = Show(transform, view)
        Hide(source, view)
    else:
        display = Show(source, view)
    display.LineWidth = 3.0
    text = Text(Text=label)
    textDisplay = Show(text, view)
    textDisplay.FontFamily = 'Arial'
    textDisplay.FontSize = 20
    textDisplay.Color = [0, 0, 0]
    textDisplay.WindowLocation = 'Any Location'
    bounds = source.GetDataInformation().GetBounds()
    text_displays.append((textDisplay, [(bounds[0] + bounds[1]) / 2 + x_offset, bounds[2], (bounds[4] + bounds[5]) / 2]))
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


def _viewBestCLI():
    import sys

    if len(sys.argv) < 2:
        print("Usage: viennafit-view-best <optimization-run-dir>")
        sys.exit(1)
    openBestInParaview(sys.argv[1])
