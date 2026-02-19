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
