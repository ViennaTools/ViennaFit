"""Module for opening VTP files in ParaView with annotations."""

import fnmatch
import glob
import os
import subprocess
import tempfile


def openInParaview(
    folder, patterns=None, translations=None, paraview_executable="paraview"
):
    """Open .vtp files in a folder in ParaView with filename annotations.

    Launches the ParaView GUI with a script that loads .vtp files found in
    the given folder. Each source is renamed to its filename stem in the Pipeline
    Browser, and a single text annotation summarizes the loaded files.

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
    labels = [os.path.splitext(os.path.basename(f))[0] for f in vtp_files]
    labels_repr = repr(labels)
    translations_repr = repr(file_translations)
    summary_text = f"Folder: {folder}\\nFiles: {len(vtp_files)}"

    script_content = f"""\
from paraview.simple import *

files = {filepaths_repr}
labels = {labels_repr}
translations = {translations_repr}

view = GetActiveViewOrCreate('RenderView')

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

# Add a single summary annotation
text = Text(Text="{summary_text}")
textDisplay = Show(text, view)
textDisplay.WindowLocation = 'Upper Left Corner'

ResetCamera()
Render()
"""

    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="paraview_vtp_", delete=False
    )
    script_file.write(script_content)
    script_file.close()

    subprocess.Popen([paraview_executable, f"--script={script_file.name}"])
    print(f"Launched ParaView with {len(vtp_files)} .vtp files from {folder}")
