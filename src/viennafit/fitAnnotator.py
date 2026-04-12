"""Interactive annotator for SEM images.

Produces space-separated x y coordinate files compatible with
``readPointsFromFile``.

CLI usage:
    viennafit-annotate <image_path> [--scale NM_PER_PX] [--output FILE]

Keys:
    a  — add mode (default)
    d  — delete mode
    Close window to save and exit.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def annotate(image_path, scale=1.0, output=None):
    """Interactively annotate boundary points on a SEM image.

    Opens the image in a matplotlib window. Click once to set the coordinate
    origin, then click to add points. Press ``a`` to switch to add mode and
    ``d`` to switch to delete mode. Close the window to save.

    Output is a space-separated ``x y`` file (one point per line) compatible
    with :func:`readPointsFromFile`.

    Parameters
    ----------
    image_path : str or Path
        Path to the SEM image.
    scale : float, optional
        Nanometres per pixel. When 1.0 (default) coordinates are in pixels.
    output : str or Path, optional
        Output file path. Defaults to ``<stem>_points_<unit>.dat`` next to the
        image.

    Returns
    -------
    Path
        Path to the written output file, or ``None`` if no points were picked.
    """
    image_path = Path(image_path)
    img = np.array(Image.open(image_path).convert("L"))

    unit = "nm" if scale != 1.0 else "px"
    if output is None:
        output = image_path.parent / f"{image_path.stem}_points_{unit}.dat"
    output = Path(output)

    state = {
        "mode": "set_origin",   # "set_origin" | "add" | "delete"
        "origin": None,          # (ox, oy) in pixel coords
        "points": [],            # list of (phys_x, phys_y)
        "px_points": [],         # list of (px_x, px_y)
    }

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(image_path.name)
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_axis_off()

    origin_marker = ax.plot([], [], "m+", markersize=16, markeredgewidth=2)[0]
    scatter = ax.scatter([], [], c="red", s=30, zorder=5)
    labels = []

    status_text = ax.text(
        0.01, 0.99, "Click to set origin",
        transform=ax.transAxes,
        color="yellow", fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )
    coord_text = ax.text(
        0.5, 0.01, "",
        transform=ax.transAxes,
        color="cyan", fontsize=14,
        va="bottom", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )

    def _mode_label():
        if state["mode"] == "set_origin":
            return "Click to set origin"
        n = len(state["points"])
        tag = "ADD" if state["mode"] == "add" else "DELETE"
        return f"[{tag}] {n} pts"

    def _redraw():
        if state["origin"] is not None:
            ox, oy = state["origin"]
            origin_marker.set_data([ox], [oy])

        for lbl in labels:
            lbl.remove()
        labels.clear()

        if state["px_points"]:
            xs, ys = zip(*state["px_points"])
            scatter.set_offsets(np.column_stack([xs, ys]))
            for i, (px, py) in enumerate(state["px_points"]):
                lbl = ax.text(px + 4, py - 4, str(i),
                              color="red", fontsize=7, zorder=6)
                labels.append(lbl)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        status_text.set_text(_mode_label())
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        px_x, px_y = event.xdata, event.ydata

        if state["mode"] == "set_origin":
            state["origin"] = (px_x, px_y)
            state["mode"] = "add"
            _redraw()
            return

        ox, oy = state["origin"]

        if state["mode"] == "add":
            phys_x = (px_x - ox) * scale
            phys_y = -(px_y - oy) * scale
            state["points"].append((phys_x, phys_y))
            state["px_points"].append((px_x, px_y))
            _redraw()

        elif state["mode"] == "delete":
            if not state["px_points"]:
                return
            pts = np.array(state["px_points"])
            dists = np.hypot(pts[:, 0] - px_x, pts[:, 1] - px_y)
            idx = int(np.argmin(dists))
            state["points"].pop(idx)
            state["px_points"].pop(idx)
            _redraw()

    def on_motion(event):
        if event.inaxes is not ax or state["origin"] is None:
            return
        ox, oy = state["origin"]
        phys_x = (event.xdata - ox) * scale
        phys_y = -(event.ydata - oy) * scale
        coord_text.set_text(f"x={phys_x:.2f}  y={phys_y:.2f}  {unit}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "a":
            if state["mode"] != "set_origin":
                state["mode"] = "add"
                _redraw()
        elif event.key == "d":
            if state["mode"] != "set_origin":
                state["mode"] = "delete"
                _redraw()

    result = [None]

    def on_close(event):
        if not state["points"]:
            print("No points picked — file not written.")
            return
        rows = "\n".join(f"{x} {y}" for x, y in state["points"])
        output.write_text(rows + "\n")
        print(f"Saved {len(state['points'])} points → {output}")
        result[0] = output

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    plt.tight_layout()
    plt.show()

    return result[0]


def _annotateCLI():
    parser = argparse.ArgumentParser(
        description="Interactive annotator for SEM images. "
                    "Output is compatible with readPointsFromFile."
    )
    parser.add_argument("image", help="Path to the SEM image")
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="nm per pixel (default: 1.0, output in px)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: <stem>_points_<unit>.dat)"
    )
    args = parser.parse_args()
    annotate(args.image, args.scale, args.output)
