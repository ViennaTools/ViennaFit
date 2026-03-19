# ViennaFit Notebooks

Interactive Jupyter notebooks for the ViennaFit tutorials.

## Setup

```bash
pip install jupytext
pip install jupyter
```

Or install ViennaFit with the dev extras:

```bash
pip install "ViennaFit[dev]"
```

## Run a notebook

```bash
jupyter notebook notebooks/tutorial-1-basic-optimization.ipynb
```

## Sync notebooks with docs

The notebooks are paired with the Markdown tutorial files via jupytext.
Both files stay in sync — edit either one, then sync:

```bash
# After editing the Markdown source in docs/
jupytext --sync docs/tutorials/tutorial-1-basic-optimization.md

# After editing the notebook
jupytext --sync notebooks/tutorial-1-basic-optimization.ipynb
```

## File pairing

| Notebook | Paired Markdown |
|---|---|
| `notebooks/tutorial-1-basic-optimization.ipynb` | `docs/tutorials/tutorial-1-basic-optimization.md` |
