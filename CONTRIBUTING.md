# Contributing to ViennaFit

Thank you for your interest in contributing to ViennaFit! Contributions of
all kinds are welcome: bug reports, feature requests, documentation
improvements, and code.

## Reporting Issues

Please report bugs and request features through the
[GitHub issue tracker](https://github.com/ViennaTools/ViennaFit/issues).

For bug reports, include:

- ViennaFit, ViennaPS, and Python versions
  (`python -c "import viennafit, viennaps; print(viennafit.__version__, viennaps.__version__)"`)
- Operating system
- A minimal script that reproduces the problem
- The full error message and traceback

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ViennaTools/ViennaFit
cd ViennaFit

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Install and activate the formatting hooks
pip install pre-commit
pre-commit install
```

## Code Style

- Code is formatted with [black](https://github.com/psf/black) (line length
  88); the pre-commit hook and CI enforce this. Run `black .` before
  committing if you don't use pre-commit.
- The public API uses **camelCase** naming (e.g. `setProcessSequence`,
  `plotParameterProgression`) for consistency with the ViennaTools family
  (ViennaPS, ViennaLS). Please follow this convention for new public
  functions and methods.
- Add docstrings to new functions and classes, and type hints to function
  signatures.

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, keeping commits focused and messages descriptive.
3. Update documentation (`docs/`) and `CHANGELOG.md` if your change affects
   user-facing behavior.
4. Open a pull request against `main` describing what the change does and
   why.

## Questions

For questions that are not bug reports, open a
[discussion or issue](https://github.com/ViennaTools/ViennaFit/issues), or
contact the maintainers at viennatools@iue.tuwien.ac.at.
