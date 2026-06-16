# Changelog

All notable changes to ViennaFit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-06-16

### Added

- Multi-domain optimization: fit parameters across multiple geometries
  simultaneously, with train/validate domain subsets
- Bayesian optimization via Ax/BoTorch (optional dependency group
  `bayesian`: `pip install viennafit[bayesian]`)
- CMA-ES optimizer support via `pycma`
- Chamfer distance metric (CCH), critical dimension metric (CCD), and
  experimental CSF-IS metric
- Image annotation tool (`viennafit-annotate`) for extracting target points
  from experimental images
- ParaView viewer with side-by-side multi-domain display and CLI entry
  points (`viennafit-view-best`, `viennafit-view-custom-eval`)
- Optimization run summaries and finalization plots
- Parameter position snapshots on reaching a new minimum
- Documentation site (MkDocs) with tutorials
- `CITATION.cff` for citation support on GitHub
- `py.typed` marker so downstream type checkers can use ViennaFit's type hints
- `CHANGELOG.md` and `CONTRIBUTING.md`

### Changed

- **Breaking:** requires ViennaPS >= 4.0.0 (ViennaPS 3.x is no longer
  supported; use ViennaFit 1.x for ViennaPS 3.5.1)
- Public API converted to camelCase for consistency with the ViennaTools family
- Relaxed exact dependency pins (`numpy`, `cma`, `dlib-bin`) to lower bounds
- Package version is now single-sourced from `viennafit.__version__`
- Restored verbatim GPL-3.0 license text so the license is machine-detectable;
  the copyright notice moved to `NOTICE`

## [1.1.0]

### Added

- Nevergrad optimizer support
- Custom parameter evaluation (grid search, specific combinations,
  repeatability tests)
- Postprocessing submodule with progress plotting
- Improved progress files, evaluation saving, and notes option for runs

## [1.0.0]

Initial release: project management, optimization of ViennaPS process
sequences with dlib, level-set distance metrics, sensitivity analysis, and
progress tracking.

[2.0.0]: https://github.com/ViennaTools/ViennaFit/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/ViennaTools/ViennaFit/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ViennaTools/ViennaFit/releases/tag/v1.0.0
