#!/usr/bin/env python3
"""
Local pre-pull check for ViennaFit's upstream dependencies (ViennaPS/ViennaLS).

Run this from your dev loop *before* you `git pull` the sibling ViennaPS/ViennaLS
repos and rebuild. It tells you, without building anything:

  * installed vs. local-repo-HEAD vs. remote versions,
  * how many commits each sibling repo is behind its remote,
  * whether any ViennaPS/ViennaLS symbol ViennaFit actually uses is removed or
    renamed in the unpulled binding stubs (.pyi) -- i.e. a likely break.

It then runs the live contract+functional compat checks against whatever is
currently installed.

Usage:
    python scripts/check_upstream.py
    python scripts/check_upstream.py --vps ../ViennaPS --vls ../ViennaLS
    python scripts/check_upstream.py --no-fetch     # skip network git fetch

Sibling repos default to ../ViennaPS and ../ViennaLS relative to this repo.
Missing repos are skipped (the installed-version + live checks still run).
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Binding stub files where the used symbols live, per repo.
PYI_GLOBS = {
    "ViennaPS": ["python/viennaps/__init__.pyi"],
    "ViennaLS": ["python/viennals/__init__.pyi"],
}


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
    )


def _cmake_version(repo, ref):
    """Read project VERSION from CMakeLists.txt at a given git ref."""
    out = _git(repo, "show", f"{ref}:CMakeLists.txt")
    if out.returncode != 0:
        return "?"
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.upper().startswith("VERSION ") and s[8:9].isdigit():
            return s.split()[1].rstrip(")")
    return "?"


def _installed_versions():
    print("== Installed in current environment ==")
    for mod in ("viennaps", "viennals", "viennafit"):
        try:
            m = __import__(mod)
            print(f"  {mod:10s} {getattr(m, '__version__', '?')}")
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  {mod:10s} NOT IMPORTABLE ({e})")
    print()


def _used_symbols():
    """Pull the symbol contract straight from the compat test (single source)."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
    import test_upstream_compat as t

    return {"ViennaPS": list(t.VPS_SYMBOLS), "ViennaLS": list(t.VLS_SYMBOLS)}


def _check_repo(name, repo, do_fetch, used):
    print(f"== {name}  ({repo}) ==")
    if not os.path.isdir(os.path.join(repo, ".git")):
        print("  not a git repo -- skipped\n")
        return
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if do_fetch:
        _git(repo, "fetch", "--quiet")
    remote = f"origin/{branch}"
    local_v = _cmake_version(repo, "HEAD")
    remote_v = _cmake_version(repo, remote)
    behind = _git(repo, "rev-list", "--count", f"HEAD..{remote}").stdout.strip() or "?"
    print(f"  branch {branch}: HEAD {local_v}  ->  {remote} {remote_v}  ({behind} commits behind)")

    if behind in ("0", "?"):
        print("  up to date with remote.\n" if behind == "0" else "  could not compare to remote.\n")
        return

    # Scan the unpulled .pyi diff for removals of symbols ViennaFit uses.
    removed = []
    for rel in PYI_GLOBS.get(name, []):
        diff = _git(repo, "diff", f"HEAD..{remote}", "--", rel).stdout
        if not diff:
            continue
        removed_lines = [ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        added_lines = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        for sym in used[name]:
            token = f" {sym}"
            gone = any(token in r for r in removed_lines)
            still = any(token in a for a in added_lines) or any(sym in a for a in added_lines)
            if gone and not still:
                removed.append(f"{sym} (in {rel})")
    if removed:
        print("  !! Symbols ViennaFit uses appear removed/renamed upstream:")
        for r in removed:
            print(f"       - {r}")
        print("     Review before pulling/rebuilding.\n")
    else:
        print("  No ViennaFit-used symbol removed in the unpulled binding stubs.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vps", default=os.path.join(REPO_ROOT, "..", "ViennaPS"))
    ap.add_argument("--vls", default=os.path.join(REPO_ROOT, "..", "ViennaLS"))
    ap.add_argument("--no-fetch", action="store_true", help="skip git fetch")
    ap.add_argument("--skip-tests", action="store_true", help="skip live compat tests")
    args = ap.parse_args()

    _installed_versions()
    used = _used_symbols()
    _check_repo("ViennaPS", os.path.abspath(args.vps), not args.no_fetch, used)
    _check_repo("ViennaLS", os.path.abspath(args.vls), not args.no_fetch, used)

    if args.skip_tests:
        return 0
    print("== Live compatibility checks (installed versions) ==")
    rc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "tests", "test_upstream_compat.py")]
    ).returncode
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
