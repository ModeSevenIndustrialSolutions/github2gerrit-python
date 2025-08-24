# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any


# Ensure src directory is on sys.path so tests can import the package without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Isolate coverage output to a unique temp file to avoid data mixing across runs
if not os.getenv("COVERAGE_FILE"):
    import tempfile
    import uuid

    cov_file = (
        Path(tempfile.gettempdir())
        / f".coverage.pytest.{os.getpid()}.{uuid.uuid4().hex}"
    )
    os.environ["COVERAGE_FILE"] = str(cov_file)


def _remove_path(p: Path) -> None:
    try:
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.exists() and p.is_dir():
            shutil.rmtree(p)
    except Exception as exc:
        # Do not fail test collection/session start if cleanup cannot proceed.
        print(
            f"[conftest] Warning: failed to remove {p}: {exc}", file=sys.stderr
        )


def _remove_coverage_files(bases: Iterable[Path]) -> None:
    for base in bases:
        base_path = base
        with suppress(Exception):
            base_path = base_path.resolve()

        # Standard coverage data file
        cov_file = base_path / ".coverage"
        if cov_file.exists():
            _remove_path(cov_file)

        # Any leftover parallel/previous data artifacts like .coverage.hostname.pid.*
        for child in base_path.iterdir():
            name = child.name
            if name.startswith(".coverage.") and child.is_file():
                _remove_path(child)

    # Respect explicit COVERAGE_FILE override if set
    cov_env = os.getenv("COVERAGE_FILE", "").strip()
    if cov_env:
        p = Path(cov_env)
        if p.exists():
            _remove_path(p)


def pytest_sessionstart(session: Any) -> None:
    """
    Ensure clean coverage data by removing any pre-existing coverage files
    that could cause branch/statement data mixing errors when combining.
    """
    bases: set[Path] = {Path.cwd()}
    rootpath = getattr(session.config, "rootpath", None)
    if isinstance(rootpath, Path):
        bases.add(rootpath)
    # Some pytest versions expose 'invocation_params.dir'
    inv_params = getattr(session.config, "invocation_params", None)
    if inv_params is not None:
        inv_dir = getattr(inv_params, "dir", None)
        if isinstance(inv_dir, Path):
            bases.add(inv_dir)

    # Force hermetic config path and token for tests
    try:
        import tempfile

        cfg_tmp = (
            Path(tempfile.gettempdir()) / f"g2g-empty-config-{os.getpid()}.ini"
        )
        # Ensure the file exists and is empty
        cfg_tmp.write_text("", encoding="utf-8")
        os.environ["G2G_CONFIG_PATH"] = str(cfg_tmp)
    except Exception:
        # Fallback: use a non-existent path to disable config loading entirely
        os.environ["G2G_CONFIG_PATH"] = "/dev/null/nonexistent-config.ini"

    # Provide a dummy token so any incidental GitHub client construction succeeds
    os.environ.setdefault("GITHUB_TOKEN", "dummy")

    # Ensure tests don't write to real GitHub output files
    if "GITHUB_OUTPUT" not in os.environ:
        os.environ["GITHUB_OUTPUT"] = "/dev/null"
    _remove_coverage_files(bases)
