# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from github2gerrit_python.gitutils import CommandError
from github2gerrit_python.gitutils import CommandResult
from github2gerrit_python.gitutils import git
from github2gerrit_python.gitutils import mask_text
from github2gerrit_python.gitutils import run_cmd
from github2gerrit_python.gitutils import run_cmd_with_retries


def _no_cov_env() -> dict[str, str]:
    keys = [
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
        "COVERAGE_PROCESS_START",
        "COVERAGE_FILE",
        "COVERAGE_RCFILE",
        "PYTEST_ADDOPTS",
        "PYTEST_CURRENT_TEST",
        "PYTEST_XDIST_WORKER",
    ]
    return dict.fromkeys(keys, "")


def test_mask_text_replaces_tokens_and_ignores_empty() -> None:
    text = (
        "token=ABC and again ABC; mixed abc should not change case-sensitively"
    )
    out = mask_text(text, masks=["ABC", ""])
    assert (
        out
        == "token=*** and again ***; mixed abc should not change case-sensitively"
    )


def test_run_cmd_success_captures_stdout_and_stderr() -> None:
    # Use the current Python to produce deterministic output without external tools.
    code = 'import sys; print("hello"); print("warn", file=sys.stderr)'
    res = run_cmd([sys.executable, "-c", code], env=_no_cov_env())
    assert res.returncode == 0
    assert "hello" in res.stdout
    assert "warn" in res.stderr


def test_run_cmd_failure_raises_and_includes_output() -> None:
    code = (
        'import sys; print("out"); print("err", file=sys.stderr); sys.exit(3)'
    )
    with pytest.raises(CommandError) as ei:
        run_cmd([sys.executable, "-c", code], check=True, env=_no_cov_env())
    err = ei.value
    assert err.returncode == 3
    assert err.stdout is not None and "out" in err.stdout
    assert err.stderr is not None and "err" in err.stderr


def test_run_cmd_timeout_raises_commanderror() -> None:
    # Sleep longer than the timeout to trigger TimeoutExpired -> CommandError
    code = "import time; time.sleep(0.2)"
    with pytest.raises(CommandError) as ei:
        run_cmd([sys.executable, "-c", code], timeout=0.05, env=_no_cov_env())
    # On timeout, returncode is None per implementation
    assert ei.value.returncode is None


def test_run_cmd_env_merge_passes_extra_env() -> None:
    code = 'import os; v=os.getenv("EXTRA_VAR",""); print(v, end="")'
    res = run_cmd(
        [sys.executable, "-c", code],
        env={**_no_cov_env(), "EXTRA_VAR": "value-123"},
    )
    assert res.stdout == "value-123"


def test_run_cmd_with_retries_retries_on_transient_error_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make time.sleep a no-op to avoid slowing the test during retries.
    sleeps: list[float] = []

    def _fake_sleep(secs: float) -> None:
        sleeps.append(secs)

    monkeypatch.setattr("time.sleep", _fake_sleep)

    # Script: on first run create a marker and exit non-zero with a transient error in stderr.
    # On subsequent runs (marker exists) print success and exit 0.
    marker = tmp_path / "attempted"

    code = f"""
import os, sys, pathlib
p = pathlib.Path({str(marker)!r})
if not p.exists():
    p.write_text("1", encoding="utf-8")
    print("could not resolve host: example.com", file=sys.stderr)
    sys.exit(128)
else:
    print("ok")
"""

    res = run_cmd_with_retries(
        [sys.executable, "-c", code], cwd=tmp_path, env=_no_cov_env()
    )
    # Should succeed after retry, marker must exist, and we should have slept at least once
    assert marker.exists()
    assert res.returncode == 0
    assert "ok" in res.stdout
    assert len(sleeps) >= 1


def test_run_cmd_with_retries_does_not_retry_on_non_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Track if sleep is called (it should not be for non-transient errors).
    slept: list[float] = []

    def _fake_sleep(secs: float) -> None:
        slept.append(secs)

    monkeypatch.setattr("time.sleep", _fake_sleep)

    # Exit with a non-transient failure message; should raise immediately without retries.
    code = (
        'import sys; print("permanent failure", file=sys.stderr); sys.exit(2)'
    )

    with pytest.raises(CommandError) as ei:
        run_cmd_with_retries([sys.executable, "-c", code], env=_no_cov_env())

    err = ei.value
    assert err.returncode == 2
    # Ensure we did not sleep for retries (no transient retry)
    assert slept == []


def test_git_retries_on_transient_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Make sleep a no-op and record delays
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    def fake_run_cmd(cmd: list[str], **kwargs: object) -> CommandResult:
        # First call: fail with a transient error pattern
        attempts["n"] += 1
        if attempts["n"] == 1:
            return CommandResult(
                returncode=128,
                stdout="",
                stderr="could not resolve host: example.com",
            )
        # Subsequent calls: succeed
        return CommandResult(returncode=0, stdout="ok", stderr="")

    # Patch the low-level executor used by run_cmd_with_retries
    monkeypatch.setattr("github2gerrit_python.gitutils.run_cmd", fake_run_cmd)

    # Invoke the git wrapper; it should retry and then succeed
    res = git(["fetch", "origin"])
    assert res.returncode == 0
    assert attempts["n"] >= 2
    assert len(sleeps) >= 1


def test_git_raises_on_non_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Non-transient failure should cause git() to raise CommandError (check=True)
    def fake_run_cmd(cmd: list[str], **kwargs: object) -> CommandResult:
        return CommandResult(
            returncode=2, stdout="", stderr="permanent failure"
        )

    monkeypatch.setattr("github2gerrit_python.gitutils.run_cmd", fake_run_cmd)

    with pytest.raises(CommandError):
        git(["fetch", "origin"])


def test_run_cmd_with_retries_retries_on_http2_stream_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a transient HTTP/2 stream error on first attempt, then success.
    attempts = {"n": 0}

    def fake_run_cmd(cmd: list[str], **kwargs: object) -> CommandResult:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return CommandResult(
                returncode=128,
                stdout="",
                stderr="HTTP/2 stream 0 was not closed cleanly",
            )
        return CommandResult(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("github2gerrit_python.gitutils.run_cmd", fake_run_cmd)

    res = run_cmd_with_retries(["git", "fetch", "origin"])
    assert res.returncode == 0
    assert "ok" in res.stdout
    assert attempts["n"] >= 2


def test_git_last_commit_trailers_parsing_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch git_show to return a crafted commit message body with various trailer forms.
    import github2gerrit_python.gitutils as gitutils

    body = (
        "Subject line\n\n"
        "Key: Value\n"
        "Key: Another Value\n"
        "Empty: \n"
        "NoColonLine\n"
        "Signed-off-by: Dev <dev@example.org>\n"
        "Change-Id: Ideadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n"
        "Change-Id: Iabc123abc123abc123abc123abc123abc123ab\n"
    )

    monkeypatch.setattr(gitutils, "git_show", lambda rev, **kw: body)

    # Without filter: expect only keys with non-empty values collected.
    trailers_all = gitutils.git_last_commit_trailers()
    assert "Key" in trailers_all
    assert trailers_all["Key"] == ["Value", "Another Value"]
    # 'Empty' should be ignored due to empty value
    assert "Empty" not in trailers_all
    # NoColonLine should be ignored as it lacks ':'
    assert "NoColonLine" not in trailers_all
    # Signed-off-by preserved
    assert trailers_all["Signed-off-by"] == ["Dev <dev@example.org>"]
    # Change-Id collects both values
    assert trailers_all["Change-Id"] == [
        "Ideadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "Iabc123abc123abc123abc123abc123abc123ab",
    ]

    # With filter keys: only Change-Id should be returned.
    trailers_change = gitutils.git_last_commit_trailers(keys=["Change-Id"])
    assert list(trailers_change.keys()) == ["Change-Id"]
    assert trailers_change["Change-Id"] == [
        "Ideadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "Iabc123abc123abc123abc123abc123abc123ab",
    ]


def test_git_quiet_suppresses_failure_logging(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that git_quiet doesn't log failures like regular git commands do."""
    from github2gerrit_python.gitutils import git_quiet

    # Mock run_cmd to simulate a command failure
    def fake_run_cmd(cmd: list[str], **kwargs: object) -> CommandResult:
        return CommandResult(
            returncode=1, stdout="", stderr="config key not found"
        )

    monkeypatch.setattr("github2gerrit_python.gitutils.run_cmd", fake_run_cmd)

    # Clear any existing log messages
    caplog.clear()

    # Call git_quiet with a command that will "fail"
    result = git_quiet(["config", "--get", "nonexistent.key"])

    # Should return failure result without logging
    assert result.returncode == 1
    assert result.stderr == "config key not found"

    # No debug logs should be captured (git_quiet suppresses them)
    debug_logs = [
        record for record in caplog.records if record.levelname == "DEBUG"
    ]
    failure_logs = [
        record for record in debug_logs if "Command failed" in record.message
    ]
    assert len(failure_logs) == 0
