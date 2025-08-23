# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from github2gerrit_python.cli import app


runner = CliRunner()


def test_conflicting_options_error_message_in_stderr(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env["SUBMIT_SINGLE_COMMITS"] = "true"
    env["USE_PR_AS_COMMIT"] = "true"

    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 2
    assert "cannot be enabled at the same time" in result.stderr


def test_missing_required_input_error_message_in_stderr(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    # Remove a required input to trigger a validation error path
    env.pop("GERRIT_KNOWN_HOSTS", None)

    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 2
    assert "Missing required input" in result.stderr


def _base_env(tmp_path: Path) -> dict[str, str]:
    """Return a baseline environment with required inputs set."""
    event_path = tmp_path / "event.json"
    # Default to an event with a PR number
    event = {"action": "opened", "pull_request": {"number": 7}}
    event_path.write_text(json.dumps(event), encoding="utf-8")

    return {
        # Required inputs
        "GERRIT_KNOWN_HOSTS": "example.com ssh-rsa AAAAB3Nza...",
        "GERRIT_SSH_PRIVKEY_G2G": "-----BEGIN KEY-----\nabc\n-----END KEY-----",
        "GERRIT_SSH_USER_G2G": "gerrit-bot",
        "GERRIT_SSH_USER_G2G_EMAIL": "gerrit-bot@example.org",
        # Optional inputs
        "ORGANIZATION": "example",
        "REVIEWERS_EMAIL": "",
        # GitHub context
        "GITHUB_EVENT_NAME": "pull_request_target",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_REPOSITORY": "example/repo",
        "GITHUB_REPOSITORY_OWNER": "example",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "12345",
        "GITHUB_SHA": "deadbeef",
        "GITHUB_BASE_REF": "main",
        "GITHUB_HEAD_REF": "feature",
        "G2G_TEST_MODE": "true",
    }


def test_conflicting_options_exits_2(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env["SUBMIT_SINGLE_COMMITS"] = "true"
    env["USE_PR_AS_COMMIT"] = "true"

    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 2
    assert (
        "cannot be enabled at the same time" in result.stdout
        or "cannot be enabled at the same time" in result.stderr
    )


def test_missing_required_inputs_exits_2(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    # Remove one required input to trigger validation error
    env.pop("GERRIT_KNOWN_HOSTS", None)

    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 2
    assert "Missing required input" in (result.stdout + result.stderr)


def test_parses_pr_number_and_returns_zero(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    # Ensure non-conflicting options and sane defaults
    env["SUBMIT_SINGLE_COMMITS"] = "false"
    env["USE_PR_AS_COMMIT"] = "false"
    env["FETCH_DEPTH"] = "10"

    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 0
    # The CLI currently only validates and exits cleanly
    assert "Validation complete" in (result.stdout + result.stderr)


def test_no_pr_context_exits_2(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    # Overwrite event to remove PR number
    event_path = Path(env["GITHUB_EVENT_PATH"])
    event_path.write_text(json.dumps({}), encoding="utf-8")
    env["GITHUB_EVENT_NAME"] = "workflow_dispatch"
    # Disable test mode to ensure non-zero exit on missing PR context
    env.pop("G2G_TEST_MODE", None)
    # Force non-bulk path to avoid GitHub API token requirement
    env["SYNC_ALL_OPEN_PRS"] = "false"

    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 2
    assert "requires a valid pull request context" in (
        result.stdout + result.stderr
    )
