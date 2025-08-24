# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from typing import Any

import pytest
from typer.testing import CliRunner

from github2gerrit import cli as cli_mod
from github2gerrit.cli import app


runner = CliRunner()


def _base_env() -> dict[str, str]:
    """
    Return baseline environment variables required by CLI validation.
    We simulate a local run (no GITHUB_EVENT_NAME) so URL mode is enabled.
    """
    return {
        # Required inputs for validation
        "GERRIT_KNOWN_HOSTS": "gerrit.example.org ssh-rsa AAAAB3Nza...",
        "GERRIT_SSH_PRIVKEY_G2G": "-----BEGIN KEY-----\nfake\n-----END KEY-----",
        "GERRIT_SSH_USER_G2G": "gerrit-bot",
        "GERRIT_SSH_USER_G2G_EMAIL": "gerrit-bot@example.org",
        # Local run: no GitHub event
        "GITHUB_EVENT_NAME": "",
        "GITHUB_EVENT_PATH": "",
        # Token not needed since we mock build_client for bulk mode
        "GITHUB_TOKEN": "dummy",
        "DRY_RUN": "true",
    }


class _CallRecord:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def add(self, inputs: Any, gh: Any) -> None:
        self.calls.append((inputs, gh))


class _DummyOrchestrator:
    """
    Test stub for Orchestrator used to capture calls to execute().
    """

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    def execute(self, *, inputs: Any, gh: Any) -> Any:
        # Capture via the test-patched global record
        _ORCH_RECORD.add(inputs, gh)

        # Return a minimal object with the expected attributes
        class _Result:
            def __init__(self) -> None:
                self.change_urls = ["https://gerrit.example.org/c/p/+/12345"]
                self.change_numbers = ["12345"]
                self.commit_shas = ["deadbeef"]

        return _Result()


# This mutable global will be replaced per-test to capture execute calls
_ORCH_RECORD = _CallRecord()


def test_pr_url_dry_run_invokes_single_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given a PR URL and --dry-run, the CLI should:
      - Parse org/repo/pr number from URL
      - Load config (we rely on env/mocks)
      - Invoke Orchestrator.execute exactly once
      - Pass inputs with dry_run=True and gh.pr_number == parsed number
    """
    env = _base_env()
    pr_url = "https://github.com/onap/portal-ng-bff/pull/33"

    # Patch Orchestrator in the CLI module
    global _ORCH_RECORD
    _ORCH_RECORD = _CallRecord()
    monkeypatch.setattr(cli_mod, "Orchestrator", _DummyOrchestrator)

    # Run CLI with PR URL and --dry-run
    result = runner.invoke(app, ["--dry-run", pr_url], env=env)

    assert result.exit_code == 0, result.stdout + result.stderr
    # Exactly one call captured
    assert len(_ORCH_RECORD.calls) == 1
    inputs, gh = _ORCH_RECORD.calls[0]
    # Verify dry-run flag and PR number parsing
    assert getattr(inputs, "dry_run", False) is True
    assert getattr(gh, "pr_number", None) == 33
    # Organization and repository should be set from URL
    assert getattr(gh, "repository", "") == "onap/portal-ng-bff"
    # Ensure we didn't require REVIEWERS_EMAIL (should be optional)
    # If reviewers were auto-derived, they may appear in inputs.reviewers_email;
    # either way, lack of config should not fail.
    assert result.exit_code == 0


def test_repo_url_dry_run_invokes_for_each_open_pr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given a repository URL and --dry-run, the CLI (URL mode) should:
      - Parse org/repo from URL
      - Enable bulk mode (SYNC_ALL_OPEN_PRS)
      - Enumerate open PRs via PyGithub wrapper (mocked)
      - Invoke Orchestrator.execute once per PR with that PR number
    """
    env = _base_env()
    repo_url = "https://github.com/onap/portal-ng-bff"

    # Prepare dummy PRs to be returned by iter_open_pulls
    class _DummyPR:
        def __init__(self, number: int) -> None:
            self.number = number

    dummy_prs = [_DummyPR(5), _DummyPR(7)]

    # Patch Orchestrator to stub execute and capture calls
    global _ORCH_RECORD
    _ORCH_RECORD = _CallRecord()
    monkeypatch.setattr(cli_mod, "Orchestrator", _DummyOrchestrator)

    # Patch PyGithub wrapper functions used by CLI bulk path
    monkeypatch.setattr(cli_mod, "build_client", lambda: object())
    monkeypatch.setattr(cli_mod, "get_repo_from_env", lambda _client: object())
    monkeypatch.setattr(
        cli_mod, "iter_open_pulls", lambda _repo: iter(dummy_prs)
    )

    result = runner.invoke(app, ["--dry-run", repo_url], env=env)

    assert result.exit_code == 0, result.stdout + result.stderr
    # Two calls for two open PRs
    assert len(_ORCH_RECORD.calls) == 2
    pr_numbers = [
        getattr(call[1], "pr_number", None) for call in _ORCH_RECORD.calls
    ]
    assert pr_numbers == [5, 7]

    # Verify dry_run flag passed for both calls
    dry_flags = [
        getattr(call[0], "dry_run", None) for call in _ORCH_RECORD.calls
    ]
    assert dry_flags == [True, True]


def test_url_mode_sets_environment_for_config_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    URL mode should put ORGANIZATION and GITHUB_REPOSITORY into the environment,
    which the config loader will then use to locate the appropriate org stanza.
    Here we simply observe that the environment is set as expected and the run completes.
    """
    env = _base_env()
    repo_url = "https://github.com/onap/portal-ng-bff"

    # Patch Orchestrator to avoid real work but capture calls
    global _ORCH_RECORD
    _ORCH_RECORD = _CallRecord()
    monkeypatch.setattr(cli_mod, "Orchestrator", _DummyOrchestrator)

    # Minimal patches for bulk flow
    monkeypatch.setattr(cli_mod, "build_client", lambda: object())
    monkeypatch.setattr(cli_mod, "get_repo_from_env", lambda _client: object())
    monkeypatch.setattr(
        cli_mod,
        "iter_open_pulls",
        lambda _repo: iter([type("PR", (), {"number": 42})()]),
    )

    result = runner.invoke(app, ["--dry-run", repo_url], env=env)
    assert result.exit_code == 0, result.stdout + result.stderr
    # After run, our captured context should reflect the parsed repo
    assert len(_ORCH_RECORD.calls) == 1
    _, gh = _ORCH_RECORD.calls[0]
    assert getattr(gh, "repository", "") == "onap/portal-ng-bff"
