# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from github2gerrit import cli as cli_mod
from github2gerrit.cli import app


runner = CliRunner()


class _DummyResult:
    def __init__(
        self, urls: list[str], nums: list[str], shas: list[str] | None = None
    ) -> None:
        self.change_urls = urls
        self.change_numbers = nums
        self.commit_shas = shas or []


class _DummyOrchestratorSingle:
    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    def execute(self, *, inputs: Any, gh: Any) -> Any:
        # Simulate the orchestrator also exporting commit sha(s) in the environment
        # so the CLI writes them to $GITHUB_OUTPUT.
        os.environ["GERRIT_COMMIT_SHA"] = "deadbeefcafebabe"
        return _DummyResult(
            urls=["https://gerrit.example.org/c/repo/+/101"],
            nums=["101"],
            shas=["deadbeefcafebabe"],
        )


class _DummyOrchestratorMulti:
    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace

    def execute(self, *, inputs: Any, gh: Any) -> Any:
        # For multi-PR path we only need urls/nums to be aggregated
        # The CLI only writes commit_sha if present in env; we omit it here.
        return _DummyResult(
            urls=[f"https://gerrit.example.org/c/repo/+/{gh.pr_number}"],
            nums=[str(gh.pr_number)],
        )


def _base_env_with_event(tmp_path: Path) -> dict[str, str]:
    event_path = tmp_path / "event.json"
    event = {"action": "opened", "pull_request": {"number": 77}}
    event_path.write_text(json.dumps(event), encoding="utf-8")
    return {
        # Required inputs
        "GERRIT_KNOWN_HOSTS": "example.com ssh-rsa AAAAB3Nza...",
        "GERRIT_SSH_PRIVKEY_G2G": "-----BEGIN KEY-----\nabc\n-----END KEY-----",
        "GERRIT_SSH_USER_G2G": "gerrit-bot",
        "GERRIT_SSH_USER_G2G_EMAIL": "gerrit-bot@example.org",
        # GitHub event context
        "GITHUB_EVENT_NAME": "pull_request_target",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_REPOSITORY": "example/repo",
        "GITHUB_REPOSITORY_OWNER": "example",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "12345",
        "GITHUB_SHA": "deadbeef",
        "GITHUB_BASE_REF": "master",
        "GITHUB_HEAD_REF": "feature",
        # Ensure real execution path (not short-circuited)
        "G2G_TEST_MODE": "false",
    }


def test_single_pr_path_writes_outputs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare $GITHUB_OUTPUT file path
    outputs_file = tmp_path / "gh_output.txt"
    env = _base_env_with_event(tmp_path)
    env["GITHUB_OUTPUT"] = str(outputs_file)

    # Patch Orchestrator used by CLI to a dummy that returns a single result
    monkeypatch.setattr(cli_mod, "Orchestrator", _DummyOrchestratorSingle)

    # Invoke the CLI root to use the single-PR path
    result = runner.invoke(app, [], env=env)
    assert result.exit_code == 0, result.stdout + result.stderr

    # Read and validate outputs written to $GITHUB_OUTPUT
    content = outputs_file.read_text(encoding="utf-8").splitlines()
    # Turn into dict-like mapping for easy assertions (last occurrence wins)
    mapping: dict[str, str] = {}
    for line in content:
        if "=" in line:
            k, v = line.split("=", 1)
            mapping[k.strip()] = v

    assert (
        mapping.get("gerrit_change_request_url")
        == "https://gerrit.example.org/c/repo/+/101"
    )
    assert mapping.get("gerrit_change_request_num") == "101"
    # commit sha is written because our dummy orchestrator exported GERRIT_COMMIT_SHA
    assert mapping.get("gerrit_commit_sha") == "deadbeefcafebabe"


def test_multi_pr_url_mode_writes_aggregated_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare $GITHUB_OUTPUT and baseline environment
    outputs_file = tmp_path / "gh_output_multi.txt"
    env = {
        # Required Gerrit inputs for validation
        "GERRIT_KNOWN_HOSTS": "gerrit.example.org ssh-rsa AAAAB3Nza...",
        "GERRIT_SSH_PRIVKEY_G2G": "-----BEGIN KEY-----\nfake\n-----END KEY-----",
        "GERRIT_SSH_USER_G2G": "gerrit-bot",
        "GERRIT_SSH_USER_G2G_EMAIL": "gerrit-bot@example.org",
        # Simulate a local/URL invocation (no GH event file)
        "GITHUB_EVENT_NAME": "",
        "GITHUB_EVENT_PATH": "",
        # Provide a token placeholder (we patch build_client, so it's unused)
        "GITHUB_TOKEN": "dummy",
        "DRY_RUN": "true",
        "G2G_TEST_MODE": "false",
        "GITHUB_OUTPUT": str(outputs_file),
    }

    # Patch Orchestrator to our multi-PR dummy
    monkeypatch.setattr(cli_mod, "Orchestrator", _DummyOrchestratorMulti)

    # Patch the GitHub API helpers used by the bulk path
    class _DummyPR:
        def __init__(self, number: int) -> None:
            self.number = number

    monkeypatch.setattr(cli_mod, "build_client", lambda: object())
    monkeypatch.setattr(cli_mod, "get_repo_from_env", lambda _client: object())
    monkeypatch.setattr(
        cli_mod,
        "iter_open_pulls",
        lambda _repo: iter([_DummyPR(5), _DummyPR(7)]),
    )

    # Provide a repository URL to enter URL mode and bulk path (SYNC_ALL_OPEN_PRS)
    repo_url = "https://github.com/org/repo"
    result = runner.invoke(app, ["--dry-run", repo_url], env=env)
    assert result.exit_code == 0, result.stdout + result.stderr

    # Validate aggregated outputs in $GITHUB_OUTPUT
    content = outputs_file.read_text(encoding="utf-8").splitlines()
    # Parse multi-line GitHub Output format where values can span multiple lines
    mapping: dict[str, str] = {}
    i = 0
    n = len(content)
    while i < n:
        line = content[i]
        # GitHub Actions multiline format: key<<DELIM ... DELIM
        if "<<" in line and "=" not in line:
            parts = line.split("<<", 1)
            key = parts[0].strip()
            delim = parts[1].strip()
            i += 1
            vals: list[str] = []
            while i < n and content[i] != delim:
                vals.append(content[i])
                i += 1
            # Skip closing delimiter line if present
            if i < n and content[i] == delim:
                i += 1
            mapping[key] = "\n".join(vals)
            continue
        # Legacy simple format: key=value (single line)
        if "=" in line:
            k, v = line.split("=", 1)
            mapping[k.strip()] = v
        i += 1

    # The CLI aggregates newline-separated values for multiple PRs
    # Ensure both PR 5 and 7 are present in the respective outputs
    assert "gerrit_change_request_url" in mapping
    assert "gerrit_change_request_num" in mapping

    urls = mapping["gerrit_change_request_url"].split("\n")
    nums = mapping["gerrit_change_request_num"].split("\n")

    # Order should match iteration: PRs [5, 7]
    assert urls == [
        "https://gerrit.example.org/c/repo/+/5",
        "https://gerrit.example.org/c/repo/+/7",
    ]
    assert nums == ["5", "7"]

    # No commit SHA is expected in multi-PR path unless set elsewhere
    # The CLI only writes gerrit_commit_sha if the environment variable exists
    assert (
        "gerrit_commit_sha" not in mapping or mapping["gerrit_commit_sha"] == ""
    )
