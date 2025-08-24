# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github2gerrit_python.core import GerritInfo
from github2gerrit_python.core import Orchestrator
from github2gerrit_python.core import OrchestratorError
from github2gerrit_python.core import RepoNames
from github2gerrit_python.gitutils import CommandError
from github2gerrit_python.gitutils import CommandResult
from github2gerrit_python.models import Inputs


# -----------------------------
# Helpers
# -----------------------------


def _minimal_inputs() -> Inputs:
    # Provide the minimal, valid Inputs dataclass needed by core methods.
    # Values below are mostly placeholders appropriate for unit tests.
    return Inputs(
        submit_single_commits=False,
        use_pr_as_commit=False,
        fetch_depth=10,
        gerrit_known_hosts="example.org ssh-rsa AAAAB3Nza...",
        gerrit_ssh_privkey_g2g="-----BEGIN KEY-----\nabc\n-----END KEY-----",
        gerrit_ssh_user_g2g="gerrit-bot",
        gerrit_ssh_user_g2g_email="gerrit-bot@example.org",
        organization="example",
        reviewers_email="",
        preserve_github_prs=False,
        dry_run=False,
        gerrit_server="gerrit.example.org",
        gerrit_server_port="29418",
        gerrit_project="example/project",
        issue_id="",
        allow_duplicates=False,
    )


# -----------------------------
# Target branch resolution
# -----------------------------


def test_resolve_target_branch_prefers_env_gerrit_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GERRIT_BRANCH", "release-1.2")
    monkeypatch.setenv(
        "GITHUB_BASE_REF", "main"
    )  # should be ignored due to GERRIT_BRANCH
    orch = Orchestrator(workspace=tmp_path)
    assert orch._resolve_target_branch() == "release-1.2"


def test_resolve_target_branch_falls_back_to_github_base_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GERRIT_BRANCH", raising=False)
    monkeypatch.setenv("GITHUB_BASE_REF", "feature/do-thing")
    orch = Orchestrator(workspace=tmp_path)
    assert orch._resolve_target_branch() == "feature/do-thing"


def test_resolve_target_branch_defaults_to_master(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GERRIT_BRANCH", raising=False)
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    orch = Orchestrator(workspace=tmp_path)
    assert orch._resolve_target_branch() == "master"


# -----------------------------
# Topic naming configurables
# -----------------------------


def test_push_to_gerrit_topic_default_prefix_with_pr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    orch = Orchestrator(workspace=tmp_path)
    repo = RepoNames(
        project_gerrit="releng/builder", project_github="releng-builder"
    )
    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="releng/builder"
    )

    # PR number present, default prefix GH expected: "GH-releng-builder-42"
    monkeypatch.setenv("PR_NUMBER", "42")
    monkeypatch.delenv("G2G_TOPIC_PREFIX", raising=False)

    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        # Capture calls; emulate success
        calls.append(list(cmd))
        # Provide stdout for any consumers; not used in _push_to_gerrit
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("github2gerrit_python.core.run_cmd", fake_run_cmd)

    # Act
    orch._push_to_gerrit(
        gerrit=gerrit,
        repo=repo,
        branch="master",
        reviewers="reviewer@example.org",
        single_commits=False,
    )

    # Assert: the last call should be the git-review push with -t topic including PR number
    assert calls, "Expected at least one run_cmd call"
    last = calls[-1]
    # Structure: ["git", "review", "--yes", "-t", topic, "--reviewers", reviewers]
    assert last[:3] == ["git", "review", "--yes"]
    assert "-t" in last
    t_idx = last.index("-t")
    topic = last[t_idx + 1]
    assert topic == "GH-releng-builder-42"


def test_push_to_gerrit_topic_custom_prefix_without_pr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    orch = Orchestrator(workspace=tmp_path)
    repo = RepoNames(
        project_gerrit="releng/builder", project_github="releng-builder"
    )
    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="releng/builder"
    )

    # No PR number, custom prefix should be used and omit PR suffix
    monkeypatch.delenv("PR_NUMBER", raising=False)
    monkeypatch.setenv("G2G_TOPIC_PREFIX", "MY")

    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        calls.append(list(cmd))
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("github2gerrit_python.core.run_cmd", fake_run_cmd)

    # Act
    orch._push_to_gerrit(
        gerrit=gerrit,
        repo=repo,
        branch="master",
        reviewers="reviewer@example.org",
        single_commits=True,  # also exercises the tmp_branch checkout path
    )

    # Assert
    assert calls, "Expected at least one run_cmd call"
    last = calls[-1]
    assert last[:3] == ["git", "review", "--yes"]
    t_idx = last.index("-t")
    topic = last[t_idx + 1]
    assert topic == "MY-releng-builder"


# -----------------------------
# Error handling for git-review
# -----------------------------


def test_push_to_gerrit_raises_on_git_review_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orch = Orchestrator(workspace=tmp_path)
    repo = RepoNames(
        project_gerrit="releng/builder", project_github="releng-builder"
    )
    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="releng/builder"
    )

    # Ensure PR_NUMBER is present to form topic, though failure happens at push
    monkeypatch.setenv("PR_NUMBER", "7")

    def fake_run_cmd_fail(cmd: list[str], **kwargs: Any) -> CommandResult:
        # Fail only on the git review push; allow any prior calls if present
        if len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "review":
            raise CommandError(
                "error",
                cmd=cmd,
                returncode=1,
                stdout="",
                stderr="boom",
            )
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("github2gerrit_python.core.run_cmd", fake_run_cmd_fail)

    with pytest.raises(OrchestratorError) as ei:
        orch._push_to_gerrit(
            gerrit=gerrit,
            repo=repo,
            branch="master",
            reviewers="reviewer@example.org",
            single_commits=False,
        )
    assert "Failed to push changes to Gerrit with git-review" in str(ei.value)


def test_configure_git_raises_on_git_review_init_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orch = Orchestrator(workspace=tmp_path)
    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="releng/builder"
    )
    inputs = _minimal_inputs()

    # No-op git_config to avoid touching real config
    monkeypatch.setattr(
        "github2gerrit_python.core.git_config", lambda *a, **k: None
    )

    def fake_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        # Provide a fake repo root for rev-parse
        if cmd[:2] == ["git", "rev-parse"]:
            return CommandResult(
                returncode=0, stdout=str(tmp_path) + "\n", stderr=""
            )
        # Fail the git-review initialization
        if cmd[:2] == ["git", "review"] and "-s" in cmd and "-v" in cmd:
            raise CommandError(
                "error",
                cmd=cmd,
                returncode=1,
                stdout="",
                stderr="init boom",
            )
        # Default success
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("github2gerrit_python.core.run_cmd", fake_run_cmd)

    with pytest.raises(OrchestratorError) as ei:
        orch._configure_git(gerrit, inputs)
    assert "Failed to initialize git-review" in str(ei.value)


# -----------------------------
# Environment cleanup
# -----------------------------


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _clear_env_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure environment-sensitive code paths are deterministic across tests.
    for var in (
        "GERRIT_BRANCH",
        "GITHUB_BASE_REF",
        "PR_NUMBER",
        "G2G_TOPIC_PREFIX",
    ):
        monkeypatch.delenv(var, raising=False)
