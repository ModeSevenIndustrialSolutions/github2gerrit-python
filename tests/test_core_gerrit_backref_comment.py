# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github2gerrit.core import GerritInfo
from github2gerrit.core import Orchestrator
from github2gerrit.core import RepoNames
from github2gerrit.models import GitHubContext


class _CallRecorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> None:
        # Record the command vector passed to ssh
        self.calls.append(list(cmd))


def _gh_ctx(
    *,
    repository: str = "acme/widget",
    pr_number: int = 42,
    server_url: str = "https://github.com",
    run_id: str = "12345",
) -> GitHubContext:
    return GitHubContext(
        event_name="pull_request_target",
        event_action="opened",
        event_path=None,
        repository=repository,
        repository_owner=repository.split("/")[0],
        server_url=server_url,
        run_id=run_id,
        sha="deadbeef",
        base_ref="master",
        head_ref="feature/branch",
        pr_number=pr_number,
    )


def test_add_backref_comment_invokes_ssh_with_expected_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: environment and orchestrator
    monkeypatch.setenv("GERRIT_SSH_USER_G2G", "gerrit-bot")
    orch = Orchestrator(workspace=tmp_path)

    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="releng/builder"
    )
    repo = RepoNames(
        project_gerrit="releng/builder", project_github="releng-builder"
    )
    branch = "master"
    gh = _gh_ctx(
        repository="acme/widget",
        pr_number=12,
        server_url="https://github.enterprise",
        run_id="99",
    )

    # Capture calls to run_cmd inside core._add_backref_comment_in_gerrit
    recorder = _CallRecorder()
    monkeypatch.setattr("github2gerrit.core.run_cmd", recorder)

    # Two commit SHAs should result in two ssh review invocations
    shas = ["abc123def456", "feedbead0001"]

    # Act
    orch._add_backref_comment_in_gerrit(
        gerrit=gerrit,
        repo=repo,
        branch=branch,
        commit_shas=shas,
        gh=gh,
    )

    # Assert
    assert len(recorder.calls) == 2

    # Expected message content (single argument with escaped newlines)
    expected_message = (
        "GHPR: https://github.enterprise/acme/widget/pull/12 | "
        "Action-Run: https://github.enterprise/acme/widget/actions/runs/99"
    )

    # Verify the structure of each ssh call
    for idx, sha in enumerate(shas):
        cmd = recorder.calls[idx]
        # Basic skeleton and fixed positions
        assert cmd[0:3] == ["ssh", "-n", "-p"]
        assert cmd[3] == str(gerrit.port)
        assert cmd[4] == f"gerrit-bot@{gerrit.host}"
        # Remaining program and options
        # ["gerrit", "review", "-m", message, "--branch", branch, "--project", repo.project_gerrit, sha]
        assert cmd[5:8] == ["gerrit", "review", "-m"]
        assert cmd[8] == expected_message
        # Branch and project flags
        assert cmd[9:11] == ["--branch", branch]
        assert cmd[11:13] == ["--project", repo.project_gerrit]
        # Final argument is the target commit SHA
        assert cmd[-1] == sha


def test_add_backref_comment_no_shas_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("GERRIT_SSH_USER_G2G", "gerrit-bot")
    orch = Orchestrator(workspace=tmp_path)

    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="platform/infra"
    )
    repo = RepoNames(
        project_gerrit="platform/infra", project_github="platform-infra"
    )
    gh = _gh_ctx()

    recorder = _CallRecorder()
    monkeypatch.setattr("github2gerrit.core.run_cmd", recorder)

    # Act: empty commit shas should result in no ssh calls
    orch._add_backref_comment_in_gerrit(
        gerrit=gerrit,
        repo=repo,
        branch="master",
        commit_shas=[],
        gh=gh,
    )

    # Assert
    assert recorder.calls == []
