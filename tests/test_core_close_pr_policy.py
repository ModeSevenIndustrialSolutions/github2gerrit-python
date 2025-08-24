# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github2gerrit.core import Orchestrator
from github2gerrit.models import GitHubContext


def _gh_ctx(
    *,
    event_name: str,
    pr_number: int | None = 101,
    repository: str = "owner/repo",
    owner: str = "owner",
) -> GitHubContext:
    return GitHubContext(
        event_name=event_name,
        event_action="opened",
        event_path=None,
        repository=repository,
        repository_owner=owner,
        server_url="https://github.com",
        run_id="1",
        sha="deadbeef",
        base_ref="master",
        head_ref="feature/branch",
        pr_number=pr_number,
    )


def test_close_pr_skipped_when_preserve_github_prs_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Enable preservation to bypass closing behavior
    monkeypatch.setenv("PRESERVE_GITHUB_PRS", "true")

    # Any attempt to call GitHub API should fail this test
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "GitHub API should not be called when PRESERVE_GITHUB_PRS is true"
        )

    monkeypatch.setattr("github2gerrit.core.build_client", _fail)
    monkeypatch.setattr("github2gerrit.core.get_repo_from_env", _fail)
    monkeypatch.setattr("github2gerrit.core.get_pull", _fail)
    monkeypatch.setattr("github2gerrit.core.close_pr", _fail)

    orch = Orchestrator(workspace=tmp_path)
    gh = _gh_ctx(event_name="pull_request_target", pr_number=88)

    # Act: should return without invoking any GitHub calls
    orch._close_pull_request_if_required(gh)


def test_close_pr_not_invoked_for_non_target_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure preservation is disabled so only event controls behavior
    monkeypatch.delenv("PRESERVE_GITHUB_PRS", raising=False)

    # Non-target event should not attempt to close
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "GitHub API should not be called for non-target events"
        )

    monkeypatch.setattr("github2gerrit.core.build_client", _fail)
    monkeypatch.setattr("github2gerrit.core.get_repo_from_env", _fail)
    monkeypatch.setattr("github2gerrit.core.get_pull", _fail)
    monkeypatch.setattr("github2gerrit.core.close_pr", _fail)

    orch = Orchestrator(workspace=tmp_path)
    gh = _gh_ctx(event_name="pull_request", pr_number=42)

    # Act: should no-op
    orch._close_pull_request_if_required(gh)


def test_close_pr_invoked_for_pull_request_target_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure preservation is disabled so closing is allowed
    monkeypatch.delenv("PRESERVE_GITHUB_PRS", raising=False)

    calls: dict[str, Any] = {}

    class DummyClient:
        pass

    class DummyRepo:
        pass

    class DummyPR:
        def __init__(self, number: int) -> None:
            self.number = number
            self.closed_state: str | None = None

    # Patch the GitHub helper functions used by the close path
    monkeypatch.setattr(
        "github2gerrit.core.build_client", lambda: DummyClient()
    )
    monkeypatch.setattr(
        "github2gerrit.core.get_repo_from_env", lambda _c: DummyRepo()
    )

    def _get_pull(_repo: DummyRepo, number: int) -> DummyPR:
        calls["get_pull_number"] = number
        return DummyPR(number)

    def _close_pr(pr: DummyPR, *, comment: str | None = None) -> None:
        calls["closed_pr_number"] = pr.number
        calls["comment"] = comment
        pr.closed_state = "closed"

    monkeypatch.setattr("github2gerrit.core.get_pull", _get_pull)
    monkeypatch.setattr("github2gerrit.core.close_pr", _close_pr)

    orch = Orchestrator(workspace=tmp_path)
    gh = _gh_ctx(event_name="pull_request_target", pr_number=123)

    # Act: should invoke close with a comment
    orch._close_pull_request_if_required(gh)

    # Assert
    assert calls.get("get_pull_number") == 123
    assert calls.get("closed_pr_number") == 123
    # Ensure the standard auto-close comment is provided
    assert calls.get("comment") == "Auto-closing pull request"
