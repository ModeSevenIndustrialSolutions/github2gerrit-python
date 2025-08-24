# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from typing import Any

import pytest

from github2gerrit import github_api as ghapi


def test_build_client_uses_env_token_and_returns_dummy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange a dummy Github class to capture init args without making network calls
    captured: dict[str, object] = {}

    class DummyGithub:
        def __init__(self, **kwargs: Any) -> None:
            # Extract the key parameters we care about for testing
            captured["token"] = kwargs.get("login_or_token")
            captured["per_page"] = kwargs.get("per_page", 100)
            # Handle auth object if present (newer PyGithub API)
            if "auth" in kwargs and hasattr(kwargs["auth"], "token"):
                captured["token"] = kwargs["auth"].token

    monkeypatch.setenv("GITHUB_TOKEN", "env-token-123")
    monkeypatch.setattr(ghapi, "Github", DummyGithub, raising=True)

    # Act
    client = ghapi.build_client()

    # Assert
    assert isinstance(client, DummyGithub)
    assert captured["token"] == "env-token-123"
    assert captured["per_page"] == 100


def test_get_repo_from_env_calls_client_get_repo_and_returns_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    calls: dict[str, object] = {}

    class DummyRepo:
        def get_pull(self, number: int) -> object:
            return object()

        def get_pulls(self, state: str) -> list[object]:
            return []

    class DummyClient:
        def get_repo(self, full: str) -> object:
            calls["arg"] = full
            return DummyRepo()

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/name")

    # Act
    repo = ghapi.get_repo_from_env(DummyClient())

    # Assert
    assert hasattr(repo, "get_pulls")
    assert calls["arg"] == "owner/name"


def test_iter_open_pulls_yields_open_prs() -> None:
    class DummyPR:
        def __init__(self, number: int) -> None:
            self.number = number

    class DummyRepo:
        def get_pull(self, number: int) -> DummyPR:
            return DummyPR(number)

        def get_pulls(self, state: str) -> list[DummyPR]:
            assert state == "open"
            return [DummyPR(5), DummyPR(7)]

    from typing import cast

    repo = DummyRepo()
    numbers = [
        pr.number
        for pr in ghapi.iter_open_pulls(cast(ghapi.GhRepository, repo))
    ]
    assert numbers == [5, 7]


def test_get_pr_title_body_handles_none_values() -> None:
    class DummyPR:
        def __init__(self) -> None:
            self.number = 0
            self.title = None
            self.body = None

        def as_issue(self) -> object:
            return object()

        def edit(self, *, state: str) -> None:
            pass

    from typing import cast

    title, body = ghapi.get_pr_title_body(cast(ghapi.GhPullRequest, DummyPR()))
    assert title == ""
    assert body == ""


def test_get_recent_change_ids_from_comments_extracts_ids_in_order() -> None:
    class DummyComment:
        def __init__(self, body: str) -> None:
            self.body = body

    class DummyIssue:
        def get_comments(self) -> list[DummyComment]:
            # Three comments; we will scan only the last two
            return [
                DummyComment("random text\nChange-Id: I000first\nok"),
                DummyComment("no ids here"),
                DummyComment("Change-Id: I111second\nChange-Id: I222third"),
            ]

    class DummyPR:
        def as_issue(self) -> DummyIssue:
            return DummyIssue()

    # max_comments=2 => only last two comments are scanned, in order
    found = ghapi.get_recent_change_ids_from_comments(DummyPR(), max_comments=2)
    # "no ids here" yields none, final one yields two, preserving order of appearance
    assert found == ["I111second", "I222third"]


def test_create_pr_comment_noop_on_blank_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyIssue:
        def __init__(self) -> None:
            self.created: list[str] = []

        def create_comment(self, body: str) -> None:
            self.created.append(body)

    class DummyPR:
        def __init__(self) -> None:
            self.issue = DummyIssue()
            self.as_issue_called = 0

        def as_issue(self) -> DummyIssue:
            # If body is blank, _get_issue should not be called at all
            self.as_issue_called += 1
            return self.issue

    pr = DummyPR()

    # Act: blank/whitespace should cause early return without calling as_issue
    ghapi.create_pr_comment(pr, "")
    ghapi.create_pr_comment(pr, "   ")

    # Assert
    assert pr.as_issue_called == 0
    assert pr.issue.created == []


def test_create_pr_comment_posts_when_non_blank() -> None:
    class DummyIssue:
        def __init__(self) -> None:
            self.created: list[str] = []

        def create_comment(self, body: str) -> None:
            self.created.append(body)

    class DummyPR:
        def __init__(self) -> None:
            self.issue = DummyIssue()

        def as_issue(self) -> DummyIssue:
            return self.issue

    pr = DummyPR()
    ghapi.create_pr_comment(pr, "Hello, world!")
    assert pr.issue.created == ["Hello, world!"]


def test_close_pr_posts_comment_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyIssue:
        def __init__(self) -> None:
            self.created: list[str] = []

        def create_comment(self, body: str) -> None:
            self.created.append(body)

    class DummyPR:
        def __init__(self, number: int) -> None:
            self.number = number
            self.closed_state: str | None = None
            self.issue = DummyIssue()

        def as_issue(self) -> DummyIssue:
            return self.issue

        def edit(self, *, state: str) -> None:
            self.closed_state = state

    pr = DummyPR(42)

    # Act
    ghapi.close_pr(pr, comment="Closing via automation")

    # Assert comment was posted and PR closed
    assert pr.issue.created == ["Closing via automation"]
    assert pr.closed_state == "closed"


def test_build_client_raises_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure no token is set so build_client raises before any network use
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ValueError):
        ghapi.build_client()


def test_get_repo_from_env_raises_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure the environment variable is missing so no client call is made
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)

    class DummyClient:
        def get_repo(self, full: str) -> object:
            raise AssertionError("error")

    with pytest.raises(ValueError):
        ghapi.get_repo_from_env(DummyClient())


def test_close_pr_without_comment_still_closes() -> None:
    class DummyPR:
        def __init__(self) -> None:
            self.closed_state: str | None = None

        def edit(self, *, state: str) -> None:
            self.closed_state = state

    pr = DummyPR()
    ghapi.close_pr(pr, comment=None)
    assert pr.closed_state == "closed"
