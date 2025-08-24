# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import json
from pathlib import Path

import pytest

from github2gerrit.cli import _extract_pr_number
from github2gerrit.cli import _mask_secret
from github2gerrit.cli import _read_github_context


# -----------------------------
# Tests for _mask_secret
# -----------------------------


def test_mask_secret_empty_returns_empty() -> None:
    assert _mask_secret("") == ""


def test_mask_secret_shorter_than_keep_masks_all() -> None:
    # default keep=4; len(secret)=3 -> mask all 3 characters
    assert _mask_secret("abc") == "***"


def test_mask_secret_equal_to_keep_masks_all() -> None:
    # When len(value) <= keep, the implementation masks the whole string
    assert _mask_secret("abcd", keep=4) == "****"


def test_mask_secret_longer_than_keep_keeps_prefix_and_masks_rest() -> None:
    # For longer values, first 'keep' chars are kept, rest are masked
    assert _mask_secret("abcdefgh") == "abcd****"
    assert _mask_secret("abcdefgh", keep=2) == "ab******"


# -----------------------------
# Tests for _extract_pr_number
# -----------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "evt, expected",
    [
        ({"pull_request": {"number": 17}}, 17),
        ({"pull_request": {"number": 0}}, 0),
        ({"issue": {"number": 42}}, 42),
        ({"number": 5}, 5),
        # Non-integer values should be ignored and produce None
        ({"pull_request": {"number": "x"}}, None),
        ({"issue": {"number": "y"}}, None),
        ({"number": "z"}, None),
        ({}, None),
    ],
)
def test_extract_pr_number(
    evt: dict[str, object], expected: int | None
) -> None:
    assert _extract_pr_number(evt) == expected


# -----------------------------
# Tests for _read_github_context
# -----------------------------


def test_read_github_context_reads_event_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare a fake GitHub event payload with action and pull_request.number
    event = {
        "action": "opened",
        "pull_request": {"number": 33},
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    # Set environment variables consumed by _read_github_context
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "example")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.enterprise.local")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature-branch")
    # Ensure PR_NUMBER fallback is not used for this test
    monkeypatch.delenv("PR_NUMBER", raising=False)

    ctx = _read_github_context()

    assert ctx.event_name == "pull_request"
    assert ctx.event_action == "opened"
    assert ctx.event_path == event_path
    assert ctx.repository == "example/repo"
    assert ctx.repository_owner == "example"
    assert ctx.server_url == "https://github.enterprise.local"
    assert ctx.run_id == "12345"
    assert ctx.sha == "deadbeef"
    assert ctx.base_ref == "main"
    assert ctx.head_ref == "feature-branch"
    assert ctx.pr_number == 33


def test_read_github_context_falls_back_to_PR_NUMBER_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Event without PR info
    event = {"action": "synchronize"}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("PR_NUMBER", "6")  # fallback
    # Keep other env minimal
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "owner")
    monkeypatch.delenv("GITHUB_SERVER_URL", raising=False)  # use default

    ctx = _read_github_context()

    # Action should be captured from event payload
    assert ctx.event_action == "synchronize"
    # PR number should fall back to env var
    assert ctx.pr_number == 6
    # Default server URL should be used when not set
    assert ctx.server_url == "https://github.com"


def test_read_github_context_handles_missing_event_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point to a non-existent path and minimal env
    monkeypatch.setenv("GITHUB_EVENT_NAME", "")
    monkeypatch.setenv("GITHUB_EVENT_PATH", "/nonexistent/path/to/event.json")
    monkeypatch.delenv("PR_NUMBER", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY_OWNER", raising=False)
    monkeypatch.delenv("GITHUB_SERVER_URL", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)

    ctx = _read_github_context()

    # With no event and no env, values should be empty/defaults
    assert ctx.event_name == ""
    assert ctx.event_action == ""
    assert (
        ctx.event_path is not None
    )  # Path is created from the string env var even if it doesn't exist
    assert str(ctx.event_path) == "/nonexistent/path/to/event.json"
    assert ctx.repository == ""
    assert ctx.repository_owner == ""
    assert ctx.server_url == "https://github.com"
    assert ctx.run_id == ""
    assert ctx.sha == ""
    assert ctx.base_ref == ""
    assert ctx.head_ref == ""
    assert ctx.pr_number is None
