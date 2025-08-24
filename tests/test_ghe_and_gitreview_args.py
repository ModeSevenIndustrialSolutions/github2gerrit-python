# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github2gerrit.cli import _parse_github_target
from github2gerrit.core import GerritInfo
from github2gerrit.core import Orchestrator
from github2gerrit.core import RepoNames


def test_ghe_url_parsing_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that:
      - By default, GHE (non-github.com) URLs are rejected.
      - When ALLOW_GHE_URLS is set to true, non-github.com URLs are accepted.
      - With ALLOW_GHE_URLS=false, github.com URLs are still accepted.
    """
    ghe_url = "https://ghe.example.org/org/repo/pull/123"
    gh_url = "https://github.com/org/repo/pull/456"

    # Default: reject GHE (env unset -> default False)
    monkeypatch.delenv("ALLOW_GHE_URLS", raising=False)
    assert _parse_github_target(ghe_url) == (None, None, None)

    # Enable GHE: accept non-github.com hosts
    monkeypatch.setenv("ALLOW_GHE_URLS", "true")
    assert _parse_github_target(ghe_url) == ("org", "repo", 123)

    # With ALLOW_GHE_URLS=false, standard github.com URL still parses
    monkeypatch.setenv("ALLOW_GHE_URLS", "false")
    assert _parse_github_target(gh_url) == ("org", "repo", 456)


def test_git_review_args_include_branch_and_repeated_reviewer_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    _push_to_gerrit should:
      - include branch as positional argument at the end
      - expand comma-separated reviewers into repeated '--reviewer' flags
      - include a topic derived from env and repo name (PR_NUMBER and G2G_TOPIC_PREFIX)
    """
    orch = Orchestrator(workspace=tmp_path)

    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="platform/infra"
    )
    repo = RepoNames(project_gerrit="platform/infra", project_github="my-repo")
    branch = "main"
    reviewers = "alice@example.org, bob@example.org"

    # Control topic formation via environment
    monkeypatch.setenv("PR_NUMBER", "42")
    monkeypatch.setenv("G2G_TOPIC_PREFIX", "GH")

    recorded_cmds: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], **kwargs: Any) -> None:
        # Record all commands; we care about the 'git review' invocation
        recorded_cmds.append(list(cmd))

    monkeypatch.setattr(
        "github2gerrit.core.run_cmd", fake_run_cmd, raising=True
    )

    # Act (single_commits=False to avoid extra checkout step)
    orch._push_to_gerrit(
        gerrit=gerrit,
        repo=repo,
        branch=branch,
        reviewers=reviewers,
        single_commits=False,
    )

    # Find the git-review invocation
    review_calls = [
        c
        for c in recorded_cmds
        if len(c) >= 2 and c[0] == "git" and c[1] == "review"
    ]
    assert review_calls, (
        f"No git-review invocation captured in: {recorded_cmds!r}"
    )
    args = review_calls[-1]

    # Basic structure
    assert args[0:2] == ["git", "review"]

    # Topic flag present with expected composition
    assert "-t" in args
    t_idx = args.index("-t")
    assert args[t_idx + 1] == "GH-my-repo-42"

    # Branch as positional argument at the end
    assert args[-1] == branch

    # Reviewers are passed via repeated --reviewer flags; no '--reviewers' aggregate flag
    assert "--reviewers" not in args
    # Collect reviewer values in order
    rev_vals: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--reviewer" and i + 1 < len(args):
            rev_vals.append(args[i + 1])
            i += 2
            continue
        i += 1
    assert rev_vals == ["alice@example.org", "bob@example.org"]
