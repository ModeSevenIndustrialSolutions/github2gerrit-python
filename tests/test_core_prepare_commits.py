# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from github2gerrit.core import Orchestrator
from github2gerrit.gitutils import CommandResult
from github2gerrit.models import GitHubContext
from github2gerrit.models import Inputs


def _minimal_inputs() -> Inputs:
    # Provide a minimal but valid Inputs structure for the orchestrator methods.
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


def _gh_ctx(
    *,
    pr_number: int | None = 7,
    event_name: str = "pull_request_target",
    event_action: str = "opened",
) -> GitHubContext:
    # Use master as the base ref per Gerrit defaults
    return GitHubContext(
        event_name=event_name,
        event_action=event_action,
        event_path=None,
        repository="owner/repo",
        repository_owner="owner",
        server_url="https://github.com",
        run_id="1",
        sha="deadbeef",
        base_ref="master",
        head_ref="feature/test",
        pr_number=pr_number,
    )


def test_prepare_single_commits_collects_unique_change_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    _prepare_single_commits should:
      - compute a commit list from origin/<branch>..HEAD
      - cherry-pick each commit onto a temp branch
      - collect Change-Id trailers from the last commit
      - de-duplicate while preserving order
    """

    orch = Orchestrator(workspace=tmp_path)
    inputs = _minimal_inputs()
    gh = _gh_ctx(pr_number=33)

    # Simulate three commits, Change-Ids (with a duplicate)
    commits: list[str] = ["c1", "c2", "c3"]
    change_ids_for_pick: dict[str, list[str]] = {
        "c1": ["I111111111111111111111111111111111111111"],
        "c2": ["I222222222222222222222222222222222222222"],
        "c3": ["I111111111111111111111111111111111111111"],  # duplicate
    }

    def fake_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        # Fetch origin branch (no-op)
        if cmd[:3] == ["git", "fetch", "origin"]:
            return CommandResult(returncode=0, stdout="", stderr="")
        # rev-list origin/<branch>..HEAD
        if cmd[:3] == ["git", "rev-list", "--reverse"]:
            out = "\n".join(commits) + "\n"
            return CommandResult(returncode=0, stdout=out, stderr="")
        # rev-parse origin/<branch> or HEAD/base
        if cmd[:2] == ["git", "rev-parse"]:
            # Return a dummy sha
            return CommandResult(returncode=0, stdout="abcde12345\n", stderr="")
        # checkout (both branch creation and switching)
        if cmd[:2] == ["git", "checkout"]:
            return CommandResult(returncode=0, stdout="", stderr="")
        # author retrieval via git show %an <%ae>
        if (
            cmd[:3] == ["git", "show", "-s"]
            and "--pretty=format:%an <%ae>" in cmd
        ):
            return CommandResult(
                returncode=0, stdout="Author <author@example.org>\n", stderr=""
            )
        # final log display (no-op)
        if cmd[:3] == ["git", "log", "-n3"]:
            return CommandResult(returncode=0, stdout="", stderr="")
        # Default: allow unknowns to pass
        return CommandResult(returncode=0, stdout="", stderr="")

    # Patch core module symbols that are imported into core's namespace.
    monkeypatch.setattr("github2gerrit.core.run_cmd", fake_run_cmd)

    def fake_git_cherry_pick(csha: str, **kwargs: Any) -> None:
        # Success path; behavior validated through trailers below.
        assert csha in commits

    monkeypatch.setattr(
        "github2gerrit.core.git_cherry_pick", fake_git_cherry_pick
    )

    def fake_git_commit_amend(**kwargs: Any) -> None:
        # Called after cherry-pick to ensure trailers exist; we can no-op here.
        pass

    monkeypatch.setattr(
        "github2gerrit.core.git_commit_amend", fake_git_commit_amend
    )

    # Use a mutable holder to return appropriate trailers for each commit pick
    picked_index = {"i": -1}

    def fake_trailers(
        keys: list[str] | None = None, **kwargs: Any
    ) -> dict[str, list[str]]:
        # After each pick + amend, core queries trailers; emulate next commit's trailers
        picked_index["i"] += 1
        idx = picked_index["i"]
        if 0 <= idx < len(commits):
            csha = commits[idx]
            return {"Change-Id": change_ids_for_pick.get(csha, [])}
        return {"Change-Id": []}

    monkeypatch.setattr(
        "github2gerrit.core.git_last_commit_trailers", fake_trailers
    )

    # Act
    from github2gerrit.core import GerritInfo

    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="example/project"
    )
    result = orch._prepare_single_commits(inputs, gh, gerrit)

    # Assert: order preserved, duplicates removed
    assert result.change_ids == [
        "I111111111111111111111111111111111111111",
        "I222222222222222222222222222222222222222",
    ]
    # Commit SHAs are collected later; not part of prepare phase
    assert result.commit_shas == []


def test_prepare_squashed_commit_reuses_change_id_from_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    _prepare_squashed_commit should:
      - build a squashed commit message from base..head log
      - reuse last Change-Id found in recent PR comments when appropriate
      - create a new commit and ensure trailers contain the reused Change-Id
    """

    orch = Orchestrator(workspace=tmp_path)
    inputs = _minimal_inputs()
    gh = _gh_ctx(
        pr_number=55,
        event_name="pull_request_target",
        event_action="synchronize",
    )

    # Prepare a reused Change-Id sourced from PR comments
    reused_cid = "Ireusedfromcomments000000000000000000000"

    def fake_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        # fetch origin/<branch>
        if cmd[:3] == ["git", "fetch", "origin"]:
            return CommandResult(returncode=0, stdout="", stderr="")
        # rev-parse origin/<branch> -> base sha
        if cmd[:2] == ["git", "rev-parse"]:
            ref = cmd[2] if len(cmd) > 2 else ""
            if ref.startswith("origin/"):
                return CommandResult(
                    returncode=0,
                    stdout="basesha00000000000000000000000000000000\n",
                    stderr="",
                )
            if ref == "HEAD":
                return CommandResult(
                    returncode=0,
                    stdout="headsha99999999999999999999999999999999\n",
                    stderr="",
                )
            return CommandResult(returncode=0, stdout="deadbeef\n", stderr="")
        # checkout and branch creation
        if cmd[:2] == ["git", "checkout"]:
            return CommandResult(returncode=0, stdout="", stderr="")
        # merge --squash
        if cmd[:3] == ["git", "merge", "--squash"]:
            return CommandResult(returncode=0, stdout="", stderr="")
        # git log body between base..head (used to compose message)
        if cmd[:3] == ["git", "log", "-v"] and "--format=%B" in cmd:
            body = (
                "Feature: add things\n\n"
                "Signed-off-by: Dev One <dev1@example.org>\n"
                "Change-Id: Ioldoldoldoldoldoldoldoldoldoldoldoldold\n"
                "Signed-off-by: Dev Two <dev2@example.org>\n"
            )
            return CommandResult(returncode=0, stdout=body, stderr="")
        # author extraction for head commit
        if (
            cmd[:3] == ["git", "show", "-s"]
            and "--pretty=format:%an <%ae>" in cmd
        ):
            return CommandResult(
                returncode=0, stdout="Lead Dev <lead@example.org>\n", stderr=""
            )
        return CommandResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("github2gerrit.core.run_cmd", fake_run_cmd)

    # GitHub API helpers used to fetch PR comments
    monkeypatch.setattr("github2gerrit.core.build_client", lambda: object())
    monkeypatch.setattr(
        "github2gerrit.core.get_repo_from_env", lambda _c: object()
    )
    monkeypatch.setattr(
        "github2gerrit.core.get_pull", lambda _repo, _num: object()
    )
    monkeypatch.setattr(
        "github2gerrit.core.get_recent_change_ids_from_comments",
        lambda _pr, max_comments=50: ["Ix", reused_cid],
    )

    # git commit creation and amendment
    created = {"new": 0, "amend": 0}

    def fake_git_commit_new(**kwargs: Any) -> None:
        created["new"] += 1

    def fake_git_commit_amend(**kwargs: Any) -> None:
        created["amend"] += 1

    monkeypatch.setattr(
        "github2gerrit.core.git_commit_new", fake_git_commit_new
    )
    monkeypatch.setattr(
        "github2gerrit.core.git_commit_amend", fake_git_commit_amend
    )

    # After commit, trailers should reflect the reused Change-Id
    def fake_trailers(
        keys: list[str] | None = None, **kwargs: Any
    ) -> dict[str, list[str]]:
        return {"Change-Id": [reused_cid]}

    monkeypatch.setattr(
        "github2gerrit.core.git_last_commit_trailers", fake_trailers
    )

    # Act
    from github2gerrit.core import GerritInfo

    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="example/project"
    )
    result = orch._prepare_squashed_commit(inputs, gh, gerrit)

    # Assert: the reused Change-Id should be returned
    assert result.change_ids == [reused_cid]
    # No commit SHAs are determined at prepare time
    assert result.commit_shas == []
    # Should have created exactly one new commit; amend may or may not happen depending on presence
    assert created["new"] == 1
