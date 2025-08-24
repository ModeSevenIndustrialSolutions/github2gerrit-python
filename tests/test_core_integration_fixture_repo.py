# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from github2gerrit.core import GerritInfo
from github2gerrit.core import Orchestrator
from github2gerrit.core import RepoNames
from github2gerrit.models import GitHubContext
from github2gerrit.models import Inputs


sys.path.append(str(Path(__file__).parent))
from fixtures.make_repo import init_repo
from fixtures.make_repo import write_gitreview


def _minimal_inputs(*, dry_run: bool = False) -> Inputs:
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
        dry_run=dry_run,
        gerrit_server="gerrit.example.org",
        gerrit_server_port="29418",
        gerrit_project="example/project",
        issue_id="",
        allow_duplicates=False,
    )


def _gh_ctx(
    *,
    repository: str = "owner/repo-name",
    owner: str = "owner",
    pr_number: int | None = 7,
) -> GitHubContext:
    return GitHubContext(
        event_name="pull_request_target",
        event_action="opened",
        event_path=None,
        repository=repository,
        repository_owner=owner,
        server_url="https://github.com",
        run_id="1",
        sha="deadbeef",
        base_ref="master",
        head_ref="feature/test",
        pr_number=pr_number,
    )


def test_read_gitreview_parses_file(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo", default_branch="main")
    # Write a .gitreview with non-default port and project with .git suffix
    p = write_gitreview(
        repo,
        host="gerrit.acme.org",
        port=29419,
        project="acme/widgets",
    )
    assert p.exists()
    orch = Orchestrator(workspace=repo.path)
    info = orch._read_gitreview(repo.path / ".gitreview")
    assert info is not None
    assert info.host == "gerrit.acme.org"
    assert info.port == 29419
    # The .git suffix should be removed by the reader
    assert info.project == "acme/widgets"


def test_derive_repo_names_from_gitreview(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo2", default_branch="main")
    write_gitreview(
        repo,
        host="gerrit.example.org",
        port=29418,
        project="releng/builder",
    )
    orch = Orchestrator(workspace=repo.path)
    gitreview = orch._read_gitreview(repo.path / ".gitreview")
    assert gitreview is not None
    names = orch._derive_repo_names(gitreview, _gh_ctx())
    assert names.project_gerrit == "releng/builder"
    # GitHub project name should be Gerrit path with '/' replaced by '-'
    assert names.project_github == "releng-builder"


def test_derive_repo_names_from_context_fallback(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo3", default_branch="main")
    orch = Orchestrator(workspace=repo.path)
    # No .gitreview present; derive from GitHub repository owner/name
    gh = _gh_ctx(repository="acme/my-repo-name", owner="acme")
    names = orch._derive_repo_names(None, gh)
    # Fallback maps '-' to '/' for Gerrit path
    assert names.project_gerrit == "my/repo/name"
    assert names.project_github == "my-repo-name"


def test_resolve_gerrit_info_prefers_gitreview(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo4", default_branch="main")
    write_gitreview(
        repo,
        host="gerrit.example.net",
        port=29420,
        project="apps/service",
    )
    orch = Orchestrator(workspace=repo.path)
    gitreview = orch._read_gitreview(repo.path / ".gitreview")
    assert gitreview is not None
    gh = _gh_ctx(repository="org/service-repo", owner="org")
    names = orch._derive_repo_names(gitreview, gh)
    info = orch._resolve_gerrit_info(gitreview, _minimal_inputs(), names)
    # Should return the gitreview values directly
    assert info.host == "gerrit.example.net"
    assert info.port == 29420
    assert info.project == "apps/service"


def test_resolve_gerrit_info_dry_run_uses_derived_project_when_missing(
    tmp_path: Path,
) -> None:
    repo = init_repo(tmp_path / "repo5", default_branch="main")
    orch = Orchestrator(workspace=repo.path)
    gh = _gh_ctx(repository="team/reusable-action", owner="team")
    names = orch._derive_repo_names(None, gh)
    # Provide inputs with missing project but dry-run True to allow derivation
    inputs = _minimal_inputs(dry_run=True)
    inputs = Inputs(
        submit_single_commits=inputs.submit_single_commits,
        use_pr_as_commit=inputs.use_pr_as_commit,
        fetch_depth=inputs.fetch_depth,
        gerrit_known_hosts=inputs.gerrit_known_hosts,
        gerrit_ssh_privkey_g2g=inputs.gerrit_ssh_privkey_g2g,
        gerrit_ssh_user_g2g=inputs.gerrit_ssh_user_g2g,
        gerrit_ssh_user_g2g_email=inputs.gerrit_ssh_user_g2g_email,
        organization=inputs.organization,
        reviewers_email=inputs.reviewers_email,
        preserve_github_prs=inputs.preserve_github_prs,
        dry_run=True,
        gerrit_server=inputs.gerrit_server,
        gerrit_server_port=inputs.gerrit_server_port,
        gerrit_project="",  # Missing, should derive from repo name
        issue_id=inputs.issue_id,
        allow_duplicates=inputs.allow_duplicates,
    )
    info = orch._resolve_gerrit_info(None, inputs, names)
    assert info.host == "gerrit.example.org"
    assert info.port == 29418
    # Project should be derived from RepoNames.project_gerrit
    assert info.project == names.project_gerrit


def test_dry_run_preflight_network_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Initialize a repo but do not require any network or remote
    repo = init_repo(tmp_path / "repo6", default_branch="main")
    orch = Orchestrator(workspace=repo.path)
    gh = _gh_ctx(repository="acme/system", owner="acme", pr_number=33)
    names = RepoNames(project_gerrit="acme/system", project_github="system")
    # Disable network within preflight to avoid DNS/HTTP/SSH probes
    monkeypatch.setenv("G2G_DRYRUN_DISABLE_NETWORK", "true")
    # Keep config path benign and prevent close/comment behavior from looking at network
    monkeypatch.setenv("PRESERVE_GITHUB_PRS", "true")
    inputs = _minimal_inputs(dry_run=True)
    gerrit = GerritInfo(
        host="gerrit.acme.org", port=29418, project="acme/system"
    )
    # Should complete without raising exceptions
    orch._dry_run_preflight(gerrit=gerrit, inputs=inputs, gh=gh, repo=names)
