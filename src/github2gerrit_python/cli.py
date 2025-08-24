# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from typing import cast
from urllib.parse import urlparse

import typer

from . import models
from .config import apply_config_to_env
from .config import load_org_config
from .core import Orchestrator
from .core import SubmissionResult
from .duplicate_detection import DuplicateChangeError
from .duplicate_detection import check_for_duplicates
from .github_api import build_client
from .github_api import get_pull
from .github_api import get_repo_from_env
from .github_api import iter_open_pulls
from .gitutils import run_cmd
from .models import GitHubContext
from .models import Inputs


def _parse_github_target(url: str) -> tuple[str | None, str | None, int | None]:
    """
    Parse a GitHub repository or pull request URL.

    Returns:
      (org, repo, pr_number) where pr_number may be None for repo URLs.
    """
    try:
        u = urlparse(url)
    except Exception:
        return None, None, None

    allow_ghe = _env_bool("ALLOW_GHE_URLS", False)
    bad_hosts = {
        "gitlab.com",
        "www.gitlab.com",
        "bitbucket.org",
        "www.bitbucket.org",
    }
    if u.netloc in bad_hosts:
        return None, None, None
    if not allow_ghe and u.netloc not in ("github.com", "www.github.com"):
        return None, None, None

    parts = [p for p in (u.path or "").split("/") if p]
    if len(parts) < 2:
        return None, None, None

    owner, repo = parts[0], parts[1]
    pr_number: int | None = None
    if len(parts) >= 4 and parts[2] in ("pull", "pulls"):
        try:
            pr_number = int(parts[3])
        except Exception:
            pr_number = None

    return owner, repo, pr_number


APP_NAME = "github2gerrit"
app: typer.Typer = typer.Typer(add_completion=False, no_args_is_help=False)


def _resolve_org(default_org: str | None) -> str:
    if default_org:
        return default_org
    gh_owner = os.getenv("GITHUB_REPOSITORY_OWNER")
    if gh_owner:
        return gh_owner
    # Fallback to empty string for compatibility with existing action
    return ""


@app.callback(invoke_without_command=True)  # type: ignore[misc]
def main(
    ctx: typer.Context,
    target_url: str | None = typer.Argument(
        None, help="GitHub repository or PR URL"
    ),
    submit_single_commits: bool = typer.Option(
        False,
        "--submit-single-commits",
        help="Submit one commit at a time to the Gerrit repository.",
    ),
    use_pr_as_commit: bool = typer.Option(
        False,
        "--use-pr-as-commit",
        help="Use PR title and body as the commit message.",
    ),
    fetch_depth: int = typer.Option(
        10,
        "--fetch-depth",
        envvar="FETCH_DEPTH",
        help="Fetch-depth for the clone.",
    ),
    gerrit_known_hosts: str = typer.Option(
        "",
        "--gerrit-known-hosts",
        envvar="GERRIT_KNOWN_HOSTS",
        help="Known hosts entries for Gerrit SSH.",
    ),
    gerrit_ssh_privkey_g2g: str = typer.Option(
        "",
        "--gerrit-ssh-privkey-g2g",
        envvar="GERRIT_SSH_PRIVKEY_G2G",
        help="SSH private key for Gerrit (string content).",
    ),
    gerrit_ssh_user_g2g: str = typer.Option(
        "",
        "--gerrit-ssh-user-g2g",
        envvar="GERRIT_SSH_USER_G2G",
        help="Gerrit SSH user.",
    ),
    gerrit_ssh_user_g2g_email: str = typer.Option(
        "",
        "--gerrit-ssh-user-g2g-email",
        envvar="GERRIT_SSH_USER_G2G_EMAIL",
        help="Email address for the Gerrit SSH user.",
    ),
    organization: str | None = typer.Option(
        None,
        "--organization",
        envvar="ORGANIZATION",
        help=("Organization (defaults to GITHUB_REPOSITORY_OWNER when unset)."),
    ),
    reviewers_email: str = typer.Option(
        "",
        "--reviewers-email",
        envvar="REVIEWERS_EMAIL",
        help="Comma-separated list of reviewer emails.",
    ),
    allow_ghe_urls: bool = typer.Option(
        False,
        "--allow-ghe-urls/--no-allow-ghe-urls",
        envvar="ALLOW_GHE_URLS",
        help="Allow non-github.com GitHub Enterprise URLs in direct URL mode.",
    ),
    preserve_github_prs: bool = typer.Option(
        False,
        "--preserve-github-prs",
        envvar="PRESERVE_GITHUB_PRS",
        help="Do not close GitHub PRs after pushing to Gerrit.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        envvar="DRY_RUN",
        help="Validate settings and PR metadata; do not write to Gerrit.",
    ),
    gerrit_server: str = typer.Option(
        "",
        "--gerrit-server",
        envvar="GERRIT_SERVER",
        help="Gerrit server hostname (optional; .gitreview preferred).",
    ),
    gerrit_server_port: str = typer.Option(
        "29418",
        "--gerrit-server-port",
        envvar="GERRIT_SERVER_PORT",
        help="Gerrit SSH port (default: 29418).",
    ),
    gerrit_project: str = typer.Option(
        "",
        "--gerrit-project",
        envvar="GERRIT_PROJECT",
        help="Gerrit project (optional; .gitreview preferred).",
    ),
    issue_id: str = typer.Option(
        "",
        "--issue-id",
        envvar="ISSUE_ID",
        help="Issue ID to include in commit message (e.g., Issue-ID: ABC-123).",
    ),
    allow_duplicates: bool = typer.Option(
        False,
        "--allow-duplicates",
        envvar="ALLOW_DUPLICATES",
        help="Allow submitting duplicate changes without error.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        envvar="G2G_VERBOSE",
        help="Enable verbose debug logging.",
    ),
) -> None:
    """
    Root entrypoint supporting three scenarios:
      1) No arguments (CI): read context and inputs from environment.
      2) Repo URL: process all open PRs for that repository.
      3) PR URL: process the specified PR for that repository.
    """
    # Set up logging level based on verbose flag
    if verbose:
        os.environ["G2G_LOG_LEVEL"] = "DEBUG"
        _reconfigure_logging()
    # Normalize CLI options into environment for unified processing.
    # For boolean flags, only set if explicitly provided via CLI
    if submit_single_commits:
        os.environ["SUBMIT_SINGLE_COMMITS"] = "true"
    if use_pr_as_commit:
        os.environ["USE_PR_AS_COMMIT"] = "true"
    os.environ["FETCH_DEPTH"] = str(fetch_depth)
    if gerrit_known_hosts:
        os.environ["GERRIT_KNOWN_HOSTS"] = gerrit_known_hosts
    if gerrit_ssh_privkey_g2g:
        os.environ["GERRIT_SSH_PRIVKEY_G2G"] = gerrit_ssh_privkey_g2g
    if gerrit_ssh_user_g2g:
        os.environ["GERRIT_SSH_USER_G2G"] = gerrit_ssh_user_g2g
    if gerrit_ssh_user_g2g_email:
        os.environ["GERRIT_SSH_USER_G2G_EMAIL"] = gerrit_ssh_user_g2g_email
    resolved_org = _resolve_org(organization)
    if resolved_org:
        os.environ["ORGANIZATION"] = resolved_org
    if reviewers_email:
        os.environ["REVIEWERS_EMAIL"] = reviewers_email
    if preserve_github_prs:
        os.environ["PRESERVE_GITHUB_PRS"] = "true"
    if dry_run:
        os.environ["DRY_RUN"] = "true"
    os.environ["ALLOW_GHE_URLS"] = "true" if allow_ghe_urls else "false"
    if gerrit_server:
        os.environ["GERRIT_SERVER"] = gerrit_server
    if gerrit_server_port:
        os.environ["GERRIT_SERVER_PORT"] = gerrit_server_port
    if gerrit_project:
        os.environ["GERRIT_PROJECT"] = gerrit_project
    if issue_id:
        os.environ["ISSUE_ID"] = issue_id
    if allow_duplicates:
        os.environ["ALLOW_DUPLICATES"] = "true"
    # URL mode handling
    if target_url:
        org, repo, pr = _parse_github_target(target_url)
        if org:
            os.environ["ORGANIZATION"] = org
        if org and repo:
            os.environ["GITHUB_REPOSITORY"] = f"{org}/{repo}"
        if pr:
            os.environ["PR_NUMBER"] = str(pr)
            os.environ["SYNC_ALL_OPEN_PRS"] = "false"
        else:
            os.environ["SYNC_ALL_OPEN_PRS"] = "true"
        os.environ["G2G_TARGET_URL"] = "1"
    # Delegate to common processing path
    try:
        _process()
    except typer.Exit:
        # Propagate expected exit codes (e.g., validation errors)
        raise
    except Exception as exc:
        log.debug("main(): _process failed: %s", exc)
    return


def _setup_logging() -> logging.Logger:
    level_name = os.getenv("G2G_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = (
        "%(asctime)s %(levelname)-8s %(name)s "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger(APP_NAME)


def _reconfigure_logging() -> None:
    """Reconfigure logging level based on current environment variables."""
    level_name = os.getenv("G2G_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


log = _setup_logging()


def _env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return val if val is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    s = val.strip().lower()
    return s in ("1", "true", "yes", "on")


def _build_inputs_from_env() -> Inputs:
    return Inputs(
        submit_single_commits=_env_bool("SUBMIT_SINGLE_COMMITS", False),
        use_pr_as_commit=_env_bool("USE_PR_AS_COMMIT", False),
        fetch_depth=int(_env_str("FETCH_DEPTH", "10") or "10"),
        gerrit_known_hosts=_env_str("GERRIT_KNOWN_HOSTS"),
        gerrit_ssh_privkey_g2g=_env_str("GERRIT_SSH_PRIVKEY_G2G"),
        gerrit_ssh_user_g2g=_env_str("GERRIT_SSH_USER_G2G"),
        gerrit_ssh_user_g2g_email=_env_str("GERRIT_SSH_USER_G2G_EMAIL"),
        organization=_env_str(
            "ORGANIZATION", _env_str("GITHUB_REPOSITORY_OWNER")
        ),
        reviewers_email=_env_str("REVIEWERS_EMAIL", ""),
        preserve_github_prs=_env_bool("PRESERVE_GITHUB_PRS", False),
        dry_run=_env_bool("DRY_RUN", False),
        gerrit_server=_env_str("GERRIT_SERVER", ""),
        gerrit_server_port=_env_str("GERRIT_SERVER_PORT", "29418"),
        gerrit_project=_env_str("GERRIT_PROJECT"),
        issue_id=_env_str("ISSUE_ID"),
        allow_duplicates=_env_bool("ALLOW_DUPLICATES", False),
    )


def _process_bulk(data: Inputs, gh: GitHubContext) -> None:
    client = build_client()
    repo = get_repo_from_env(client)

    all_urls: list[str] = []
    all_nums: list[str] = []

    prs_list = list(iter_open_pulls(repo))
    log.info("Found %d open PRs to process", len(prs_list))
    for pr in prs_list:
        pr_number = int(getattr(pr, "number", 0) or 0)
        if pr_number <= 0:
            continue

        per_ctx = models.GitHubContext(
            event_name=gh.event_name,
            event_action=gh.event_action,
            event_path=gh.event_path,
            repository=gh.repository,
            repository_owner=gh.repository_owner,
            server_url=gh.server_url,
            run_id=gh.run_id,
            sha=gh.sha,
            base_ref=gh.base_ref,
            head_ref=gh.head_ref,
            pr_number=pr_number,
        )

        log.info("Starting processing of PR #%d", pr_number)
        log.debug(
            "Processing PR #%d in multi-PR mode with event_name=%s, "
            "event_action=%s",
            pr_number,
            gh.event_name,
            gh.event_action,
        )

        try:
            check_for_duplicates(
                per_ctx, allow_duplicates=data.allow_duplicates
            )
        except DuplicateChangeError as exc:
            log.exception("Skipping PR #%d", pr_number)
            typer.echo(f"Skipping PR #{pr_number}: {exc}")
            continue

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir)
                orch = Orchestrator(workspace=workspace)
                result_multi = orch.execute(inputs=data, gh=per_ctx)
                if result_multi.change_urls:
                    all_urls.extend(result_multi.change_urls)
                    for url in result_multi.change_urls:
                        typer.echo(f"Gerrit change URL: {url}")
                        log.info(
                            "PR #%d created Gerrit change: %s",
                            pr_number,
                            url,
                        )
                if result_multi.change_numbers:
                    all_nums.extend(result_multi.change_numbers)
                    log.info(
                        "PR #%d change numbers: %s",
                        pr_number,
                        result_multi.change_numbers,
                    )
        except Exception as exc:
            log.exception("Failed to process PR #%d", pr_number)
            typer.echo(f"Failed to process PR #{pr_number}: {exc}")
            log.info("Continuing to next PR despite failure")
            continue

    if all_urls:
        os.environ["GERRIT_CHANGE_REQUEST_URL"] = "\n".join(all_urls)
    if all_nums:
        os.environ["GERRIT_CHANGE_REQUEST_NUM"] = "\n".join(all_nums)

    _append_github_output(
        {
            "gerrit_change_request_url": os.getenv(
                "GERRIT_CHANGE_REQUEST_URL", ""
            ),
            "gerrit_change_request_num": os.getenv(
                "GERRIT_CHANGE_REQUEST_NUM", ""
            ),
        }
    )

    log.info("Submission pipeline complete (multi-PR).")
    return


def _process_single(data: Inputs, gh: GitHubContext) -> None:
    # Create temporary directory for all git operations
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        try:
            _prepare_local_checkout(workspace, gh, data)
        except Exception as exc:
            log.debug("Local checkout preparation failed: %s", exc)

        orch = Orchestrator(workspace=workspace)
        try:
            result = orch.execute(inputs=data, gh=gh)
        except Exception as exc:
            log.debug("Execution failed; continuing to write outputs: %s", exc)

            result = SubmissionResult(
                change_urls=[], change_numbers=[], commit_shas=[]
            )
        if result.change_urls:
            os.environ["GERRIT_CHANGE_REQUEST_URL"] = "\n".join(
                result.change_urls
            )
            # Output Gerrit change URL(s) to console
            for url in result.change_urls:
                typer.echo(f"Gerrit change URL: {url}")
        if result.change_numbers:
            os.environ["GERRIT_CHANGE_REQUEST_NUM"] = "\n".join(
                result.change_numbers
            )

        # Also write outputs to GITHUB_OUTPUT if available
        _append_github_output(
            {
                "gerrit_change_request_url": os.getenv(
                    "GERRIT_CHANGE_REQUEST_URL", ""
                ),
                "gerrit_change_request_num": os.getenv(
                    "GERRIT_CHANGE_REQUEST_NUM", ""
                ),
                "gerrit_commit_sha": os.getenv("GERRIT_COMMIT_SHA", ""),
            }
        )

        log.info("Submission pipeline complete.")
        return


def _prepare_local_checkout(
    workspace: Path, gh: GitHubContext, data: Inputs
) -> None:
    repo_full = gh.repository.strip() if gh.repository else ""
    server_url = gh.server_url or os.getenv(
        "GITHUB_SERVER_URL", "https://github.com"
    )
    server_url = (server_url or "https://github.com").rstrip("/")
    base_ref = gh.base_ref or ""
    pr_num_str: str = str(gh.pr_number) if gh.pr_number else "0"

    if not repo_full:
        return

    repo_url = f"{server_url}/{repo_full}.git"
    run_cmd(["git", "init"], cwd=workspace)
    run_cmd(["git", "remote", "add", "origin", repo_url], cwd=workspace)

    # Fetch base branch and PR head
    if base_ref:
        try:
            branch_ref = f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}"
            run_cmd(
                [
                    "git",
                    "fetch",
                    f"--depth={data.fetch_depth}",
                    "origin",
                    branch_ref,
                ],
                cwd=workspace,
            )
        except Exception as exc:
            log.debug("Base branch fetch failed for %s: %s", base_ref, exc)

    if pr_num_str:
        pr_ref = (
            f"refs/pull/{pr_num_str}/head:"
            f"refs/remotes/origin/pr/{pr_num_str}/head"
        )
        run_cmd(
            [
                "git",
                "fetch",
                f"--depth={data.fetch_depth}",
                "origin",
                pr_ref,
            ],
            cwd=workspace,
        )
        run_cmd(
            [
                "git",
                "checkout",
                "-B",
                "g2g_pr_head",
                f"refs/remotes/origin/pr/{pr_num_str}/head",
            ],
            cwd=workspace,
        )


def _load_effective_inputs() -> Inputs:
    # Build inputs from environment (used by URL callback path)
    data = _build_inputs_from_env()

    # Load per-org configuration and apply to environment before validation
    org_for_cfg = (
        data.organization
        or os.getenv("ORGANIZATION")
        or os.getenv("GITHUB_REPOSITORY_OWNER")
    )
    cfg = load_org_config(org_for_cfg)
    apply_config_to_env(cfg)

    # Refresh inputs after applying configuration to environment
    data = _build_inputs_from_env()

    # Derive reviewers from local git config if running locally and unset
    if not os.getenv("REVIEWERS_EMAIL") and (
        os.getenv("G2G_TARGET_URL") or not os.getenv("GITHUB_EVENT_NAME")
    ):
        try:
            from .gitutils import enumerate_reviewer_emails

            emails = enumerate_reviewer_emails()
            if emails:
                os.environ["REVIEWERS_EMAIL"] = ",".join(emails)
                data = Inputs(
                    submit_single_commits=data.submit_single_commits,
                    use_pr_as_commit=data.use_pr_as_commit,
                    fetch_depth=data.fetch_depth,
                    gerrit_known_hosts=data.gerrit_known_hosts,
                    gerrit_ssh_privkey_g2g=data.gerrit_ssh_privkey_g2g,
                    gerrit_ssh_user_g2g=data.gerrit_ssh_user_g2g,
                    gerrit_ssh_user_g2g_email=data.gerrit_ssh_user_g2g_email,
                    organization=data.organization,
                    reviewers_email=os.environ["REVIEWERS_EMAIL"],
                    preserve_github_prs=data.preserve_github_prs,
                    dry_run=data.dry_run,
                    gerrit_server=data.gerrit_server,
                    gerrit_server_port=data.gerrit_server_port,
                    gerrit_project=data.gerrit_project,
                    issue_id=data.issue_id,
                    allow_duplicates=data.allow_duplicates,
                )
                log.info("Derived reviewers: %s", data.reviewers_email)
        except Exception as exc:
            log.debug("Could not derive reviewers from git config: %s", exc)

    return data


def _append_github_output(outputs: dict[str, str]) -> None:
    gh_out = os.getenv("GITHUB_OUTPUT")
    if not gh_out:
        return
    try:
        with open(gh_out, "a", encoding="utf-8") as fh:
            for key, val in outputs.items():
                if not val:
                    continue
                if "\n" in val:
                    fh.write(f"{key}<<G2G\n")
                    fh.write(f"{val}\n")
                    fh.write("G2G\n")
                else:
                    fh.write(f"{key}={val}\n")
    except Exception as exc:
        log.debug("Failed to write GITHUB_OUTPUT: %s", exc)


def _augment_pr_refs_if_needed(gh: GitHubContext) -> GitHubContext:
    if (
        os.getenv("G2G_TARGET_URL")
        and gh.pr_number
        and (not gh.head_ref or not gh.base_ref)
    ):
        try:
            client = build_client()
            repo = get_repo_from_env(client)
            pr_obj = get_pull(repo, int(gh.pr_number))
            base_ref = str(
                getattr(getattr(pr_obj, "base", object()), "ref", "") or ""
            )
            head_ref = str(
                getattr(getattr(pr_obj, "head", object()), "ref", "") or ""
            )
            head_sha = str(
                getattr(getattr(pr_obj, "head", object()), "sha", "") or ""
            )
            if base_ref:
                os.environ["GITHUB_BASE_REF"] = base_ref
                log.info("Resolved base_ref via GitHub API: %s", base_ref)
            if head_ref:
                os.environ["GITHUB_HEAD_REF"] = head_ref
                log.info("Resolved head_ref via GitHub API: %s", head_ref)
            if head_sha:
                os.environ["GITHUB_SHA"] = head_sha
                log.info("Resolved head sha via GitHub API: %s", head_sha)
            return _read_github_context()
        except Exception as exc:
            log.debug("Could not resolve PR refs via GitHub API: %s", exc)
    return gh


def _process() -> None:
    data = _load_effective_inputs()

    # Validate inputs
    try:
        _validate_inputs(data)
    except typer.BadParameter as exc:
        log.exception("Validation failed")
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    gh = _read_github_context()
    _log_effective_config(data, gh)

    # Test mode: short-circuit after validation
    if _env_bool("G2G_TEST_MODE", False):
        log.info("Validation complete. Ready to execute submission pipeline.")
        typer.echo("Validation complete. Ready to execute submission pipeline.")
        return

    # Bulk mode for URL/workflow_dispatch
    sync_all = _env_bool("SYNC_ALL_OPEN_PRS", False)
    if sync_all and (
        gh.event_name == "workflow_dispatch" or os.getenv("G2G_TARGET_URL")
    ):
        _process_bulk(data, gh)
        return

    if not gh.pr_number:
        log.error(
            "PR_NUMBER is empty. This tool requires a valid pull request "
            "context. Current event: %s",
            gh.event_name,
        )
        typer.echo(
            "PR_NUMBER is empty. This tool requires a valid pull request "
            f"context. Current event: {gh.event_name}",
            err=True,
        )
        raise typer.Exit(code=2)

    # Test mode handled earlier

    # Execute single-PR submission
    # Augment PR refs via API when in URL mode and token present
    gh = _augment_pr_refs_if_needed(gh)

    # Check for duplicates in single-PR mode (before workspace setup)
    if gh.pr_number and not _env_bool("SYNC_ALL_OPEN_PRS", False):
        try:
            check_for_duplicates(gh, allow_duplicates=data.allow_duplicates)
        except DuplicateChangeError as exc:
            log.exception("Duplicate change detected")
            typer.echo(f"Error: {exc}", err=True)
            typer.echo(
                "Use --allow-duplicates to override this check.", err=True
            )
            raise typer.Exit(code=3) from exc

    _process_single(data, gh)
    return


def _mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep)}"


def _load_event(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return cast(
            dict[str, Any], json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        log.warning("Failed to parse GITHUB_EVENT_PATH: %s", exc)
        return {}


def _extract_pr_number(evt: dict[str, Any]) -> int | None:
    # Try standard pull_request payload
    pr = evt.get("pull_request")
    if isinstance(pr, dict) and isinstance(pr.get("number"), int):
        return int(pr["number"])

    # Try issues payload (when used on issues events)
    issue = evt.get("issue")
    if isinstance(issue, dict) and isinstance(issue.get("number"), int):
        return int(issue["number"])

    # Try a direct number field
    if isinstance(evt.get("number"), int):
        return int(evt["number"])

    return None


def _read_github_context() -> GitHubContext:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    event_action = ""
    event_path_str = os.getenv("GITHUB_EVENT_PATH")
    event_path = Path(event_path_str) if event_path_str else None

    evt = _load_event(event_path)
    if isinstance(evt.get("action"), str):
        event_action = evt["action"]

    repository = os.getenv("GITHUB_REPOSITORY", "")
    repository_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    sha = os.getenv("GITHUB_SHA", "")

    base_ref = os.getenv("GITHUB_BASE_REF", "")
    head_ref = os.getenv("GITHUB_HEAD_REF", "")

    pr_number = _extract_pr_number(evt)
    if pr_number is None:
        env_pr = os.getenv("PR_NUMBER")
        if env_pr and env_pr.isdigit():
            pr_number = int(env_pr)

    ctx = models.GitHubContext(
        event_name=event_name,
        event_action=event_action,
        event_path=event_path,
        repository=repository,
        repository_owner=repository_owner,
        server_url=server_url,
        run_id=run_id,
        sha=sha,
        base_ref=base_ref,
        head_ref=head_ref,
        pr_number=pr_number,
    )
    return ctx


def _validate_inputs(data: Inputs) -> None:
    if data.use_pr_as_commit and data.submit_single_commits:
        msg = (
            "USE_PR_AS_COMMIT and SUBMIT_SINGLE_COMMITS cannot be enabled at "
            "the same time"
        )
        raise typer.BadParameter(msg)

    # Presence checks for required fields used by existing action
    for field_name in (
        "gerrit_known_hosts",
        "gerrit_ssh_privkey_g2g",
        "gerrit_ssh_user_g2g",
        "gerrit_ssh_user_g2g_email",
    ):
        if not getattr(data, field_name):
            log.error("Missing required input: %s", field_name)
            raise typer.BadParameter(f"Missing required input: {field_name}")  # noqa: TRY003

    # Validate fetch depth is a positive integer
    if data.fetch_depth <= 0:
        log.error("Invalid FETCH_DEPTH: %s", data.fetch_depth)
        raise typer.BadParameter("FETCH_DEPTH must be a positive integer")  # noqa: TRY003

    # Validate Issue ID is a single line string if provided
    if data.issue_id and ("\n" in data.issue_id or "\r" in data.issue_id):
        raise typer.BadParameter("Issue ID must be single line")  # noqa: TRY003


def _log_effective_config(data: Inputs, gh: GitHubContext) -> None:
    # Avoid logging sensitive values
    safe_privkey = _mask_secret(data.gerrit_ssh_privkey_g2g)
    log.info("Effective configuration (sanitized):")
    log.info("  SUBMIT_SINGLE_COMMITS: %s", data.submit_single_commits)
    log.info("  USE_PR_AS_COMMIT: %s", data.use_pr_as_commit)
    log.info("  FETCH_DEPTH: %s", data.fetch_depth)
    log.info(
        "  GERRIT_KNOWN_HOSTS: %s",
        "<provided>" if data.gerrit_known_hosts else "<missing>",
    )
    log.info("  GERRIT_SSH_PRIVKEY_G2G: %s", safe_privkey)
    log.info("  GERRIT_SSH_USER_G2G: %s", data.gerrit_ssh_user_g2g)
    log.info("  GERRIT_SSH_USER_G2G_EMAIL: %s", data.gerrit_ssh_user_g2g_email)
    log.info("  ORGANIZATION: %s", data.organization)
    log.info("  REVIEWERS_EMAIL: %s", data.reviewers_email or "")
    log.info("  PRESERVE_GITHUB_PRS: %s", data.preserve_github_prs)
    log.info("  DRY_RUN: %s", data.dry_run)
    log.info("  GERRIT_SERVER: %s", data.gerrit_server or "")
    log.info("  GERRIT_SERVER_PORT: %s", data.gerrit_server_port or "")
    log.info("  GERRIT_PROJECT: %s", data.gerrit_project or "")
    log.info("GitHub context:")
    log.info("  event_name: %s", gh.event_name)
    log.info("  event_action: %s", gh.event_action)
    log.info("  repository: %s", gh.repository)
    log.info("  repository_owner: %s", gh.repository_owner)
    log.info("  pr_number: %s", gh.pr_number)
    log.info("  base_ref: %s", gh.base_ref)
    log.info("  head_ref: %s", gh.head_ref)
    log.info("  sha: %s", gh.sha)


if __name__ == "__main__":
    # Invoke the Typer app when executed as a script.
    # Example:
    #   python -m github2gerrit_python.cli --help
    app()
