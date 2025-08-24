# SPDX-FileCopyrightText: 2024 Matthew Watkins
# SPDX-License-Identifier: Apache-2.0

"""
Duplicate change detection for github2gerrit.

This module provides functionality to detect potentially duplicate changes
before submitting them to Gerrit, helping to prevent spam and redundant
submissions from automated tools like Dependabot.
"""

import hashlib
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from .github_api import GhPullRequest
from .github_api import GhRepository
from .github_api import build_client
from .github_api import get_repo_from_env
from .models import GitHubContext


# Optional Gerrit REST API support
try:
    from pygerrit2 import GerritRestAPI
    from pygerrit2 import HTTPBasicAuth
except ImportError:
    GerritRestAPI = None
    HTTPBasicAuth = None


log = logging.getLogger(__name__)

__all__ = [
    "ChangeFingerprint",
    "DuplicateChangeError",
    "DuplicateDetector",
    "check_for_duplicates",
]


class DuplicateChangeError(Exception):
    """Raised when a duplicate change is detected."""

    def __init__(self, message: str, existing_prs: list[int]) -> None:
        super().__init__(message)
        self.existing_prs = existing_prs


class ChangeFingerprint:
    """Represents a fingerprint of a change for duplicate detection."""

    def __init__(
        self, title: str, body: str = "", files_changed: list[str] | None = None
    ):
        self.title = title.strip()
        self.body = (body or "").strip()
        self.files_changed = sorted(files_changed or [])
        self._normalized_title = self._normalize_title(title)
        self._content_hash = self._compute_content_hash()

    def _normalize_title(self, title: str) -> str:
        """Normalize PR title for comparison."""
        # Remove common prefixes/suffixes
        normalized = title.strip()

        # Remove conventional commit prefixes like "feat:", "fix:", etc.
        normalized = re.sub(
            r"^(feat|fix|docs|style|refactor|test|chore|ci|build|perf)"
            r"(\(.+?\))?: ",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

        # Remove markdown formatting
        normalized = re.sub(r"[*_`]", "", normalized)

        # Remove version number variations for dependency updates
        # E.g., "from 0.6 to 0.8" -> "from x.y.z to x.y.z"
        # Handle v-prefixed versions first, then plain versions
        normalized = re.sub(r"\bv\d+(\.\d+)*(-\w+)?\b", "vx.y.z", normalized)
        normalized = re.sub(r"\b\d+(\.\d+)+(-\w+)?\b", "x.y.z", normalized)
        normalized = re.sub(r"\b\d+\.\d+\b", "x.y.z", normalized)

        # Remove specific commit hashes
        normalized = re.sub(r"\b[a-f0-9]{7,40}\b", "commit_hash", normalized)

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized.lower()

    def _compute_content_hash(self) -> str:
        """Compute a hash of the change content."""
        content = (
            f"{self._normalized_title}\n{self.body}\n"
            f"{','.join(self.files_changed)}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def is_similar_to(
        self, other: "ChangeFingerprint", similarity_threshold: float = 0.8
    ) -> bool:
        """Check if this fingerprint is similar to another."""
        # Exact normalized title match
        if self._normalized_title == other._normalized_title:
            return True

        # Content hash match
        if self._content_hash == other._content_hash:
            return True

        # Check for similar file changes (for dependency updates)
        if self.files_changed and other.files_changed:
            common_files = set(self.files_changed) & set(other.files_changed)
            union_files = set(self.files_changed) | set(other.files_changed)
            if common_files and union_files:
                overlap_ratio = len(common_files) / len(union_files)
                # If files overlap, check title similarity (lower threshold)
                if overlap_ratio > 0:
                    return self._titles_similar(other, 0.6)

        # Check title similarity even without file changes
        return self._titles_similar(other, similarity_threshold)

    def _titles_similar(
        self, other: "ChangeFingerprint", threshold: float
    ) -> bool:
        """Check if titles are similar using simple string similarity."""
        title1 = self._normalized_title
        title2 = other._normalized_title

        if not title1 or not title2:
            return False

        # Simple Jaccard similarity on words
        words1 = set(title1.split())
        words2 = set(title2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return (intersection / union) >= threshold

    def __str__(self) -> str:
        return (
            f"ChangeFingerprint(title='{self.title[:50]}...', "
            f"hash={self._content_hash})"
        )


class DuplicateDetector:
    """Detects duplicate Gerrit changes for GitHub pull requests."""

    def __init__(self, repo: GhRepository, lookback_days: int = 7):
        self.repo = repo
        self.lookback_days = lookback_days
        self._cutoff_date = datetime.now(UTC) - timedelta(days=lookback_days)

    def _match_first_group(self, pattern: str, text: str) -> str:
        """Extract first regex group match from text."""
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def _resolve_gerrit_info_from_env_or_gitreview(
        self, gh: GitHubContext
    ) -> tuple[str, str] | None:
        """Resolve Gerrit host and project from environment or .gitreview file.

        Returns:
            Tuple of (host, project) if found, None otherwise
        """
        # First try environment variables (same as core module)
        gerrit_host = os.getenv("GERRIT_SERVER", "").strip()
        gerrit_project = os.getenv("GERRIT_PROJECT", "").strip()

        if gerrit_host and gerrit_project:
            return (gerrit_host, gerrit_project)

        # Try to read .gitreview file locally first
        gitreview_path = Path(".gitreview")
        if gitreview_path.exists():
            try:
                text = gitreview_path.read_text(encoding="utf-8")
                host = self._match_first_group(r"(?m)^host=(.+)$", text)
                proj = self._match_first_group(r"(?m)^project=(.+)$", text)
                if host and proj:
                    project = proj.removesuffix(".git")
                    return (host.strip(), project.strip())
            except Exception as exc:
                log.debug("Failed to read local .gitreview: %s", exc)

        # Try to fetch .gitreview remotely (simplified version of core logic)
        try:
            repo_full = gh.repository.strip() if gh.repository else ""
            if not repo_full:
                return None

            # Try a few common branches
            branches = []
            if gh.head_ref:
                branches.append(gh.head_ref)
            if gh.base_ref:
                branches.append(gh.base_ref)
            branches.extend(["master", "main"])

            for branch in branches:
                if not branch:
                    continue

                url = (
                    f"https://raw.githubusercontent.com/"
                    f"{repo_full}/refs/heads/{branch}/.gitreview"
                )

                parsed = urllib.parse.urlparse(url)
                if (
                    parsed.scheme != "https"
                    or parsed.netloc != "raw.githubusercontent.com"
                ):
                    continue

                try:
                    log.debug("Fetching .gitreview from: %s", url)
                    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                        text_remote = resp.read().decode("utf-8")

                    host = self._match_first_group(
                        r"(?m)^host=(.+)$", text_remote
                    )
                    proj = self._match_first_group(
                        r"(?m)^project=(.+)$", text_remote
                    )

                    if host and proj:
                        project = proj.removesuffix(".git")
                        return (host.strip(), project.strip())

                except Exception as exc:
                    log.debug(
                        "Failed to fetch .gitreview from %s: %s", url, exc
                    )
                    continue

        except Exception as exc:
            log.debug("Failed to resolve .gitreview remotely: %s", exc)

        return None

    def _build_gerrit_rest_client(self, gerrit_host: str) -> object | None:
        """Build a Gerrit REST API client if pygerrit2 is available."""
        if GerritRestAPI is None:
            log.debug(
                "pygerrit2 not available, skipping Gerrit duplicate check"
            )
            return None

        base_path = os.getenv("GERRIT_HTTP_BASE_PATH", "").strip().strip("/")
        base_url = (
            f"https://{gerrit_host}/"
            if not base_path
            else f"https://{gerrit_host}/{base_path}/"
        )

        http_user = (
            os.getenv("GERRIT_HTTP_USER", "").strip()
            or os.getenv("GERRIT_SSH_USER_G2G", "").strip()
        )
        http_pass = os.getenv("GERRIT_HTTP_PASSWORD", "").strip()

        try:
            if http_user and http_pass:
                if HTTPBasicAuth is None:
                    log.debug("pygerrit2 HTTPBasicAuth not available")
                    return None
                # Type ignore needed for dynamic import returning Any
                return GerritRestAPI(  # type: ignore[no-any-return]
                    url=base_url, auth=HTTPBasicAuth(http_user, http_pass)
                )
            else:
                # Type ignore needed for dynamic import returning Any
                return GerritRestAPI(url=base_url)  # type: ignore[no-any-return]
        except Exception as exc:
            log.debug("Failed to create Gerrit REST client: %s", exc)
            return None

    def _build_gerrit_rest_client_with_r_path(
        self, gerrit_host: str
    ) -> object | None:
        """Build a Gerrit REST API client with /r/ base path for fallback."""
        if GerritRestAPI is None:
            return None

        fallback_url = f"https://{gerrit_host}/r/"
        http_user = (
            os.getenv("GERRIT_HTTP_USER", "").strip()
            or os.getenv("GERRIT_SSH_USER_G2G", "").strip()
        )
        http_pass = os.getenv("GERRIT_HTTP_PASSWORD", "").strip()

        try:
            if http_user and http_pass:
                if HTTPBasicAuth is None:
                    return None
                # Type ignore needed for dynamic import returning Any
                return GerritRestAPI(  # type: ignore[no-any-return]
                    url=fallback_url, auth=HTTPBasicAuth(http_user, http_pass)
                )
            else:
                # Type ignore needed for dynamic import returning Any
                return GerritRestAPI(url=fallback_url)  # type: ignore[no-any-return]
        except Exception as exc:
            log.debug(
                "Failed to create Gerrit REST client with /r/ path: %s", exc
            )
            return None

    def check_gerrit_for_existing_change(self, gh: GitHubContext) -> bool:
        """Check if a Gerrit change already exists for the given GitHub PR.

        Args:
            gh: GitHub context containing PR and repository information

        Returns:
            True if a Gerrit change already exists for this PR, False otherwise
        """
        if not gh.pr_number:
            return False

        # Resolve Gerrit host and project
        gerrit_info = self._resolve_gerrit_info_from_env_or_gitreview(gh)
        if not gerrit_info:
            log.debug(
                "Cannot resolve Gerrit host/project, "
                "skipping Gerrit duplicate check"
            )
            return False

        gerrit_host, gerrit_project = gerrit_info

        rest = self._build_gerrit_rest_client(gerrit_host)
        if rest is None:
            log.debug(
                "Cannot check Gerrit for duplicates, REST client unavailable"
            )
            return False

        # Generate the GitHub change hash for this PR
        github_hash = DuplicateDetector._generate_github_change_hash(gh)

        try:
            # Search for changes that contain the GitHub hash in commit messages
            # This is more reliable than comment-based searches
            query = (
                f'project:{gerrit_project} message:"GitHub-Hash: {github_hash}"'
            )
            path = f"/changes/?q={query}&n=10"

            log.debug(
                "Searching Gerrit for existing changes with GitHub hash %s, "
                "query: %s",
                github_hash,
                query,
            )
            # Use getattr for dynamic method access to avoid type checking
            changes = rest.get(path)  # type: ignore[attr-defined]

            if changes:
                log.info(
                    "Found %d existing Gerrit change(s) for GitHub PR #%d: %s",
                    len(changes),
                    gh.pr_number,
                    [f"{c.get('_number', '?')}" for c in changes],
                )
                return True
            else:
                log.debug(
                    "No existing Gerrit changes found for GitHub PR #%d",
                    gh.pr_number,
                )
                return False

        except Exception as exc:
            # Check if this is a 404 error and try /r/ fallback
            status = getattr(
                getattr(exc, "response", None), "status_code", None
            )
            if status == 404:
                try:
                    log.debug("Trying /r/ fallback for Gerrit API")
                    fallback_rest = self._build_gerrit_rest_client_with_r_path(
                        gerrit_host
                    )
                    if fallback_rest:
                        changes = fallback_rest.get(path)  # type: ignore[attr-defined]
                        if changes:
                            log.info(
                                "Found %d existing Gerrit change(s) for PR #%d "
                                "via /r/ fallback: %s",
                                len(changes),
                                gh.pr_number,
                                [f"{c.get('_number', '?')}" for c in changes],
                            )
                            return True
                        else:
                            log.debug(
                                "No existing Gerrit changes found for PR #%d "
                                "via /r/ fallback",
                                gh.pr_number,
                            )
                            return False
                except Exception as exc2:
                    log.warning(
                        "Failed to query Gerrit via /r/ fallback: %s", exc2
                    )
                    return False

            log.warning("Failed to query Gerrit for existing changes: %s", exc)
            # If we can't check Gerrit, err on the side of caution
            return False

    @staticmethod
    def _generate_github_change_hash(gh: GitHubContext) -> str:
        """Generate a deterministic hash for a GitHub PR to identify duplicates.

        This creates a SHA256 hash based on stable PR metadata that uniquely
        identifies the change content, making duplicate detection reliable
        regardless of comment formatting or API issues.

        Args:
            gh: GitHub context containing PR information

        Returns:
            Hex-encoded SHA256 hash string (first 16 characters for readability)
        """
        import hashlib

        # Build hash input from stable, unique PR identifiers
        # Use server_url + repository + pr_number for global uniqueness
        hash_input = f"{gh.server_url}/{gh.repository}/pull/{gh.pr_number}"

        # Create SHA256 hash and take first 16 characters for readability
        hash_bytes = hashlib.sha256(hash_input.encode("utf-8")).digest()
        hash_hex = hash_bytes.hex()[:16]

        log.debug(
            "Generated GitHub change hash for %s: %s", hash_input, hash_hex
        )
        return hash_hex

    def check_for_duplicates(
        self,
        target_pr: GhPullRequest,
        allow_duplicates: bool = False,
        gh: GitHubContext | None = None,
    ) -> None:
        """Check if the target PR is a duplicate in Gerrit.

        Args:
            target_pr: The PR to check for duplicates
            allow_duplicates: If True, only log warnings; if False, raise error
            gh: GitHub context for Gerrit duplicate checking

        Raises:
            DuplicateChangeError: If duplicates found and allow_duplicates=False
        """
        pr_number = getattr(target_pr, "number", 0)

        log.debug("Checking PR #%d for Gerrit duplicates", pr_number)

        # Check if this PR already has a corresponding Gerrit change
        if gh and self.check_gerrit_for_existing_change(gh):
            full_message = (
                f"PR #{pr_number} already has an existing Gerrit change. "
                f"Skipping duplicate submission. "
                f"Target PR title: '{getattr(target_pr, 'title', '')[:100]}'"
            )

            if allow_duplicates:
                log.warning(
                    "GERRIT DUPLICATE DETECTED (allowed): %s", full_message
                )
                return
            else:
                raise DuplicateChangeError(full_message, [])

        log.debug("No existing Gerrit change found for PR #%d", pr_number)


def check_for_duplicates(
    gh: GitHubContext,
    allow_duplicates: bool = False,
    lookback_days: int = 7,
) -> None:
    """Convenience function to check for duplicates.

    Args:
        gh: GitHub context containing PR information
        allow_duplicates: If True, only log warnings; if False, raise exception
        lookback_days: Number of days to look back for similar PRs

    Raises:
        DuplicateChangeError: If duplicates found and allow_duplicates=False
    """
    if not gh.pr_number:
        log.debug("No PR number provided, skipping duplicate check")
        return

    try:
        client = build_client()
        repo = get_repo_from_env(client)

        # Get the target PR
        target_pr = repo.get_pull(gh.pr_number)

        # Create detector and check
        detector = DuplicateDetector(repo, lookback_days=lookback_days)
        detector.check_for_duplicates(
            target_pr, allow_duplicates=allow_duplicates, gh=gh
        )

        log.info("Duplicate check completed for PR #%d", gh.pr_number)

    except DuplicateChangeError:
        # Re-raise duplicate errors
        raise
    except Exception as exc:
        log.warning(
            "Duplicate detection failed for PR #%d: %s", gh.pr_number, exc
        )
        # Don't fail the entire process if duplicate detection has issues
