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
import re
from datetime import UTC
from datetime import datetime
from datetime import timedelta

from .github_api import GhPullRequest
from .github_api import GhRepository
from .github_api import build_client
from .github_api import get_repo_from_env
from .models import GitHubContext


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
    """Detects duplicate changes across pull requests."""

    def __init__(self, repo: GhRepository, lookback_days: int = 7):
        self.repo = repo
        self.lookback_days = lookback_days
        self._cutoff_date = datetime.now(UTC) - timedelta(days=lookback_days)

    def get_recent_prs(self, state: str = "all") -> list[GhPullRequest]:
        """Get recent PRs within the lookback period."""
        prs = []

        try:
            # Get recent PRs (GitHub returns newest first)
            for pr in self.repo.get_pulls(state=state):
                # Check if PR was updated within our lookback period
                updated_at = getattr(pr, "updated_at", None)
                if (
                    updated_at
                    and updated_at.replace(tzinfo=UTC) < self._cutoff_date
                ):
                    break  # PRs are sorted by update time, so we can stop here

                prs.append(pr)

                # Limit to reasonable number to avoid API rate limits
                if len(prs) >= 100:
                    break

        except Exception as exc:
            log.warning("Failed to fetch recent PRs: %s", exc)

        return prs

    def create_fingerprint(self, pr: GhPullRequest) -> ChangeFingerprint:
        """Create a fingerprint for a pull request."""
        title = getattr(pr, "title", "") or ""
        body = getattr(pr, "body", "") or ""

        # Try to get files changed (may fail due to API limits)
        files_changed = []
        try:
            # Note: get_files() may not be available on all PR objects
            if hasattr(pr, "get_files"):
                files = pr.get_files()
                files_changed = [getattr(f, "filename", "") for f in files]
        except Exception as exc:
            log.debug(
                "Could not get files for PR #%s: %s",
                getattr(pr, "number", "?"),
                exc,
            )

        return ChangeFingerprint(
            title=title, body=body, files_changed=files_changed
        )

    def find_similar_prs(
        self,
        target_fingerprint: ChangeFingerprint,
        exclude_pr: int | None = None,
    ) -> list[tuple[GhPullRequest, ChangeFingerprint]]:
        """Find PRs similar to the target fingerprint."""
        similar_prs = []
        recent_prs = self.get_recent_prs()

        for pr in recent_prs:
            pr_number = getattr(pr, "number", 0)

            # Skip the PR we're checking against
            if exclude_pr and pr_number == exclude_pr:
                continue

            try:
                pr_fingerprint = self.create_fingerprint(pr)
                if target_fingerprint.is_similar_to(pr_fingerprint):
                    similar_prs.append((pr, pr_fingerprint))
                    log.debug(
                        "Found similar PR #%d: %s",
                        pr_number,
                        getattr(pr, "title", "")[:50],
                    )
            except Exception as exc:
                log.debug("Error checking PR #%d: %s", pr_number, exc)

        return similar_prs

    def check_for_duplicates(
        self,
        target_pr: GhPullRequest,
        allow_duplicates: bool = False,
    ) -> None:
        """Check if the target PR is a duplicate of recent PRs.

        Args:
            target_pr: The PR to check for duplicates
            allow_duplicates: If True, only log warnings; if False, raise error

        Raises:
            DuplicateChangeError: If duplicates found and allow_duplicates=False
        """
        pr_number = getattr(target_pr, "number", 0)
        target_fingerprint = self.create_fingerprint(target_pr)

        log.debug(
            "Checking PR #%d for duplicates: %s", pr_number, target_fingerprint
        )

        similar_prs = self.find_similar_prs(
            target_fingerprint, exclude_pr=pr_number
        )

        if not similar_prs:
            log.debug("No similar PRs found for PR #%d", pr_number)
            return

        # Categorize similar PRs
        open_similar = []
        closed_similar = []

        for pr, _fingerprint in similar_prs:
            state = getattr(pr, "state", "unknown")
            if state == "open":
                open_similar.append(pr)
            elif state in ("closed", "merged"):
                closed_similar.append(pr)

        # Build warning/error message
        messages = []
        if closed_similar:
            closed_numbers = [
                getattr(pr, "number", "?") for pr in closed_similar
            ]
            messages.append(
                f"Recently closed PRs: #{', #'.join(map(str, closed_numbers))}"
            )

        if open_similar:
            open_numbers = [getattr(pr, "number", "?") for pr in open_similar]
            messages.append(f"Open PRs: #{', #'.join(map(str, open_numbers))}")

        full_message = (
            f"PR #{pr_number} appears to be a duplicate. "
            f"Similar changes found in {', '.join(messages)}. "
            f"Target PR title: '{getattr(target_pr, 'title', '')[:100]}'"
        )

        if allow_duplicates:
            log.warning("DUPLICATE DETECTED (allowed): %s", full_message)
        else:
            all_similar_numbers = [
                getattr(pr, "number", 0) for pr, _ in similar_prs
            ]
            raise DuplicateChangeError(full_message, all_similar_numbers)


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
            target_pr, allow_duplicates=allow_duplicates
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
