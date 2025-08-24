# SPDX-FileCopyrightText: 2024 Matthew Watkins
# SPDX-License-Identifier: Apache-2.0

"""Tests for duplicate change detection."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from github2gerrit_python.duplicate_detection import ChangeFingerprint
from github2gerrit_python.duplicate_detection import DuplicateChangeError
from github2gerrit_python.duplicate_detection import DuplicateDetector
from github2gerrit_python.duplicate_detection import check_for_duplicates
from github2gerrit_python.models import GitHubContext


class TestChangeFingerprint:
    """Test ChangeFingerprint functionality."""

    def test_normalize_title_basic(self) -> None:
        """Test basic title normalization."""
        fp = ChangeFingerprint("Fix authentication issue")
        assert fp._normalized_title == "fix authentication issue"

    def test_normalize_title_removes_conventional_commits(self) -> None:
        """Test that conventional commit prefixes are removed."""
        cases = [
            ("feat: Add new feature", "add new feature"),
            ("fix(auth): Fix authentication", "fix authentication"),
            ("docs: Update README", "update readme"),
            ("chore: Update dependencies", "update dependencies"),
        ]

        for input_title, expected in cases:
            fp = ChangeFingerprint(input_title)
            assert fp._normalized_title == expected

    def test_normalize_title_removes_versions(self) -> None:
        """Test that version numbers are normalized."""
        cases = [
            (
                "Bump library from 1.2.3 to 2.0.0",
                "bump library from x.y.z to x.y.z",
            ),
            ("Update v1.0 to v2.1.5", "update vx.y.z to vx.y.z"),
            ("Upgrade package 0.6 to 0.8", "upgrade package x.y.z to x.y.z"),
        ]

        for input_title, expected in cases:
            fp = ChangeFingerprint(input_title)
            assert fp._normalized_title == expected

    def test_normalize_title_removes_commit_hashes(self) -> None:
        """Test that commit hashes are normalized."""
        fp = ChangeFingerprint("Revert commit abc1234567890def")
        assert fp._normalized_title == "revert commit commit_hash"

    def test_identical_fingerprints_are_similar(self) -> None:
        """Test that identical fingerprints are detected as similar."""
        fp1 = ChangeFingerprint("Fix authentication issue")
        fp2 = ChangeFingerprint("Fix authentication issue")
        assert fp1.is_similar_to(fp2)

    def test_version_bumps_are_similar(self) -> None:
        """Test that version bumps are detected as similar."""
        fp1 = ChangeFingerprint("Bump library from 1.0 to 1.1")
        fp2 = ChangeFingerprint("Bump library from 1.1 to 1.2")
        assert fp1.is_similar_to(fp2)

    def test_different_libraries_not_similar(self) -> None:
        """Test that different libraries are not similar."""
        fp1 = ChangeFingerprint("Bump library-a from 1.0 to 1.1")
        fp2 = ChangeFingerprint("Bump library-b from 1.0 to 1.1")
        assert not fp1.is_similar_to(fp2)

    def test_similar_files_and_titles(self) -> None:
        """Test similarity detection with file changes."""
        fp1 = ChangeFingerprint(
            "Update requirements",
            files_changed=["requirements.txt", "pyproject.toml"],
        )
        fp2 = ChangeFingerprint(
            "Update requirements file",
            files_changed=["requirements.txt", "setup.py"],
        )
        assert fp1.is_similar_to(fp2)

    def test_content_hash_similarity(self) -> None:
        """Test content hash-based similarity."""
        fp1 = ChangeFingerprint("Fix issue", "This fixes a bug")
        fp2 = ChangeFingerprint("Fix issue", "This fixes a bug")
        assert fp1.is_similar_to(fp2)
        assert fp1._content_hash == fp2._content_hash


class TestDuplicateDetector:
    """Test DuplicateDetector functionality."""

    def _create_mock_pr(
        self,
        number: int,
        title: str,
        state: str = "open",
        updated_at: datetime | None = None,
        body: str = "",
    ) -> Any:
        """Create a mock PR object."""
        if updated_at is None:
            updated_at = datetime.now(UTC)

        pr = Mock()
        pr.number = number
        pr.title = title
        pr.body = body
        pr.state = state
        pr.updated_at = updated_at
        pr.get_files.return_value = []  # Empty files by default
        return pr

    def _create_mock_repo(self, prs: list[Any]) -> Any:
        """Create a mock repository with given PRs."""
        repo = Mock()
        repo.get_pulls.return_value = prs
        repo.get_pull.side_effect = lambda num: next(
            pr for pr in prs if pr.number == num
        )
        return repo

    def test_get_recent_prs_filters_by_date(self) -> None:
        """Test that get_recent_prs filters by lookback period."""
        now = datetime.now(UTC)
        old_date = now - timedelta(days=10)
        recent_date = now - timedelta(days=2)

        prs = [
            self._create_mock_pr(1, "Recent PR", updated_at=recent_date),
            self._create_mock_pr(2, "Old PR", updated_at=old_date),
        ]

        repo = self._create_mock_repo(prs)
        detector = DuplicateDetector(repo, lookback_days=7)

        recent_prs = detector.get_recent_prs()

        # Should only include the recent PR
        assert len(recent_prs) == 1
        assert recent_prs[0].number == 1

    def test_find_similar_prs_detects_duplicates(self) -> None:
        """Test that find_similar_prs detects duplicate PRs."""
        prs = [
            self._create_mock_pr(1, "Bump library from 1.0 to 1.1"),
            self._create_mock_pr(2, "Bump library from 1.1 to 1.2"),
            self._create_mock_pr(3, "Fix authentication"),
        ]

        repo = self._create_mock_repo(prs)
        detector = DuplicateDetector(repo)

        target_fp = ChangeFingerprint("Bump library from 1.2 to 1.3")
        similar_prs = detector.find_similar_prs(target_fp)

        # Should find the two library bump PRs
        assert len(similar_prs) == 2
        similar_numbers = [pr.number for pr, _ in similar_prs]
        assert 1 in similar_numbers
        assert 2 in similar_numbers

    def test_check_for_duplicates_raises_error(self) -> None:
        """Test that check_for_duplicates raises error for duplicates."""
        prs = [
            self._create_mock_pr(1, "Bump library from 1.0 to 1.1"),
            self._create_mock_pr(2, "Bump library from 1.1 to 1.2"),
        ]

        repo = self._create_mock_repo(prs)
        detector = DuplicateDetector(repo)

        # Check PR #2 against existing PRs (should find PR #1 as similar)
        with pytest.raises(DuplicateChangeError) as exc_info:
            detector.check_for_duplicates(prs[1], allow_duplicates=False)

        assert "appears to be a duplicate" in str(exc_info.value)
        assert exc_info.value.existing_prs == [1]

    def test_check_for_duplicates_allows_with_flag(self) -> None:
        """Test that check_for_duplicates allows duplicates with flag."""
        prs = [
            self._create_mock_pr(1, "Bump library from 1.0 to 1.1"),
            self._create_mock_pr(2, "Bump library from 1.1 to 1.2"),
        ]

        repo = self._create_mock_repo(prs)
        detector = DuplicateDetector(repo)

        # Should not raise error with allow_duplicates=True
        detector.check_for_duplicates(prs[1], allow_duplicates=True)

    def test_excludes_target_pr_from_comparison(self) -> None:
        """Test that the target PR is excluded from comparison."""
        prs = [
            self._create_mock_pr(1, "Fix authentication"),
            self._create_mock_pr(2, "Fix authentication"),  # Identical
        ]

        repo = self._create_mock_repo(prs)
        detector = DuplicateDetector(repo)

        target_fp = ChangeFingerprint("Fix authentication")
        similar_prs = detector.find_similar_prs(target_fp, exclude_pr=2)

        # Should only find PR #1, not PR #2 (excluded)
        assert len(similar_prs) == 1
        assert similar_prs[0][0].number == 1

    def test_categorizes_open_vs_closed_prs(self) -> None:
        """Test that open and closed PRs are categorized correctly."""
        prs = [
            self._create_mock_pr(1, "Fix auth", state="open"),
            self._create_mock_pr(2, "Fix auth", state="closed"),
            self._create_mock_pr(3, "Fix auth", state="merged"),
        ]

        repo = self._create_mock_repo(prs)
        detector = DuplicateDetector(repo)

        with pytest.raises(DuplicateChangeError) as exc_info:
            detector.check_for_duplicates(prs[0], allow_duplicates=False)

        error_msg = str(exc_info.value)
        assert (
            "Recently closed PRs: #2, #3" in error_msg
        )  # PR #1 is excluded as target
        # Note: GitHub's "merged" state might be handled differently


class TestCheckForDuplicatesFunction:
    """Test the convenience check_for_duplicates function."""

    def _create_mock_github_context(
        self, pr_number: int | None = 123
    ) -> GitHubContext:
        """Create a mock GitHub context."""
        return GitHubContext(
            event_name="pull_request",
            event_action="opened",
            event_path=Path("event.json"),
            repository="org/repo",
            repository_owner="org",
            server_url="https://github.com",
            run_id="123456",
            sha="abc123",
            base_ref="main",
            head_ref="feature-branch",
            pr_number=pr_number,
        )

    @patch("github2gerrit_python.duplicate_detection.build_client")
    @patch("github2gerrit_python.duplicate_detection.get_repo_from_env")
    def test_check_for_duplicates_success(
        self, mock_get_repo: Any, mock_build_client: Any
    ) -> None:
        """Test successful duplicate check."""
        # Mock the GitHub API
        mock_repo = Mock()
        mock_pr = Mock()
        mock_pr.title = "Fix authentication"
        mock_pr.body = "This fixes auth issues"
        mock_pr.get_files.return_value = []

        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_pulls.return_value = []  # No other PRs
        mock_get_repo.return_value = mock_repo

        gh = self._create_mock_github_context()

        # Should not raise any exception
        check_for_duplicates(gh, allow_duplicates=False)

    @patch("github2gerrit_python.duplicate_detection.build_client")
    @patch("github2gerrit_python.duplicate_detection.get_repo_from_env")
    def test_check_for_duplicates_no_pr_number(
        self, mock_get_repo: Any, mock_build_client: Any
    ) -> None:
        """Test that function handles missing PR number gracefully."""
        gh = self._create_mock_github_context(pr_number=None)

        # Should not raise any exception or make API calls
        check_for_duplicates(gh, allow_duplicates=False)

        mock_build_client.assert_not_called()
        mock_get_repo.assert_not_called()

    @patch("github2gerrit_python.duplicate_detection.build_client")
    @patch("github2gerrit_python.duplicate_detection.get_repo_from_env")
    def test_check_for_duplicates_api_failure_doesnt_crash(
        self, mock_get_repo: Any, mock_build_client: Any
    ) -> None:
        """Test that API failures don't crash the process."""
        # Mock API failure
        mock_build_client.side_effect = Exception("API Error")

        gh = self._create_mock_github_context()

        # Should not raise exception, just log warning
        check_for_duplicates(gh, allow_duplicates=False)

    @patch("github2gerrit_python.duplicate_detection.build_client")
    @patch("github2gerrit_python.duplicate_detection.get_repo_from_env")
    def test_check_for_duplicates_reraises_duplicate_error(
        self, mock_get_repo: Any, mock_build_client: Any
    ) -> None:
        """Test that DuplicateChangeError is re-raised."""
        # Mock finding duplicates
        mock_repo = Mock()
        target_pr = Mock()
        target_pr.title = "Fix auth"
        target_pr.body = ""
        target_pr.get_files.return_value = []

        duplicate_pr = Mock()
        duplicate_pr.number = 456
        duplicate_pr.title = "Fix auth"
        duplicate_pr.body = ""
        duplicate_pr.state = "open"
        duplicate_pr.updated_at = datetime.now(UTC)
        duplicate_pr.get_files.return_value = []

        mock_repo.get_pull.return_value = target_pr
        mock_repo.get_pulls.return_value = [duplicate_pr]
        mock_get_repo.return_value = mock_repo

        gh = self._create_mock_github_context()

        with pytest.raises(DuplicateChangeError):
            check_for_duplicates(gh, allow_duplicates=False)


class TestDependabotScenarios:
    """Test specific Dependabot-style scenarios."""

    def test_identical_dependabot_prs(self) -> None:
        """Test detection of identical Dependabot PRs."""
        fp1 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.8"
        )
        fp2 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.8"
        )
        fp3 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.8"
        )

        assert fp1.is_similar_to(fp2)
        assert fp1.is_similar_to(fp3)
        assert fp2.is_similar_to(fp3)

    def test_different_dependabot_versions(self) -> None:
        """Test that different version bumps are still similar."""
        fp1 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.7"
        )
        fp2 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.8"
        )
        fp3 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.7 to 0.8"
        )

        assert fp1.is_similar_to(fp2)
        assert fp1.is_similar_to(fp3)
        assert fp2.is_similar_to(fp3)

    def test_different_dependabot_packages(self) -> None:
        """Test that different packages are not similar."""
        fp1 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.8"
        )
        fp2 = ChangeFingerprint("Bump actions/checkout from 3 to 4")

        assert not fp1.is_similar_to(fp2)

    def test_mixed_case_and_formatting(self) -> None:
        """Test that formatting differences don't affect detection."""
        fp1 = ChangeFingerprint(
            "Bump lfit/gerrit-review-action from 0.6 to 0.8"
        )
        fp2 = ChangeFingerprint(
            "bump lfit/gerrit-review-action from 0.6 to 0.8"
        )
        fp3 = ChangeFingerprint(
            "Bump `lfit/gerrit-review-action` from 0.6 to 0.8"
        )

        assert fp1.is_similar_to(fp2)
        assert fp1.is_similar_to(fp3)
        assert fp2.is_similar_to(fp3)
