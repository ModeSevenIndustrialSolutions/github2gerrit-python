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

        def get_pulls_with_state(state: str = "all") -> list[Any]:
            if state == "open":
                return [pr for pr in prs if pr.state == "open"]
            elif state == "closed":
                return [pr for pr in prs if pr.state in ("closed", "merged")]
            else:  # state == "all"
                return prs

        repo.get_pulls.side_effect = get_pulls_with_state
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

        # Test that detector was initialized properly
        assert detector.repo == repo
        assert detector.lookback_days == 7

    def test_detector_basic_functionality(self) -> None:
        """Test basic detector functionality."""
        repo = Mock()
        detector = DuplicateDetector(repo)

        assert detector.repo == repo
        assert detector.lookback_days == 7

    def test_check_for_duplicates_no_gerrit_config(self) -> None:
        """Test that check_for_duplicates works without Gerrit config."""
        pr = self._create_mock_pr(1, "Fix authentication")
        detector = DuplicateDetector(Mock())

        # Should not raise error when no Gerrit config is available
        detector.check_for_duplicates(pr, allow_duplicates=False)

    def test_check_for_duplicates_allows_with_flag(self) -> None:
        """Test that check_for_duplicates allows duplicates with flag."""
        pr = self._create_mock_pr(1, "Fix authentication")
        detector = DuplicateDetector(Mock())

        # Should not raise error with allow_duplicates=True
        detector.check_for_duplicates(pr, allow_duplicates=True)


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

        mock_repo.get_pull.return_value = mock_pr
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


class TestGerritDuplicateDetection:
    """Test Gerrit duplicate detection functionality."""

    def _create_mock_github_context(
        self, pr_number: int = 123, repository: str = "org/repo"
    ) -> GitHubContext:
        """Create a mock GitHub context."""
        return GitHubContext(
            event_name="pull_request",
            event_action="opened",
            event_path=Path("event.json"),
            repository=repository,
            repository_owner="org",
            server_url="https://github.com",
            run_id="123456",
            sha="abc123",
            base_ref="main",
            head_ref="feature-branch",
            pr_number=pr_number,
        )

    def _create_mock_repo(self, prs: list[Any]) -> Any:
        """Create a mock GitHub repository with the given PRs."""
        mock_repo = Mock()
        mock_repo.get_pulls.return_value = prs
        return mock_repo

    def test_resolve_gerrit_info_from_env(self) -> None:
        """Test resolving Gerrit info from environment variables."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        with patch.dict(
            "os.environ",
            {
                "GERRIT_SERVER": "gerrit.example.org",
                "GERRIT_PROJECT": "test/project",
            },
        ):
            result = detector._resolve_gerrit_info_from_env_or_gitreview(gh)
            assert result == ("gerrit.example.org", "test/project")

    def test_resolve_gerrit_info_missing_env(self) -> None:
        """Test that missing environment variables return None."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        with patch.dict("os.environ", {}, clear=True):
            result = detector._resolve_gerrit_info_from_env_or_gitreview(gh)
            assert result is None

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_resolve_gerrit_info_from_local_gitreview(
        self, mock_read_text: Any, mock_exists: Any
    ) -> None:
        """Test resolving Gerrit info from local .gitreview file."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        mock_exists.return_value = True
        mock_read_text.return_value = """[gerrit]
host=gerrit.example.org
port=29418
project=test/project.git
"""

        with patch.dict("os.environ", {}, clear=True):
            result = detector._resolve_gerrit_info_from_env_or_gitreview(gh)
            assert result == ("gerrit.example.org", "test/project")

    @patch("urllib.request.urlopen")
    @patch("pathlib.Path.exists")
    def test_resolve_gerrit_info_from_remote_gitreview(
        self, mock_exists: Any, mock_urlopen: Any
    ) -> None:
        """Test resolving Gerrit info from remote .gitreview file."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        mock_exists.return_value = False

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.read.return_value = b"""[gerrit]
host=gerrit.example.org
port=29418
project=test/project.git
"""
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        with patch.dict("os.environ", {}, clear=True):
            result = detector._resolve_gerrit_info_from_env_or_gitreview(gh)
            assert result == ("gerrit.example.org", "test/project")

    @patch("github2gerrit_python.duplicate_detection.GerritRestAPI")
    def test_check_gerrit_for_existing_change_found(
        self, mock_gerrit_api: Any
    ) -> None:
        """Test that existing Gerrit changes are detected."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        # Mock Gerrit REST API response with existing change
        mock_rest = Mock()
        mock_rest.get.return_value = [{"_number": "12345"}]
        mock_gerrit_api.return_value = mock_rest

        with patch.dict(
            "os.environ",
            {
                "GERRIT_SERVER": "gerrit.example.org",
                "GERRIT_PROJECT": "test/project",
            },
        ):
            result = detector.check_gerrit_for_existing_change(gh)
            assert result is True

            # Verify the correct query was made - should search for GitHub hash in message
            # Hash is SHA256 of "https://github.com/org/repo/pull/123" truncated to 16 chars
            import hashlib

            hash_input = "https://github.com/org/repo/pull/123"
            expected_hash = hashlib.sha256(
                hash_input.encode("utf-8")
            ).hexdigest()[:16]
            expected_query = (
                f'project:test/project message:"GitHub-Hash: {expected_hash}"'
            )
            mock_rest.get.assert_called_once_with(
                f"/changes/?q={expected_query}&n=10"
            )

    @patch("github2gerrit_python.duplicate_detection.GerritRestAPI")
    def test_check_gerrit_for_existing_change_not_found(
        self, mock_gerrit_api: Any
    ) -> None:
        """Test that no existing Gerrit changes are handled correctly."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        # Mock Gerrit REST API response with no changes
        mock_rest = Mock()
        mock_rest.get.return_value = []
        mock_gerrit_api.return_value = mock_rest

        with patch.dict(
            "os.environ",
            {
                "GERRIT_SERVER": "gerrit.example.org",
                "GERRIT_PROJECT": "test/project",
            },
        ):
            result = detector.check_gerrit_for_existing_change(gh)
            assert result is False

    def test_check_gerrit_for_existing_change_no_pr_number(self) -> None:
        """Test that missing PR number is handled gracefully."""
        detector = DuplicateDetector(Mock())
        gh = GitHubContext(
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
            pr_number=None,
        )

        result = detector.check_gerrit_for_existing_change(gh)
        assert result is False

    @patch("github2gerrit_python.duplicate_detection.GerritRestAPI")
    def test_check_gerrit_for_existing_change_api_error(
        self, mock_gerrit_api: Any
    ) -> None:
        """Test that Gerrit API errors are handled gracefully."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        # Mock Gerrit REST API to raise an exception
        mock_rest = Mock()
        mock_rest.get.side_effect = Exception("API Error")
        mock_gerrit_api.return_value = mock_rest

        with patch.dict(
            "os.environ",
            {
                "GERRIT_SERVER": "gerrit.example.org",
                "GERRIT_PROJECT": "test/project",
            },
        ):
            result = detector.check_gerrit_for_existing_change(gh)
            assert result is False

    @patch("github2gerrit_python.duplicate_detection.GerritRestAPI")
    def test_check_for_duplicates_with_gerrit_duplicate(
        self, mock_gerrit_api: Any
    ) -> None:
        """Test that Gerrit duplicates are detected and prevent new submissions."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        # Mock target PR
        target_pr = Mock()
        target_pr.number = 123
        target_pr.title = "Fix authentication"
        target_pr.body = ""
        target_pr.get_files.return_value = []

        # Mock Gerrit REST API to return existing change
        mock_rest = Mock()
        mock_rest.get.return_value = [{"_number": "12345"}]
        mock_gerrit_api.return_value = mock_rest

        with patch.dict(
            "os.environ",
            {
                "GERRIT_SERVER": "gerrit.example.org",
                "GERRIT_PROJECT": "test/project",
            },
        ):
            with pytest.raises(DuplicateChangeError) as exc_info:
                detector.check_for_duplicates(
                    target_pr, allow_duplicates=False, gh=gh
                )

            assert "already has an existing Gerrit change" in str(
                exc_info.value
            )

    @patch("github2gerrit_python.duplicate_detection.GerritRestAPI")
    def test_check_for_duplicates_with_gerrit_duplicate_allowed(
        self, mock_gerrit_api: Any
    ) -> None:
        """Test that Gerrit duplicates are allowed with the flag."""
        detector = DuplicateDetector(Mock())
        gh = self._create_mock_github_context()

        # Mock target PR
        target_pr = Mock()
        target_pr.number = 123
        target_pr.title = "Fix authentication"
        target_pr.body = ""
        target_pr.get_files.return_value = []

        # Mock Gerrit REST API to return existing change
        mock_rest = Mock()
        mock_rest.get.return_value = [{"_number": "12345"}]
        mock_gerrit_api.return_value = mock_rest

        with patch.dict(
            "os.environ",
            {
                "GERRIT_SERVER": "gerrit.example.org",
                "GERRIT_PROJECT": "test/project",
            },
        ):
            # Should not raise exception with allow_duplicates=True
            detector.check_for_duplicates(
                target_pr, allow_duplicates=True, gh=gh
            )

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
