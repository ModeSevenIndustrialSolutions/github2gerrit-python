# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
"""
Tests for the SSH setup functionality in core.py.

These tests verify that the SSH setup is non-invasive and doesn't
modify user SSH configuration while still providing secure access.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from github2gerrit_python.core import Orchestrator
from github2gerrit_python.models import Inputs


@pytest.fixture  # type: ignore[misc]
def minimal_inputs() -> Inputs:
    return Inputs(
        submit_single_commits=False,
        use_pr_as_commit=False,
        fetch_depth=10,
        gerrit_known_hosts="gerrit.example.org ssh-rsa AAAAB3NzaC1yc2E...",
        gerrit_ssh_privkey_g2g="-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "fake_key_content\n"
        "-----END OPENSSH PRIVATE KEY-----",
        gerrit_ssh_user_g2g="gerrit-bot",
        gerrit_ssh_user_g2g_email="gerrit-bot@example.org",
        organization="example",
        reviewers_email="",
        preserve_github_prs=False,
        dry_run=False,
        gerrit_server="",
        gerrit_server_port="29418",
        gerrit_project="",
        issue_id="",
        allow_duplicates=False,
    )


def test_ssh_setup_creates_workspace_specific_files(
    tmp_path: Path, minimal_inputs: Inputs
) -> None:
    """SSH setup should create files in workspace, not user SSH directory."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create fake user SSH directory to ensure it's not modified
    user_ssh_dir = Path.home() / ".ssh"
    original_user_ssh_exists = user_ssh_dir.exists()

    orch = Orchestrator(workspace=workspace)

    # Act
    orch._setup_ssh(minimal_inputs)

    # Assert: tool-specific SSH directory is created
    tool_ssh_dir = workspace / ".ssh-g2g"
    assert tool_ssh_dir.exists()
    assert tool_ssh_dir.is_dir()
    assert oct(tool_ssh_dir.stat().st_mode)[-3:] == "700"

    # Assert: SSH key is created in tool directory
    key_path = tool_ssh_dir / "gerrit_key"
    assert key_path.exists()
    assert key_path.is_file()
    assert oct(key_path.stat().st_mode)[-3:] == "600"

    # Assert: known hosts is created in tool directory
    known_hosts_path = tool_ssh_dir / "known_hosts"
    assert known_hosts_path.exists()
    assert known_hosts_path.is_file()

    # Assert: user SSH directory is not modified
    user_ssh_modified = user_ssh_dir.exists() != original_user_ssh_exists
    assert not user_ssh_modified, "User SSH directory should not be modified"

    # Assert: user's id_rsa is not touched
    user_id_rsa = user_ssh_dir / "id_rsa"
    if user_id_rsa.exists():
        # If user has id_rsa, check it wasn't modified
        content = user_id_rsa.read_text()
        assert "fake_key_content" not in content


def test_ssh_setup_skips_when_credentials_missing(tmp_path: Path) -> None:
    """SSH setup should skip when SSH credentials are not provided."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orch = Orchestrator(workspace=workspace)

    inputs_no_key = Inputs(
        submit_single_commits=False,
        use_pr_as_commit=False,
        fetch_depth=10,
        gerrit_known_hosts="",
        gerrit_ssh_privkey_g2g="",
        gerrit_ssh_user_g2g="gerrit-bot",
        gerrit_ssh_user_g2g_email="gerrit-bot@example.org",
        organization="example",
        reviewers_email="",
        preserve_github_prs=False,
        dry_run=False,
        gerrit_server="",
        gerrit_server_port="29418",
        gerrit_project="",
        issue_id="",
        allow_duplicates=False,
    )

    # Act
    orch._setup_ssh(inputs_no_key)

    # Assert: no SSH directory is created
    tool_ssh_dir = workspace / ".ssh-g2g"
    assert not tool_ssh_dir.exists()

    # Assert: SSH command is None
    assert orch._git_ssh_command is None


def test_git_ssh_command_prevents_agent_scanning(
    tmp_path: Path, minimal_inputs: Inputs
) -> None:
    """Generated SSH command should prevent SSH agent scanning."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orch = Orchestrator(workspace=workspace)

    # Act
    orch._setup_ssh(minimal_inputs)
    ssh_cmd = orch._git_ssh_command

    # Assert: SSH command is generated
    assert ssh_cmd is not None
    assert ssh_cmd.startswith("ssh ")

    # Assert: critical options are present
    assert "-o IdentitiesOnly=yes" in ssh_cmd
    assert "-o StrictHostKeyChecking=yes" in ssh_cmd
    assert "-o PasswordAuthentication=no" in ssh_cmd

    # Assert: tool-specific files are referenced
    assert str(workspace / ".ssh-g2g" / "gerrit_key") in ssh_cmd
    assert str(workspace / ".ssh-g2g" / "known_hosts") in ssh_cmd


def test_ssh_cleanup_removes_temporary_files(
    tmp_path: Path, minimal_inputs: Inputs
) -> None:
    """SSH cleanup should remove all temporary files."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orch = Orchestrator(workspace=workspace)

    # Setup SSH files
    orch._setup_ssh(minimal_inputs)
    tool_ssh_dir = workspace / ".ssh-g2g"
    assert tool_ssh_dir.exists()

    # Act
    orch._cleanup_ssh()

    # Assert: temporary directory is removed
    assert not tool_ssh_dir.exists()


def test_ssh_cleanup_handles_missing_files_gracefully(tmp_path: Path) -> None:
    """SSH cleanup should handle missing files without error."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orch = Orchestrator(workspace=workspace)

    # Act: cleanup without setup (no files to clean)
    # Should not raise an exception
    orch._cleanup_ssh()

    # Assert: no errors occurred (test passes if no exception raised)
    assert True


def test_ssh_setup_preserves_existing_user_config(
    tmp_path: Path, minimal_inputs: Inputs
) -> None:
    """SSH setup should not modify existing user SSH configuration."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create a fake user SSH config to protect
    fake_user_ssh = tmp_path / "fake_user_ssh"
    fake_user_ssh.mkdir(mode=0o700)
    fake_config = fake_user_ssh / "config"
    fake_config.write_text("Host example.com\n    User testuser\n")
    original_config = fake_config.read_text()

    orch = Orchestrator(workspace=workspace)

    # Mock Path.home() to point to our fake directory
    with patch("pathlib.Path.home", return_value=tmp_path):
        # Act
        orch._setup_ssh(minimal_inputs)

        # Assert: user SSH config is unchanged
        if fake_config.exists():
            current_config = fake_config.read_text()
            assert current_config == original_config


def test_ssh_command_isolation_from_environment(
    tmp_path: Path, minimal_inputs: Inputs
) -> None:
    """SSH command should be isolated from SSH agent environment."""
    # Arrange
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orch = Orchestrator(workspace=workspace)

    # Setup with SSH agent environment variables
    original_auth_sock = os.environ.get("SSH_AUTH_SOCK")
    original_agent_pid = os.environ.get("SSH_AGENT_PID")

    try:
        os.environ["SSH_AUTH_SOCK"] = str(tmp_path / "fake_ssh_agent")
        os.environ["SSH_AGENT_PID"] = "12345"

        # Act
        orch._setup_ssh(minimal_inputs)
        ssh_cmd = orch._git_ssh_command

        # Assert: IdentitiesOnly prevents agent usage regardless of env
        assert ssh_cmd is not None
        assert "-o IdentitiesOnly=yes" in ssh_cmd

    finally:
        # Cleanup environment
        if original_auth_sock is not None:
            os.environ["SSH_AUTH_SOCK"] = original_auth_sock
        elif "SSH_AUTH_SOCK" in os.environ:
            del os.environ["SSH_AUTH_SOCK"]

        if original_agent_pid is not None:
            os.environ["SSH_AGENT_PID"] = original_agent_pid
        elif "SSH_AGENT_PID" in os.environ:
            del os.environ["SSH_AGENT_PID"]
