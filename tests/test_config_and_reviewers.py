# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation


import os
from pathlib import Path

import pytest

from github2gerrit.config import apply_config_to_env
from github2gerrit.config import load_org_config
from github2gerrit.gitutils import enumerate_reviewer_emails


def test_config_loading_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify that:
    - INI values are loaded for the specified org and normalized
    - ${ENV:VAR} expansions work
    - Quotes are stripped
    - apply_config_to_env does not override existing environment vars
    """
    cfg_text = """
[default]
GERRIT_SERVER = "gerrit.example.org"
GERRIT_SERVER_PORT = "29418"
PRESERVE_GITHUB_PRS = "false"

[onap]
GERRIT_HTTP_USER = "user1"
GERRIT_HTTP_PASSWORD = "${ENV:ONAP_GERRIT_TOKEN}"
REVIEWERS_EMAIL = "conf_override@example.org"
SUBMIT_SINGLE_COMMITS = "true"
"""
    cfg_file = tmp_path / "configuration.txt"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    # Point the loader to our temp config and select the org
    monkeypatch.setenv("G2G_CONFIG_PATH", str(cfg_file))
    monkeypatch.setenv("ORGANIZATION", "onap")
    # Token for ${ENV:ONAP_GERRIT_TOKEN}
    monkeypatch.setenv("ONAP_GERRIT_TOKEN", "sekret-token")
    # Pre-existing env should not be overridden by apply_config_to_env
    monkeypatch.setenv("REVIEWERS_EMAIL", "env_override@example.org")

    cfg = load_org_config(org="onap")
    # Expected normalized keys/values
    assert cfg["GERRIT_HTTP_USER"] == "user1"
    assert cfg["GERRIT_HTTP_PASSWORD"] == "sekret-token"
    assert cfg["SUBMIT_SINGLE_COMMITS"] == "true"
    # From [default]
    assert cfg["GERRIT_SERVER"] == "gerrit.example.org"
    assert cfg["GERRIT_SERVER_PORT"] == "29418"
    assert cfg["PRESERVE_GITHUB_PRS"] == "false"

    # Clear any pre-existing org-specific env so apply_config_to_env can set them
    monkeypatch.delenv("GERRIT_HTTP_USER", raising=False)
    monkeypatch.delenv("GERRIT_HTTP_PASSWORD", raising=False)
    monkeypatch.delenv("GERRIT_SERVER", raising=False)
    monkeypatch.delenv("GERRIT_SERVER_PORT", raising=False)
    monkeypatch.delenv("SUBMIT_SINGLE_COMMITS", raising=False)
    # Now apply to environment; REVIEWERS_EMAIL should stay as env value
    apply_config_to_env(cfg)
    assert os.getenv("GERRIT_HTTP_USER") == "user1"
    assert os.getenv("GERRIT_HTTP_PASSWORD") == "sekret-token"
    assert os.getenv("GERRIT_SERVER") == "gerrit.example.org"
    assert os.getenv("GERRIT_SERVER_PORT") == "29418"
    assert os.getenv("SUBMIT_SINGLE_COMMITS") == "true"
    # Pre-existing env var preserved
    assert os.getenv("REVIEWERS_EMAIL") == "env_override@example.org"


def test_enumerate_reviewer_emails_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify reviewer email enumeration from git config:
    - Reads from multiple keys
    - Splits comma-separated values
    - Deduplicates while preserving order
    - Falls back to user.email (local/global)
    """

    # Simulate git config getters used by enumeration without invoking git.
    def fake_get_all(key: str, *, global_: bool = False) -> list[str]:
        if key == "github2gerrit.reviewersEmail" and not global_:
            return ["a@example.org,b@example.org"]
        if key == "g2g.reviewersEmail" and not global_:
            return ["b@example.org", "c@example.org"]
        if key == "reviewers.email":
            # No entries here (local/global)
            return []
        return []

    def fake_get(key: str, *, global_: bool = False) -> str | None:
        # Only provide a global fallback for user.email
        if key == "user.email" and global_:
            return "d@example.org"
        return None

    monkeypatch.setattr(
        "github2gerrit.gitutils.git_config_get_all",
        fake_get_all,
        raising=True,
    )
    monkeypatch.setattr(
        "github2gerrit.gitutils.git_config_get",
        fake_get,
        raising=True,
    )

    emails = enumerate_reviewer_emails()
    # Expect order: a, b (from csv), then c; fallback user.email 'd'
    assert emails == [
        "a@example.org",
        "b@example.org",
        "c@example.org",
        "d@example.org",
    ]
