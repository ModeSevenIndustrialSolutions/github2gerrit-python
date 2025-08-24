# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import os
from typing import Any

import pytest

from github2gerrit.core import GerritInfo
from github2gerrit.core import Orchestrator
from github2gerrit.core import RepoNames


class _DummyHTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = type("Resp", (), {"status_code": status})()


def test_query_gerrit_results_success_base_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """
    When Gerrit REST at the base URL returns data successfully, _query_gerrit_for_results
    should parse and return URLs, numbers, and shas, and export env vars accordingly.
    """

    # Ensure a clean env for outputs and base path
    for k in (
        "GERRIT_CHANGE_REQUEST_URL",
        "GERRIT_CHANGE_REQUEST_NUM",
        "GERRIT_COMMIT_SHA",
    ):
        monkeypatch.delenv(k, raising=False)
    # No base path necessary for success; ensure unset
    monkeypatch.delenv("GERRIT_HTTP_BASE_PATH", raising=False)
    # No auth provided for this test
    monkeypatch.delenv("GERRIT_HTTP_USER", raising=False)
    monkeypatch.delenv("GERRIT_HTTP_PASSWORD", raising=False)

    # Dummy REST client that always succeeds regardless of URL
    class DummyRest:
        def __init__(self, url: str, auth: Any | None = None) -> None:
            self.url = url
            self.auth = auth

        def get(self, path: str) -> list[dict[str, Any]]:
            # Return a single change with a number and a current revision
            return [
                {
                    "_number": 12345,
                    "current_revision": "deadbeefcafebabe1234abcd5678ef90aabbccdd",
                }
            ]

    monkeypatch.setattr(
        "github2gerrit.core.GerritRestAPI", DummyRest, raising=True
    )

    orch = Orchestrator(workspace=tmp_path_factory.mktemp("repo"))
    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="releng/builder"
    )
    repo = RepoNames(
        project_gerrit="releng/builder", project_github="releng-builder"
    )

    # Provide a single valid Change-Id
    change_ids: list[str] = ["Iabc123"]
    result = orch._query_gerrit_for_results(
        gerrit=gerrit, repo=repo, change_ids=change_ids
    )

    assert result.change_numbers == ["12345"]
    assert result.commit_shas == ["deadbeefcafebabe1234abcd5678ef90aabbccdd"]
    assert result.change_urls == [
        "https://gerrit.example.org/c/releng/builder/+/12345"
    ]

    # Env exports should be set
    assert (
        os.getenv("GERRIT_CHANGE_REQUEST_URL")
        == "https://gerrit.example.org/c/releng/builder/+/12345"
    )
    assert os.getenv("GERRIT_CHANGE_REQUEST_NUM") == "12345"
    assert (
        os.getenv("GERRIT_COMMIT_SHA")
        == "deadbeefcafebabe1234abcd5678ef90aabbccdd"
    )


def test_query_gerrit_results_fallback_to_r_on_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """
    When Gerrit REST at the base URL raises a 404, the orchestrator should retry with
    '/r' base path and parse results successfully.
    """

    # Clean environment outputs and ensure no explicit base path is configured
    for k in (
        "GERRIT_CHANGE_REQUEST_URL",
        "GERRIT_CHANGE_REQUEST_NUM",
        "GERRIT_COMMIT_SHA",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("GERRIT_HTTP_BASE_PATH", raising=False)
    monkeypatch.delenv("GERRIT_HTTP_USER", raising=False)
    monkeypatch.delenv("GERRIT_HTTP_PASSWORD", raising=False)

    # Dummy REST client that raises 404 for the base URL and succeeds for '/r/'
    class DummyRest:
        def __init__(self, url: str, auth: Any | None = None) -> None:
            self.url = url
            self.auth = auth

        def get(self, path: str) -> list[dict[str, Any]]:
            # If the base URL ends with '/r/', simulate success; else raise 404 to trigger fallback
            if self.url.rstrip("/").endswith("/r"):
                return [
                    {
                        "_number": 6789,
                        "current_revision": "cafebabedeadbeef1234567890abcdef12345678",
                    }
                ]
            raise _DummyHTTPError(status=404)

    monkeypatch.setattr(
        "github2gerrit.core.GerritRestAPI", DummyRest, raising=True
    )

    orch = Orchestrator(workspace=tmp_path_factory.mktemp("repo2"))
    gerrit = GerritInfo(
        host="gerrit.example.org", port=29418, project="platform/infra"
    )
    repo = RepoNames(
        project_gerrit="platform/infra", project_github="platform-infra"
    )

    change_ids: list[str] = ["Ifallback1"]
    result = orch._query_gerrit_for_results(
        gerrit=gerrit, repo=repo, change_ids=change_ids
    )

    assert result.change_numbers == ["6789"]
    assert result.commit_shas == ["cafebabedeadbeef1234567890abcdef12345678"]
    assert result.change_urls == [
        "https://gerrit.example.org/c/platform/infra/+/6789"
    ]

    assert (
        os.getenv("GERRIT_CHANGE_REQUEST_URL")
        == "https://gerrit.example.org/c/platform/infra/+/6789"
    )
    assert os.getenv("GERRIT_CHANGE_REQUEST_NUM") == "6789"
    assert (
        os.getenv("GERRIT_COMMIT_SHA")
        == "cafebabedeadbeef1234567890abcdef12345678"
    )
