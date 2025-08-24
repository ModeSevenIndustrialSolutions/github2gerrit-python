# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import cast

import pytest

from github2gerrit import github_api as ghapi


def _placeholder_non_test() -> None:
    # Placeholder to avoid duplicate test name; no-op
    pass


def _wrap_retry(
    attempts: int,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    dec = ghapi._retry_on_github(attempts=attempts)
    return cast(Callable[[Callable[..., Any]], Callable[..., Any]], dec)


def test_retry_on_rate_limit_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(ghapi.time, "sleep", lambda s: sleeps.append(float(s)))

    attempts = {"n": 0}

    @_wrap_retry(attempts=3)
    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ghapi.RateLimitExceededExceptionType()
        return "ok"

    assert flaky() == "ok"
    # Should have retried at least once
    assert len(sleeps) >= 1
    # Exactly two calls: one fail + one success
    assert attempts["n"] == 2


def test_retry_on_5xx_github_exception_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(ghapi.time, "sleep", lambda s: sleeps.append(float(s)))

    class Dummy5xx(ghapi.GithubExceptionType):
        def __init__(self) -> None:
            super().__init__("server error")
            self.status = 503
            self.data = ""

    calls = {"n": 0}

    @_wrap_retry(attempts=3)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise Dummy5xx()
        return "ok"

    assert flaky() == "ok"
    assert len(sleeps) >= 1
    assert calls["n"] == 2


def test_retry_on_403_with_rate_limit_text_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(ghapi.time, "sleep", lambda s: sleeps.append(float(s)))

    class Dummy403(ghapi.GithubExceptionType):
        def __init__(self, data: Any) -> None:
            super().__init__("forbidden")
            self.status = 403
            self.data = data

    calls = {"n": 0}

    @_wrap_retry(attempts=3)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            # Both str and bytes are handled by the retry logic
            raise Dummy403("Rate limit exceeded")
        return "ok"

    assert flaky() == "ok"
    assert len(sleeps) >= 1
    assert calls["n"] == 2


def test_non_retryable_exception_bubbles_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure we do not sleep on non-retryable errors
    slept: list[float] = []
    monkeypatch.setattr(ghapi.time, "sleep", lambda s: slept.append(float(s)))

    class Dummy400(ghapi.GithubExceptionType):
        def __init__(self) -> None:
            super().__init__("bad request")
            self.status = 400
            self.data = ""

    @_wrap_retry(attempts=3)
    def bad() -> str:
        raise Dummy400()

    with pytest.raises(Dummy400):
        bad()
    assert slept == []


def test_retry_exhaustion_raises_last_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(ghapi.time, "sleep", lambda s: sleeps.append(float(s)))

    @_wrap_retry(attempts=2)
    def always_rate_limited() -> str:
        raise ghapi.RateLimitExceededExceptionType()

    with pytest.raises(ghapi.RateLimitExceededExceptionType):
        always_rate_limited()
    # With attempts=2, we sleep once
    assert len(sleeps) == 1


def test_build_client_raises_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure no token present
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ValueError):
        ghapi.build_client()


def test_build_client_raises_when_pygithub_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Provide a token to pass the token gate
    monkeypatch.setenv("GITHUB_TOKEN", "token-xyz")

    class NoGithub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyGithub is required to access the GitHub API")

    # Force the alias used by build_client to raise on construction
    monkeypatch.setattr(ghapi, "Github", NoGithub, raising=True)

    with pytest.raises(RuntimeError) as ei:
        ghapi.build_client()
    assert "PyGithub is required to access the GitHub API" in str(ei.value)
