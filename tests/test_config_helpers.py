# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import os
from pathlib import Path

import pytest

from github2gerrit.config import _coerce_value
from github2gerrit.config import _expand_env_refs
from github2gerrit.config import _normalize_bool_like
from github2gerrit.config import _select_section
from github2gerrit.config import _strip_quotes
from github2gerrit.config import apply_config_to_env
from github2gerrit.config import filter_known
from github2gerrit.config import load_org_config
from github2gerrit.config import overlay_missing


def test_expand_env_refs_expands_present_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOO", "bar")
    monkeypatch.setenv("NUM", "123")
    value = "prefix-${ENV:FOO}-x-${ENV:NUM}-suffix"
    assert _expand_env_refs(value) == "prefix-bar-x-123-suffix"


def test_expand_env_refs_missing_vars_become_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING", raising=False)
    value = "start-${ENV:MISSING}-end"
    assert _expand_env_refs(value) == "start--end"


@pytest.mark.parametrize(  # type: ignore[misc]
    "raw, expected",
    [
        ('"hello"', "hello"),
        ("'hello'", "hello"),
        ('  " spaced "  ', " spaced "),
        ("noquotes", "noquotes"),
        ("", ""),
    ],
)
def test_strip_quotes_various_forms(raw: str, expected: str) -> None:
    assert _strip_quotes(raw) == expected


@pytest.mark.parametrize(  # type: ignore[misc]
    "raw, expected",
    [
        ("true", "true"),
        ("TRUE", "true"),
        ("Yes", "true"),
        ("on", "true"),
        ("1", "true"),
        ("false", "false"),
        ("FALSE", "false"),
        ("No", "false"),
        ("off", "false"),
        ("0", "false"),
        ("maybe", None),
        ("", None),
        ("  y  ", None),
    ],
)
def test_normalize_bool_like(raw: str, expected: str | None) -> None:
    assert _normalize_bool_like(raw) == expected


def test_coerce_value_handles_quotes_bools_and_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Quoted string with escaped newlines -> real newlines, quotes stripped
    multi = '"Line1\\nLine2\\nLine3"'
    assert _coerce_value(multi) == "Line1\nLine2\nLine3"

    # Mixed CRLF/escaped newlines normalized to LF
    mixed = '"A\\r\\nB\\nC\\r\\n"'
    assert _coerce_value(mixed) == "A\nB\nC\n"

    # Boolean-like normalization
    assert _coerce_value(" TRUE ") == "true"
    assert _coerce_value("no") == "false"

    # Environment expansion before quote stripping
    monkeypatch.setenv("TOKEN", "sekret")
    conf = '"Bearer ${ENV:TOKEN}"'
    assert _coerce_value(conf) == "Bearer sekret"


def test_select_section_case_insensitive() -> None:
    import configparser
    from typing import Any
    from typing import cast

    cp = configparser.RawConfigParser()
    cast(Any, cp).optionxform = str  # preserve key case
    cp.read_string(
        """
[Default]
A = 1
[MyOrg]
B = 2
"""
    )
    assert _select_section(cp, "myorg") == "MyOrg"
    assert _select_section(cp, "MYORG") == "MyOrg"
    assert _select_section(cp, "unknown") is None


def test_load_org_config_merges_default_and_org_and_normalizes(
    tmp_path: Path,
) -> None:
    cfg_text = """
[default]
GERRIT_SERVER = "gerrit.example.org"
PRESERVE_GITHUB_PRS = "false"

[OnAp]
GERRIT_HTTP_USER = "user1"
GERRIT_HTTP_PASSWORD = "${ENV:ONAP_GERRIT_TOKEN}"
SSH_BLOCK = "
-----BEGIN KEY-----
abc
-----END KEY-----
"
SUBMIT_SINGLE_COMMITS = "YES"
"""
    cfg_file = tmp_path / "configuration.txt"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    # Provide token referenced via ${ENV:ONAP_GERRIT_TOKEN}
    os.environ["ONAP_GERRIT_TOKEN"] = "sekret-token"

    cfg = load_org_config(org="onap", path=cfg_file)

    # From default
    assert cfg["GERRIT_SERVER"] == "gerrit.example.org"
    # Bool normalized
    assert cfg["PRESERVE_GITHUB_PRS"] == "false"
    # Org values
    assert cfg["GERRIT_HTTP_USER"] == "user1"
    assert cfg["GERRIT_HTTP_PASSWORD"] == "sekret-token"
    # Multiline quoted block round-tripped to real newlines and quotes stripped
    assert cfg["SSH_BLOCK"] == "-----BEGIN KEY-----\nabc\n-----END KEY-----"
    # Bool-like normalization in org section
    assert cfg["SUBMIT_SINGLE_COMMITS"] == "true"


def test_load_org_config_uses_env_detected_org_and_path_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_text = """
[default]
A = "x"
[Acme]
A = "y"
B = "z"
"""
    cfg_file = tmp_path / "conf.ini"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    monkeypatch.setenv("ORGANIZATION", "ACME")
    monkeypatch.setenv("G2G_CONFIG_PATH", str(cfg_file))

    cfg = load_org_config()

    assert cfg["A"] == "y"  # org overlays default
    assert cfg["B"] == "z"


def test_apply_config_to_env_sets_only_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pre-populate env with an existing value
    monkeypatch.setenv("EXISTING_KEY", "keepme")

    cfg = {
        "EXISTING_KEY": "donotoverride",
        "NEW_KEY": "setme",
        "ANOTHER": "value",
    }
    apply_config_to_env(cfg)

    assert os.getenv("EXISTING_KEY") == "keepme"
    assert os.getenv("NEW_KEY") == "setme"
    assert os.getenv("ANOTHER") == "value"


def test_filter_known_with_and_without_extras() -> None:
    sample = {
        "SUBMIT_SINGLE_COMMITS": "true",  # known
        "REVIEWERS_EMAIL": "a@example.org",  # known
        "EXTRA_OPTION": "42",  # unknown
    }
    # include_extra=True returns all keys
    out_all = filter_known(sample, include_extra=True)
    assert out_all == sample

    # include_extra=False filters out unknown keys
    out_known = filter_known(sample, include_extra=False)
    assert "SUBMIT_SINGLE_COMMITS" in out_known
    assert "REVIEWERS_EMAIL" in out_known
    assert "EXTRA_OPTION" not in out_known


def test_overlay_missing_prefers_primary_and_fills_empty_strings() -> None:
    primary = {
        "A": "1",
        "B": "",
        "C": "keep",
    }
    fallback = {
        "B": "2",  # should fill because primary["B"] == ""
        "C": "override",  # should NOT override
        "D": "added",  # new key
    }
    merged = overlay_missing(primary, fallback)
    assert merged["A"] == "1"
    assert merged["B"] == "2"
    assert merged["C"] == "keep"
    assert merged["D"] == "added"
