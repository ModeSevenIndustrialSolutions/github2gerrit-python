# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation
#
# Configuration loader for github2gerrit.
#
# This module provides a simple INI-based configuration system that lets
# you define per-organization settings in a file such as:
#
#   ~/.config/github2gerrit/configuration.txt
#
# Example:
#
#   [default]
#   GERRIT_SERVER = "gerrit.example.org"
#   GERRIT_SERVER_PORT = "29418"
#
#   [onap]
#   GERRIT_HTTP_USER = "modesevenindustrialsolutions"
#   GERRIT_HTTP_PASSWORD = "my_gerrit_token"
#   GERRIT_PROJECT = "integration/test-repo"
#   REVIEWERS_EMAIL = "a@example.org,b@example.org"
#   PRESERVE_GITHUB_PRS = "true"
#
# Values are returned as strings with surrounding quotes stripped.
# Boolean-like values are normalized to "true"/"false" strings.
# You can reference environment variables using ${ENV:VAR_NAME}.
#
# Precedence model (recommended):
#   - CLI flags (highest)
#   - Environment variables
#   - Config file values (loaded by this module)
#   - Tool defaults (lowest)
#
# Callers can:
#   - load_org_config() to retrieve a dict of key->value (strings)
#   - apply_config_to_env() to export values to process environment for
#     any keys not already set by the environment/runner
#
# Notes:
#   - Section names are matched case-insensitively.
#   - If no organization is provided, we try ORGANIZATION, then
#     GITHUB_REPOSITORY_OWNER from the environment.
#   - A [default] section can provide baseline values for all orgs.
#   - Unknown keys are preserved (uppercased) to keep this future-proof.

from __future__ import annotations

import configparser
import logging
import os
import re
from pathlib import Path
from typing import Any
from typing import cast


log = logging.getLogger("github2gerrit.config")

DEFAULT_CONFIG_PATH = "~/.config/github2gerrit/configuration.txt"

# Recognized keys. Unknown keys are still passed through, but these are
# the most relevant to current tooling and action inputs/options.
KNOWN_KEYS: set[str] = {
    # Action inputs
    "SUBMIT_SINGLE_COMMITS",
    "USE_PR_AS_COMMIT",
    "FETCH_DEPTH",
    "GERRIT_KNOWN_HOSTS",
    "GERRIT_SSH_PRIVKEY_G2G",
    "GERRIT_SSH_USER_G2G",
    "GERRIT_SSH_USER_G2G_EMAIL",
    "ORGANIZATION",
    "REVIEWERS_EMAIL",
    "PR_NUMBER",
    "SYNC_ALL_OPEN_PRS",
    "PRESERVE_GITHUB_PRS",
    # Optional inputs (reusable workflow compatibility)
    "GERRIT_SERVER",
    "GERRIT_SERVER_PORT",
    "GERRIT_HTTP_BASE_PATH",
    "GERRIT_PROJECT",
    # Gerrit REST auth
    "GERRIT_HTTP_USER",
    "GERRIT_HTTP_PASSWORD",
}

_ENV_REF = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_refs(value: str) -> str:
    """Expand ${ENV:VAR} references using current environment."""

    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        return os.getenv(var, "") or ""

    return _ENV_REF.sub(repl, value)


def _strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _normalize_bool_like(value: str) -> str | None:
    """Return 'true'/'false' for boolean-like values, else None."""
    s = value.strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return "true"
    if s in {"0", "false", "no", "off"}:
        return "false"
    return None


def _coerce_value(raw: str) -> str:
    """Coerce a raw string to normalized representation."""
    expanded = _expand_env_refs(raw)
    unquoted = _strip_quotes(expanded)
    # Normalize escaped newline sequences into real newlines so that values
    # like SSH keys or known_hosts entries can be specified inline using
    # '\n' or '\r\n' in configuration files.
    normalized_newlines = (
        unquoted.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\r\n", "\n")
    )
    b = _normalize_bool_like(normalized_newlines)
    return b if b is not None else normalized_newlines


def _select_section(
    cp: configparser.RawConfigParser,
    org: str,
) -> str | None:
    """Find a section name case-insensitively."""
    target = org.strip().lower()
    for sec in cp.sections():
        if sec.strip().lower() == target:
            return sec
    return None


def _load_ini(path: Path) -> configparser.RawConfigParser:
    cp = configparser.RawConfigParser()
    # Preserve option case; mypy requires a cast for attribute assignment
    cast(Any, cp).optionxform = str
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw_text = fh.read()
        # Pre-process simple multi-line quoted values of the form:
        #   key = "
        #   line1
        #   line2
        #   "
        # We collapse these into a single line with '\n' escapes so that
        # configparser can ingest them reliably; later, _coerce_value()
        # converts the escapes back to real newlines.
        lines = raw_text.splitlines()
        out_lines: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            eq_idx = line.find("=")
            if eq_idx != -1:
                left = line[: eq_idx + 1]
                rhs = line[eq_idx + 1 :].strip()
                if rhs == '"':
                    i += 1
                    block: list[str] = []
                    # Collect until a line with only a closing quote
                    # (ignoring spaces)
                    while i < len(lines) and lines[i].strip() != '"':
                        block.append(lines[i])
                        i += 1
                    if i < len(lines) and lines[i].strip() == '"':
                        joined = "\\n".join(block)
                        out_lines.append(f'{left} "{joined}"')
                        i += 1
                        continue
                    else:
                        # No closing quote found; fall through
                        # and keep original line
                        out_lines.append(line)
                        continue
            out_lines.append(line)
            i += 1
        preprocessed = "\n".join(out_lines) + ("\n" if out_lines else "")
        cp.read_string(preprocessed)
    except FileNotFoundError:
        log.debug("Config file not found: %s", path)
    except Exception as exc:
        log.warning("Failed to read config file %s: %s", path, exc)
    return cp


def _detect_org() -> str | None:
    # Prefer explicit ORGANIZATION, then GitHub default env var
    org = os.getenv("ORGANIZATION", "").strip()
    if org:
        return org
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "").strip()
    return owner or None


def _merge_dicts(
    base: dict[str, str],
    override: dict[str, str],
) -> dict[str, str]:
    out = dict(base)
    out.update(override)
    return out


def _normalize_keys(d: dict[str, str]) -> dict[str, str]:
    return {k.strip().upper(): v for k, v in d.items() if k.strip()}


def load_org_config(
    org: str | None = None,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Load configuration for a GitHub organization.

    Args:
      org:
        Name of the GitHub org (stanza). If not provided, inferred from
        ORGANIZATION or GITHUB_REPOSITORY_OWNER environment variables.
      path:
        Path to the INI file. If not provided, uses:
        ~/.config/github2gerrit/configuration.txt
        If G2G_CONFIG_PATH is set, it takes precedence.

    Returns:
      A dict mapping KEY -> value (strings). Unknown keys are preserved,
      known boolean-like values are normalized to 'true'/'false', quotes
      are stripped, and ${ENV:VAR} are expanded.
    """
    if path is None:
        path = os.getenv("G2G_CONFIG_PATH", "").strip() or DEFAULT_CONFIG_PATH
    cfg_path = Path(path).expanduser()

    cp = _load_ini(cfg_path)
    effective_org = org or _detect_org()
    result: dict[str, str] = {}

    # Start with [default]
    if cp.has_section("default"):
        for k, v in cp.items("default"):
            result[k.strip().upper()] = _coerce_value(v)

    # Overlay with [org] if present
    if effective_org:
        chosen = _select_section(cp, effective_org)
        if chosen:
            for k, v in cp.items(chosen):
                result[k.strip().upper()] = _coerce_value(v)
        else:
            log.debug(
                "Org section '%s' not found in %s",
                effective_org,
                cfg_path,
            )

    normalized = _normalize_keys(result)
    return normalized


def apply_config_to_env(cfg: dict[str, str]) -> None:
    """Set environment variables for any keys not already set.

    This is useful to make configuration values visible to downstream
    code that reads via os.environ, while still letting explicit env
    or CLI flags take precedence.

    We only set keys that are not already present in the environment.
    """
    for k, v in cfg.items():
        if (os.getenv(k) or "").strip() == "":
            os.environ[k] = v


def filter_known(
    cfg: dict[str, str],
    include_extra: bool = True,
) -> dict[str, str]:
    """Return a filtered view of cfg.

    If include_extra is False, only keys from KNOWN_KEYS are included.
    If True (default), all keys are included.
    """
    if include_extra:
        return dict(cfg)
    return {k: v for k, v in cfg.items() if k in KNOWN_KEYS}


def overlay_missing(
    primary: dict[str, str],
    fallback: dict[str, str],
) -> dict[str, str]:
    """Merge fallback into primary for any missing keys.

    This is a helper when composing precedence:
      merged = overlay_missing(env_view, config_view)
    """
    merged = dict(primary)
    for k, v in fallback.items():
        if k not in merged or merged[k] == "":
            merged[k] = v
    return merged
