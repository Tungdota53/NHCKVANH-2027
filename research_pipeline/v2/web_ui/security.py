"""Loopback enforcement, path-traversal/junction containment, and CSRF protection."""

import hmac
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import secrets
from urllib.parse import unquote

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def validate_loopback_host(host: str) -> str:
    value = (host or "").strip().lower()
    if value not in LOOPBACK_HOSTS:
        raise ValueError(
            f"Web UI binds strictly to loopback ({', '.join(sorted(LOOPBACK_HOSTS))}); got {host!r}"
        )
    return value


def _iterative_unquote(value: str, passes: int = 3) -> str:
    cur = str(value)
    for _ in range(passes):
        nxt = unquote(cur)
        if nxt == cur:
            break
        cur = nxt
    return cur


def is_raw_traversal(raw_path: str) -> bool:
    if raw_path is None:
        return True
    s = str(raw_path)
    if "\x00" in s:
        return True
    cur = s
    for _ in range(4):
        low = cur.lower()
        if "%2e%2e" in low or "..\\" in low or "../" in low or low == ".." or low.endswith("/..") or low.endswith("\\.."):
            return True
        if ".." in PurePosixPath(cur.replace("\\", "/")).parts or ".." in PureWindowsPath(cur).parts:
            return True
        nxt = unquote(cur)
        if nxt == cur:
            break
        cur = nxt
    return False


def resolve_contained_path(
    runs_root,
    raw_path: str,
    *,
    allow_absolute_inside: bool = True,
    must_exist: bool = False,
) -> Path:
    if raw_path is None or not str(raw_path).strip():
        raise PermissionError("Empty path is not allowed")
    raw_str = str(raw_path).strip()
    if is_raw_traversal(raw_str):
        raise PermissionError(f"Path traversal rejected: {raw_path!r}")
    decoded = _iterative_unquote(raw_str, passes=3).strip()
    if "\x00" in decoded:
        raise PermissionError("NUL byte rejected")
    if ".." in PurePosixPath(decoded.replace("\\", "/")).parts or ".." in PureWindowsPath(decoded).parts:
        raise PermissionError(f"Parent traversal rejected: {raw_path!r}")

    base = Path(runs_root).resolve()
    base_real = Path(os.path.realpath(str(base)))

    cand = Path(decoded)
    if cand.is_absolute():
        if not allow_absolute_inside:
            raise PermissionError(f"Absolute path not permitted: {raw_path!r}")
        target = cand.resolve()
    else:
        # Reject leading slash or Windows drive-relative paths when relative expected.
        if decoded.startswith(("/", "\\")) or PureWindowsPath(decoded).drive:
            raise PermissionError(f"Invalid relative path: {raw_path!r}")
        target = (base / decoded).resolve()

    target_real = Path(os.path.realpath(str(target)))
    if not target.is_relative_to(base) or not target_real.is_relative_to(base_real):
        raise PermissionError(f"Path escapes runs root: {raw_path!r}")
    if must_exist and not target.exists():
        raise FileNotFoundError(f"Path not found: {raw_path!r}")
    return target


def resolve_dataset_input(raw_path: str) -> Path:
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("Dataset path is required")
    raw_str = str(raw_path).strip()
    if is_raw_traversal(raw_str):
        raise PermissionError(f"Path traversal rejected in dataset path: {raw_path!r}")
    decoded = _iterative_unquote(raw_str, passes=3).strip()
    if "\x00" in decoded:
        raise PermissionError("NUL byte rejected")
    target = Path(decoded).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Dataset path not found: {raw_path!r}")
    return target


class CSRFManager:
    def __init__(self, token: str | None = None):
        self.token = token or secrets.token_urlsafe(32)

    def verify(self, provided: str | None) -> bool:
        if not provided or not isinstance(provided, str):
            return False
        return hmac.compare_digest(self.token, provided.strip())
