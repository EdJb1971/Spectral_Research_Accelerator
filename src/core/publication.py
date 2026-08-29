"""Crash-safe publication of immutable evidence artifacts (D71).

Scientific receipts have two filesystem requirements that must hold at the same time:

* readers either see no target or the complete, flushed payload; and
* a publisher never replaces an existing target, including when writers race.

Python's :func:`os.replace` supplies the first property but deliberately violates the second.
An existence check followed by rename has a time-of-check/time-of-use race.  This module uses
the operating system's atomic *no-replace* namespace primitive instead: ``rename`` on Windows,
``renameat2(RENAME_NOREPLACE)`` on Linux, ``renamex_np(RENAME_EXCL)`` on macOS, and a hard-link
publication fallback on other POSIX filesystems.  A filesystem supporting none of those
operations is refused; immutability is never weakened to make a write succeed.

The temporary file is created beside the target, so publication never crosses filesystems.
The payload is flushed and ``fsync``'d before its name becomes visible.  This establishes
process-crash atomicity.  Persistence across sudden power loss additionally depends on the
filesystem and storage device's directory-entry durability guarantees; the API deliberately
does not claim more than the platform can prove.
"""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import sys
import tempfile
from typing import Union


PathLike = Union[str, os.PathLike[str]]

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_UNSUPPORTED_RENAME_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _raise_from_errno(result: int, source: Path, target: Path) -> None:
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(code, os.strerror(code), str(target))
    raise OSError(code, os.strerror(code), "%s -> %s" % (source, target))


def _linux_rename_noreplace(source: Path, target: Path) -> bool:
    """Use Linux's atomic no-replace rename; return False when unavailable to the filesystem."""
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError:
        return False
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(target), _RENAME_NOREPLACE)
    if result == 0:
        return True
    code = ctypes.get_errno()
    if code in _UNSUPPORTED_RENAME_ERRNOS:
        return False
    _raise_from_errno(result, source, target)
    raise AssertionError("unreachable")


def _darwin_rename_noreplace(source: Path, target: Path) -> bool:
    """Use macOS's atomic exclusive rename; return False when the API/filesystem lacks it."""
    try:
        function = ctypes.CDLL(None, use_errno=True).renamex_np
    except AttributeError:
        return False
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(os.fsencode(source), os.fsencode(target), _RENAME_EXCL)
    if result == 0:
        return True
    code = ctypes.get_errno()
    if code in _UNSUPPORTED_RENAME_ERRNOS:
        return False
    _raise_from_errno(result, source, target)
    raise AssertionError("unreachable")


def _publish_temporary_no_replace(source: Path, target: Path) -> str:
    """Expose ``source`` at ``target`` atomically, returning the primitive that succeeded."""
    if os.name == "nt":
        # Windows rename is a same-volume atomic operation and refuses an existing target.
        # Unlike hard links, it is supported by exFAT -- the deployed filesystem that found D71.
        os.rename(source, target)
        return "windows-rename-no-replace"

    if sys.platform.startswith("linux") and _linux_rename_noreplace(source, target):
        return "linux-renameat2-noreplace"
    if sys.platform == "darwin" and _darwin_rename_noreplace(source, target):
        return "darwin-renamex-noreplace"

    # Creating a hard link is one atomic directory operation and fails with EEXIST.  Removing
    # the temporary name afterwards cannot make the published target incomplete.
    os.link(source, target)
    return "posix-link-no-replace"


def publish_new_bytes(path: PathLike, payload: bytes, label: str = "artifact") -> str:
    """Atomically publish complete ``payload`` while refusing to replace ``path``.

    Returns the name of the OS primitive used, making the acceptance test and verification
    record explicit about which guarantee was exercised.  ``FileExistsError`` is normal control
    flow for an immutable target.  Other ``OSError`` instances mean the filesystem could not
    provide the contract and should be translated into the caller's domain-specific error.
    """
    if not isinstance(payload, bytes):
        raise TypeError("immutable publication payload must be bytes, got %s" % type(payload).__name__)
    if not isinstance(label, str) or not label.strip():
        raise ValueError("publication label must be a non-empty string")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            return _publish_temporary_no_replace(temporary, target)
        except FileExistsError:
            raise FileExistsError(
                "refusing to overwrite %s at %s" % (label, target)) from None
        except OSError as exc:
            raise OSError(
                exc.errno,
                "cannot atomically publish %s at %s without weakening no-overwrite semantics: %s"
                % (label, target, exc),
                str(target),
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
