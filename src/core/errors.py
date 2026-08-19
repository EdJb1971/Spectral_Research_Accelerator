"""Explanatory error taxonomy (roadmap T3.5.14, defect D14, standard E6).

An error message is a user interface. The previous code raised bare `ValueError`s and the
API turned every one of them into `500 Internal Server Error` with the detail
*"An internal error occurred"* - which tells a researcher nothing, and misreports their typo
as our fault. Three specific properties are required here:

**1. Say what to do, not only what happened.** "Unknown transform 'dwt2'" is a diagnosis;
"Unknown transform 'dwt2'; did you mean 'dwt'? Available: dct, dtcwt, dwt, fft, hybrid, swt"
is a fix. Every error in this module carries the valid alternatives when it can enumerate
them, and a `did you mean` suggestion when one is close.

**2. Distinguish user error from internal fault.** A bad parameter is `4xx` and should be
shown to the caller verbatim; a genuine internal fault is `5xx` and must *not* leak a stack
trace. That decision belongs to the exception type, not to a `try/except` at the call site
guessing after the fact.

**3. Carry structured context.** A pipeline failure that says "shape mismatch" is useless in
a 20-run sweep. It has to name the step, the action, the parameter combination and both
shapes, so the failing configuration can be found without re-running anything.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _suggest(name: str, options: Iterable[str], n: int = 2) -> str:
    """`did you mean` clause, or empty. Uses edit distance rather than prefix matching."""
    close = difflib.get_close_matches(str(name), [str(o) for o in options], n=n, cutoff=0.6)
    if not close:
        return ""
    if len(close) == 1:
        return " Did you mean %r?" % close[0]
    return " Did you mean %s?" % " or ".join(repr(c) for c in close)


def _available(options: Iterable[str], limit: int = 24) -> str:
    opts = sorted(str(o) for o in options)
    if not opts:
        return " Nothing is registered under that kind yet."
    shown = opts[:limit]
    tail = "" if len(opts) <= limit else ", ... (%d more)" % (len(opts) - limit)
    return " Available: %s%s." % (", ".join(shown), tail)


class SpectralEarthError(Exception):
    """Base class. Carries structured context and an HTTP status classification.

    ``status_code`` is a property of the *error kind*: a malformed request is 400 wherever
    it is raised, and an internal fault is 500 wherever it is raised. Deciding this at the
    raise site rather than in the API layer is what stops user mistakes being reported as
    server failures.
    """

    status_code: int = 500
    #: Whether the message is safe to return to an API client verbatim.
    client_safe: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = {k: v for k, v in context.items() if v is not None}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": self.message,
            "context": self.context,
        }

    def api_detail(self) -> str:
        """What the API should return. Internal faults are deliberately opaque."""
        if self.client_safe:
            return self.message
        return ("An internal error occurred. The failure has been logged with its full "
                "context; quote this error type when reporting it: %s" % type(self).__name__)


# --------------------------------------------------------------------- user errors

class UserInputError(SpectralEarthError):
    """The caller asked for something impossible. 4xx, and safe to echo back."""

    status_code = 400
    client_safe = True


class UnknownNameError(UserInputError):
    """A registry lookup failed. Always names the alternatives."""

    status_code = 404

    def __init__(self, kind: str, name: Any, options: Sequence[str],
                 **context: Any) -> None:
        message = ("Unknown %s %r.%s%s" % (kind, name, _suggest(name, options),
                                           _available(options)))
        super().__init__(message, kind=kind, name=name,
                         available=sorted(str(o) for o in options), **context)
        self.kind = kind
        self.name = name
        self.options = list(options)


class DuplicateRegistrationError(SpectralEarthError):
    """Registering the same name twice. A programming error, hence 500.

    Not a silent overwrite: a second registration would make the first implementation
    disappear at import time depending on module load order, which is close to
    undiagnosable.
    """

    status_code = 500

    def __init__(self, kind: str, name: str, existing: str) -> None:
        super().__init__(
            "%s %r is already registered (by %s). Registry names appear in the API and in "
            "experiment configs, so a silent overwrite would change behaviour depending on "
            "import order. Pick a different name, or call `replace()` if you genuinely mean "
            "to override it." % (kind.capitalize(), name, existing),
            kind=kind, name=name, existing=existing)


class InvalidParameterError(UserInputError):
    """A parameter is present but wrong."""

    def __init__(self, parameter: str, value: Any, expected: str, **context: Any) -> None:
        super().__init__(
            "Parameter %r = %r is invalid: expected %s." % (parameter, value, expected),
            parameter=parameter, value=repr(value), expected=expected, **context)


class MissingParameterError(UserInputError):
    """A required parameter is absent."""

    def __init__(self, parameter: str, action: Optional[str] = None,
                 required: Optional[Sequence[str]] = None, **context: Any) -> None:
        where = " for %r" % action if action else ""
        req = (" Required parameters%s: %s." % (where, ", ".join(sorted(required)))
               if required else "")
        super().__init__(
            "Missing required parameter %r%s.%s" % (parameter, where, req),
            parameter=parameter, action=action,
            required=sorted(required) if required else None, **context)


class ShapeMismatchError(UserInputError):
    """Two things that had to agree did not. Names both, and the fix."""

    def __init__(self, what_a: str, shape_a: Any, what_b: str, shape_b: Any,
                 fix: Optional[str] = None, **context: Any) -> None:
        suggestion = fix or (
            "Insert a resampling step, or make the two steps read the same source field.")
        super().__init__(
            "Shape mismatch: %s is %s but %s is %s. They must match. %s"
            % (what_a, tuple(shape_a), what_b, tuple(shape_b), suggestion),
            what_a=what_a, shape_a=list(shape_a), what_b=what_b, shape_b=list(shape_b),
            fix=suggestion, **context)


class FieldTooSmallError(UserInputError):
    """A transform needs more samples than the field has. Names the minimum."""

    def __init__(self, operation: str, shape: Any, minimum: Any,
                 remedy: Optional[str] = None, **context: Any) -> None:
        super().__init__(
            "%s needs at least %s samples but the field is %s. %s"
            % (operation, minimum, tuple(shape),
               remedy or "Reduce the number of levels, or use a larger crop."),
            operation=operation, shape=list(shape), minimum=minimum, **context)


# --------------------------------------------------------------------- pipeline

class PipelineStepError(SpectralEarthError):
    """A step in a declarative pipeline failed. Names the step and its configuration.

    Wraps the underlying cause and inherits its status: a bad parameter inside a step is
    still the caller's mistake, and reporting it as a 500 would send a researcher looking
    for a platform bug that is not there.
    """

    def __init__(self, step_name: str, action: str, cause: BaseException,
                 params: Optional[Dict[str, Any]] = None,
                 run_index: Optional[int] = None, **context: Any) -> None:
        underlying = getattr(cause, "message", None) or str(cause)
        where = " (run %d)" % run_index if run_index is not None else ""
        super().__init__(
            "Step %r%s, action %r, failed: %s" % (step_name, where, action, underlying),
            step=step_name, action=action, params=params, run_index=run_index,
            cause_type=type(cause).__name__, **context)
        self.cause = cause
        self.status_code = getattr(cause, "status_code", 500)
        self.client_safe = getattr(cause, "client_safe", False)


class ReferenceResolutionError(UserInputError):
    """A `{step.key}` reference in a config could not be resolved."""

    def __init__(self, reference: str, available: Sequence[str], **context: Any) -> None:
        super().__init__(
            "Could not resolve reference %r.%s%s" % (
                reference, _suggest(reference, available),
                " Resolvable so far: %s." % ", ".join(sorted(available)) if available
                else " No earlier step has produced a value yet - a reference can only "
                     "point at a step that runs before it."),
            reference=reference, available=sorted(available), **context)


# --------------------------------------------------------------------- data

class DataSourceError(SpectralEarthError):
    """A data source could not satisfy a request."""

    status_code = 502
    client_safe = True


class AllSourcesFailedError(DataSourceError):
    """Every source in the fallback chain failed. Reports what each one said.

    Standard E2: a fallback is a provenance fact. Collapsing four different failures into
    "could not load data" throws away exactly the information needed to fix any of them.
    """

    status_code = 502

    def __init__(self, dataset_id: str, attempts: List[Dict[str, Any]]) -> None:
        lines = ["  - %s: %s" % (a.get("source", "?"), a.get("reason", "?"))
                 for a in attempts]
        super().__init__(
            "No source could provide %r. Tried %d:\n%s"
            % (dataset_id, len(attempts), "\n".join(lines)),
            dataset_id=dataset_id, attempts=attempts)


def classify(exc: BaseException) -> Dict[str, Any]:
    """Map any exception onto an API status and a safe detail string.

    Non-`SpectralEarthError` exceptions are treated as internal faults, which is the
    conservative default: an unrecognised error might carry a file path or a query in its
    message, so it is logged in full and reported opaquely.
    """
    if isinstance(exc, SpectralEarthError):
        return {"status_code": exc.status_code, "detail": exc.api_detail(),
                "error": type(exc).__name__, "context": exc.context}
    return {"status_code": 500,
            "detail": "An internal error occurred while processing the request.",
            "error": type(exc).__name__, "context": {}}
