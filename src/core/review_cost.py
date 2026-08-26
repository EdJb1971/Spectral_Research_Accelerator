"""TG7.3: cost control for the recorded adversarial-review layer.

This module does two deliberately separate jobs.

``GeminiBatchTransport`` is the first concrete provider transport for TG7.1.  It maps the
provider-neutral ``CallRequest`` record onto Gemini's asynchronous Batch GenerateContent API,
requests the already-declared JSON schema, polls one inlined request to completion, and maps the
provider response back onto TG7.1's transport contract.  The API key is a constructor secret: it
is sent only in the ``x-goog-api-key`` header and is never placed in a request, usage record,
exception, digest or repr.

``audit_review_cost`` is provider-neutral.  It accepts a review only when every call followed a
predeclared model/effort route, used the required service mode, recorded finite token counts, and
the review as a whole contains a measured cache hit.  A cache *configuration* is not evidence of
a hit.  The resulting immutable receipt binds the audit to the exact review-record digest.

No price is baked into the receipt.  Prices change; a historical token and route receipt does
not.  The policy records the reviewed provider documentation and the declared Batch discount,
while monetary reconstruction belongs to a dated price table outside this scientific record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import httpx

from src.core.errors import DataSourceError, InvalidParameterError
from src.core.recorded_call import EFFORTS, REVIEW_ROLES, ReviewedBundle
from src.core.round_robin import PanelSeat, ReviewPanel


COST_POLICY_SCHEMA = "review-cost-policy/v1"
COST_RECEIPT_SCHEMA = "review-cost-receipt/v1"
GEMINI_PROVIDER = "google-gemini"
GEMINI_35_FLASH = "gemini-3.5-flash"
GEMINI_BATCH_MODE = "batch"
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_CACHE_MINIMUM_TOKENS = 4096

GEMINI_MODEL_SOURCE = "https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5"
GEMINI_CACHE_SOURCE = "https://ai.google.dev/gemini-api/docs/caching"
GEMINI_BATCH_SOURCE = "https://ai.google.dev/api/batch-api"
GEMINI_PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
DOCUMENTATION_REVIEWED_AT = "2026-08-26T00:00:00+12:00"

_TERMINAL_BATCH_STATES = {
    "BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}
_USAGE_FIELDS = (
    "provider", "service_mode", "input_tokens", "output_tokens", "cached_input_tokens",
    "total_tokens", "batch_name", "provider_usage",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("cost-control value", type(value).__name__,
                                    "finite canonical JSON data") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParameterError(name, value, "a non-empty string")
    return value.strip()


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidParameterError(name, value, "a non-negative integer token count")
    return value


def _exact(value: Any, fields: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidParameterError(name, type(value).__name__, "a JSON object")
    missing = sorted(set(fields) - set(value))
    extra = sorted(set(value) - set(fields))
    if missing or extra:
        raise InvalidParameterError(name, {"missing": missing, "extra": extra},
                                    "exactly the versioned cost-control fields")
    return value


def _json_schema(request: Mapping[str, Any]) -> Mapping[str, Any]:
    response_schema = request.get("response_schema")
    if not isinstance(response_schema, Mapping):
        raise InvalidParameterError("request.response_schema", response_schema,
                                    "the TG7.1 JSON-schema mapping")
    schema = response_schema.get("schema")
    if not isinstance(schema, Mapping):
        raise InvalidParameterError("request.response_schema.schema", schema,
                                    "a JSON Schema object")
    return _thaw(schema)


def build_gemini_batch_request(request: Mapping[str, Any], *,
                               cached_content: Optional[str] = None) -> Dict[str, Any]:
    """Translate one TG7.1 request into one inlined Gemini Batch API request."""
    model = _nonempty("request.model_id", request.get("model_id"))
    if model != GEMINI_35_FLASH:
        raise InvalidParameterError("request.model_id", model, GEMINI_35_FLASH)
    effort = request.get("effort")
    if effort not in EFFORTS:
        raise InvalidParameterError("request.effort", effort,
                                    "one of %s" % ", ".join(EFFORTS))
    request_sha256 = _nonempty("request.request_sha256", request.get("request_sha256"))
    system = _nonempty("request.system", request.get("system"))
    instruction = _nonempty("request.instruction", request.get("instruction"))
    context = request.get("context")
    if not isinstance(context, Mapping):
        raise InvalidParameterError("request.context", type(context).__name__, "a JSON object")

    prompt = json.dumps({"context": _thaw(context), "instruction": instruction},
                        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    generation: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": effort},
            "responseMimeType": "application/json",
            "responseJsonSchema": _json_schema(request),
        },
    }
    if cached_content is not None:
        cache_name = _nonempty("cached_content", cached_content)
        if not cache_name.startswith("cachedContents/"):
            raise InvalidParameterError("cached_content", cache_name,
                                        "a Gemini resource named cachedContents/{id}")
        generation["cachedContent"] = cache_name

    return {
        "batch": {
            "displayName": "spectralearth-%s" % request_sha256[:20],
            "inputConfig": {"requests": {"requests": [{
                "request": generation,
                "metadata": {"request_sha256": request_sha256},
            }]}}
        }
    }


def _batch(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap either an Operation response or a direct GenerateContentBatch mapping."""
    if not isinstance(value, Mapping):
        raise DataSourceError("Gemini Batch API returned a non-object response")
    response = value.get("response")
    if isinstance(response, Mapping):
        return response
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping) and ("state" in metadata or "output" in metadata):
        return metadata
    return value


def _gemini_text(response: Mapping[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise DataSourceError("Gemini returned no unique structured-output candidate")
    content = candidates[0].get("content") if isinstance(candidates[0], Mapping) else None
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        raise DataSourceError("Gemini candidate contains no response parts")
    texts = [part.get("text") for part in parts
             if isinstance(part, Mapping) and isinstance(part.get("text"), str)]
    raw = "".join(texts).strip()
    if not raw:
        raise DataSourceError("Gemini candidate contains no structured response text")
    return raw


def _usage(response: Mapping[str, Any], batch_name: str) -> Dict[str, Any]:
    raw = response.get("usageMetadata", {})
    if not isinstance(raw, Mapping):
        raise DataSourceError("Gemini response usageMetadata is not an object")
    try:
        input_tokens = int(raw.get("promptTokenCount", 0))
        output_tokens = int(raw.get("candidatesTokenCount", 0))
        cached_tokens = int(raw.get("cachedContentTokenCount", 0))
        total_tokens = int(raw.get("totalTokenCount", input_tokens + output_tokens))
    except (TypeError, ValueError) as exc:
        raise DataSourceError("Gemini response contains a non-integer token count") from exc
    for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens),
                        ("cached_input_tokens", cached_tokens),
                        ("total_tokens", total_tokens)):
        _count(name, value)
    if cached_tokens > input_tokens:
        raise DataSourceError("Gemini cached token count exceeds its input token count")
    if total_tokens < input_tokens + output_tokens:
        raise DataSourceError("Gemini total token count is smaller than input plus output")
    return {
        "provider": GEMINI_PROVIDER,
        "service_mode": GEMINI_BATCH_MODE,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "batch_name": batch_name,
        "provider_usage": _thaw(raw),
    }


class GeminiBatchTransport:
    """Callable TG7.1 transport backed by Gemini's asynchronous inlined Batch API.

    ``client`` is injectable for offline acceptance tests and defaults to ``httpx.Client``.
    The public object intentionally has no serialisation method and its repr redacts the key.
    """

    def __init__(self, api_key: str, *, cached_content: Optional[str] = None,
                 client: Optional[Any] = None, poll_interval_seconds: float = 5.0,
                 timeout_seconds: float = 86400.0,
                 sleeper: Callable[[float], None] = time.sleep,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._api_key = _nonempty("api_key", api_key)
        if cached_content is not None:
            cached_content = _nonempty("cached_content", cached_content)
            if not cached_content.startswith("cachedContents/"):
                raise InvalidParameterError("cached_content", cached_content,
                                            "a Gemini resource named cachedContents/{id}")
        if not isinstance(poll_interval_seconds, (int, float)) \
                or isinstance(poll_interval_seconds, bool) or poll_interval_seconds < 0:
            raise InvalidParameterError("poll_interval_seconds", poll_interval_seconds,
                                        "a non-negative number")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) \
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise InvalidParameterError("timeout_seconds", timeout_seconds,
                                        "a positive finite number")
        self.cached_content = cached_content
        self.client = client or httpx.Client(timeout=60.0)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.sleeper = sleeper
        self.now = now

    def __repr__(self) -> str:
        return ("GeminiBatchTransport(api_key=<redacted>, cached_content=%r, "
                "poll_interval_seconds=%r, timeout_seconds=%r)"
                % (self.cached_content, self.poll_interval_seconds, self.timeout_seconds))

    @property
    def _headers(self) -> Dict[str, str]:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    @staticmethod
    def _json(response: Any) -> Mapping[str, Any]:
        try:
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            status = getattr(response, "status_code", "unknown")
            raise DataSourceError("Gemini Batch API request failed (HTTP %s)" % status) from exc
        if not isinstance(value, Mapping):
            raise DataSourceError("Gemini Batch API returned a non-object response")
        return value

    def __call__(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        payload = build_gemini_batch_request(request, cached_content=self.cached_content)
        model = request["model_id"]
        url = "%s/models/%s:batchGenerateContent" % (GEMINI_API_ROOT, model)
        initial_response = self.client.post(url, headers=self._headers, json=payload)
        operation = self._json(initial_response)
        name = operation.get("name")
        if not isinstance(name, str) or not name.startswith("batches/"):
            raise DataSourceError("Gemini Batch API returned no batches/{id} operation name")

        started = time.monotonic()
        current = operation
        while True:
            batch = _batch(current)
            state = batch.get("state")
            if current.get("done") is True or state in _TERMINAL_BATCH_STATES:
                break
            if time.monotonic() - started >= self.timeout_seconds:
                raise DataSourceError("Gemini batch %s did not finish before the local timeout"
                                      % name)
            self.sleeper(self.poll_interval_seconds)
            polled = self.client.get("%s/%s" % (GEMINI_API_ROOT, name),
                                     headers=self._headers)
            current = self._json(polled)

        batch = _batch(current)
        state = batch.get("state")
        if state not in (None, "BATCH_STATE_SUCCEEDED") or "error" in current:
            raise DataSourceError("Gemini batch %s ended in state %s" % (name, state))
        output = batch.get("output")
        inline = output.get("inlinedResponses") if isinstance(output, Mapping) else None
        rows = inline.get("inlinedResponses") if isinstance(inline, Mapping) else None
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise DataSourceError("Gemini batch %s returned no unique inlined response" % name)
        if "error" in rows[0]:
            raise DataSourceError("Gemini batch %s returned an item-level error" % name)
        response = rows[0].get("response")
        if not isinstance(response, Mapping):
            raise DataSourceError("Gemini batch %s returned no GenerateContentResponse" % name)
        raw = _gemini_text(response)
        try:
            structured = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataSourceError("Gemini structured output was not valid JSON") from exc
        responded = batch.get("endTime") or batch.get("updateTime")
        if not isinstance(responded, str):
            responded = self.now().isoformat()
        return {
            "raw": raw,
            "structured": structured,
            "model_id": model,
            "api_request_id": name,
            "responded_at": responded,
            "usage": _usage(response, name),
        }


DEFAULT_ROLE_EFFORTS = MappingProxyType({
    "candidate_synthesis": "high",
    "statistical_challenge": "low",
    "confounder_challenge": "low",
    "domain_plausibility_challenge": "low",
    "provenance_challenge": "low",
    "response_and_revision": "medium",
    "independent_reassessment": "high",
    "final_synthesis": "high",
})


@dataclass(frozen=True)
class ReviewCostPolicy:
    """The model, effort, service tier and measurable cache criterion fixed before review."""

    provider: str = GEMINI_PROVIDER
    model_id: str = GEMINI_35_FLASH
    role_efforts: Mapping[str, str] = dc_field(default_factory=lambda: DEFAULT_ROLE_EFFORTS)
    required_service_mode: str = GEMINI_BATCH_MODE
    minimum_total_cached_tokens: int = 1
    batch_discount_fraction: float = 0.5
    documentation_reviewed_at: str = DOCUMENTATION_REVIEWED_AT
    source_urls: Tuple[str, ...] = (
        GEMINI_MODEL_SOURCE, GEMINI_CACHE_SOURCE, GEMINI_BATCH_SOURCE, GEMINI_PRICING_SOURCE)
    policy_sha256: str = ""
    schema: str = dc_field(default=COST_POLICY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _nonempty("policy.provider", self.provider))
        object.__setattr__(self, "model_id", _nonempty("policy.model_id", self.model_id))
        if set(self.role_efforts) != set(REVIEW_ROLES):
            raise InvalidParameterError("policy.role_efforts", sorted(self.role_efforts),
                                        "exactly the eight TG7 review roles")
        efforts = dict(self.role_efforts)
        for role, effort in efforts.items():
            if effort not in EFFORTS:
                raise InvalidParameterError("policy.role_efforts.%s" % role, effort,
                                            "one of %s" % ", ".join(EFFORTS))
        object.__setattr__(self, "role_efforts", MappingProxyType(efforts))
        object.__setattr__(self, "required_service_mode",
                           _nonempty("policy.required_service_mode",
                                     self.required_service_mode))
        _count("policy.minimum_total_cached_tokens", self.minimum_total_cached_tokens)
        if not isinstance(self.batch_discount_fraction, (int, float)) \
                or isinstance(self.batch_discount_fraction, bool) \
                or not 0 < float(self.batch_discount_fraction) <= 1:
            raise InvalidParameterError("policy.batch_discount_fraction",
                                        self.batch_discount_fraction,
                                        "a fraction greater than zero and at most one")
        try:
            parsed = datetime.fromisoformat(self.documentation_reviewed_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            raise InvalidParameterError("policy.documentation_reviewed_at",
                                        self.documentation_reviewed_at,
                                        "a timezone-bearing ISO-8601 timestamp") from None
        if parsed.tzinfo is None:
            raise InvalidParameterError("policy.documentation_reviewed_at",
                                        self.documentation_reviewed_at,
                                        "a timezone-bearing ISO-8601 timestamp")
        urls = tuple(_nonempty("policy.source_url", url) for url in self.source_urls)
        if len(urls) != len(set(urls)):
            raise InvalidParameterError("policy.source_urls", urls, "unique documentation URLs")
        object.__setattr__(self, "source_urls", urls)
        computed = _digest(self.body())
        if not self.policy_sha256:
            object.__setattr__(self, "policy_sha256", computed)
        elif self.policy_sha256 != computed:
            raise InvalidParameterError("policy.policy_sha256", self.policy_sha256,
                                        "the digest recomputed from the cost policy")

    def panel(self) -> ReviewPanel:
        return ReviewPanel(seats={
            role: PanelSeat(role=role, model_id=self.model_id, effort=self.role_efforts[role])
            for role in REVIEW_ROLES
        })

    def body(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "provider": self.provider, "model_id": self.model_id,
            "role_efforts": dict(sorted(self.role_efforts.items())),
            "required_service_mode": self.required_service_mode,
            "minimum_total_cached_tokens": self.minimum_total_cached_tokens,
            "batch_discount_fraction": float(self.batch_discount_fraction),
            "documentation_reviewed_at": self.documentation_reviewed_at,
            "source_urls": list(self.source_urls),
        }

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), policy_sha256=self.policy_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewCostPolicy":
        fields = tuple(cls().to_mapping())
        record = _exact(value, fields, "review cost policy")
        if record["schema"] != COST_POLICY_SCHEMA:
            raise InvalidParameterError("policy.schema", record["schema"], COST_POLICY_SCHEMA)
        return cls(provider=record["provider"], model_id=record["model_id"],
                   role_efforts=record["role_efforts"],
                   required_service_mode=record["required_service_mode"],
                   minimum_total_cached_tokens=record["minimum_total_cached_tokens"],
                   batch_discount_fraction=record["batch_discount_fraction"],
                   documentation_reviewed_at=record["documentation_reviewed_at"],
                   source_urls=tuple(record["source_urls"]),
                   policy_sha256=record["policy_sha256"])


@dataclass(frozen=True)
class ReviewCostReceipt:
    """Content-addressed route and token evidence for one exact review record."""

    review_record_sha256: str
    policy_sha256: str
    call_count: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    total_tokens: int
    batch_names: Tuple[str, ...]
    cache_hit_fraction: float
    receipt_sha256: str = ""
    schema: str = dc_field(default=COST_RECEIPT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("review_record_sha256", "policy_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise InvalidParameterError(name, value,
                                            "a lowercase 64-character SHA-256 digest")
        for name in ("call_count", "input_tokens", "output_tokens",
                     "cached_input_tokens", "total_tokens"):
            _count(name, getattr(self, name))
        names = tuple(_nonempty("batch_name", name) for name in self.batch_names)
        if len(names) != self.call_count:
            raise InvalidParameterError("batch_names", names, "one batch name per call")
        if len(names) != len(set(names)):
            raise InvalidParameterError("batch_names", names,
                                        "one unique provider batch identity per call")
        object.__setattr__(self, "batch_names", names)
        expected = (self.cached_input_tokens / self.input_tokens
                    if self.input_tokens else 0.0)
        if not math.isclose(self.cache_hit_fraction, expected, rel_tol=0.0, abs_tol=1e-15):
            raise InvalidParameterError("cache_hit_fraction", self.cache_hit_fraction,
                                        "cached_input_tokens / input_tokens")
        computed = _digest(self.body())
        if not self.receipt_sha256:
            object.__setattr__(self, "receipt_sha256", computed)
        elif self.receipt_sha256 != computed:
            raise InvalidParameterError("receipt_sha256", self.receipt_sha256,
                                        "the digest recomputed from the cost receipt")

    def body(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "review_record_sha256": self.review_record_sha256,
            "policy_sha256": self.policy_sha256, "call_count": self.call_count,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens, "total_tokens": self.total_tokens,
            "batch_names": list(self.batch_names), "cache_hit_fraction": self.cache_hit_fraction,
        }

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), receipt_sha256=self.receipt_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewCostReceipt":
        fields = (
            "schema", "review_record_sha256", "policy_sha256", "call_count",
            "input_tokens", "output_tokens", "cached_input_tokens", "total_tokens",
            "batch_names", "cache_hit_fraction", "receipt_sha256",
        )
        record = _exact(value, fields, "review cost receipt")
        if record["schema"] != COST_RECEIPT_SCHEMA:
            raise InvalidParameterError("receipt.schema", record["schema"],
                                        COST_RECEIPT_SCHEMA)
        return cls(review_record_sha256=record["review_record_sha256"],
                   policy_sha256=record["policy_sha256"], call_count=record["call_count"],
                   input_tokens=record["input_tokens"], output_tokens=record["output_tokens"],
                   cached_input_tokens=record["cached_input_tokens"],
                   total_tokens=record["total_tokens"],
                   batch_names=tuple(record["batch_names"]),
                   cache_hit_fraction=record["cache_hit_fraction"],
                   receipt_sha256=record["receipt_sha256"])


def audit_review_cost(reviewed: ReviewedBundle, policy: ReviewCostPolicy) -> ReviewCostReceipt:
    """Refuse an unaudited route/usage record; otherwise bind normalized totals to the review."""
    if not isinstance(reviewed, ReviewedBundle):
        raise InvalidParameterError("reviewed", type(reviewed).__name__, "a ReviewedBundle")
    if not isinstance(policy, ReviewCostPolicy):
        raise InvalidParameterError("policy", type(policy).__name__, "a ReviewCostPolicy")
    totals = {"input_tokens": 0, "output_tokens": 0,
              "cached_input_tokens": 0, "total_tokens": 0}
    batch_names = []
    for call in reviewed.review.calls:
        role = call.request.role
        if call.request.model_id != policy.model_id:
            raise InvalidParameterError("call.model_id", call.request.model_id,
                                        "the predeclared model %r" % policy.model_id)
        if call.request.effort != policy.role_efforts[role]:
            raise InvalidParameterError("call.effort", call.request.effort,
                                        "the predeclared %s effort %r"
                                        % (role, policy.role_efforts[role]))
        usage = _exact(_thaw(call.response.usage), _USAGE_FIELDS, "call.response.usage")
        if usage["provider"] != policy.provider:
            raise InvalidParameterError("usage.provider", usage["provider"], policy.provider)
        if usage["service_mode"] != policy.required_service_mode:
            raise InvalidParameterError("usage.service_mode", usage["service_mode"],
                                        policy.required_service_mode)
        name = _nonempty("usage.batch_name", usage["batch_name"])
        if name != call.response.api_request_id:
            raise InvalidParameterError("usage.batch_name", name,
                                        "the recorded API request id %r"
                                        % call.response.api_request_id)
        provider_usage = usage["provider_usage"]
        if not isinstance(provider_usage, Mapping):
            raise InvalidParameterError("usage.provider_usage", type(provider_usage).__name__,
                                        "the unmodified provider usage object")
        counts = {key: _count("usage.%s" % key, usage[key]) for key in totals}
        reconciled = _usage({"usageMetadata": provider_usage}, name)
        for key in totals:
            if counts[key] != reconciled[key]:
                raise InvalidParameterError(
                    "usage.%s" % key, counts[key],
                    "the count independently parsed from provider_usage (%d)"
                    % reconciled[key])
        if counts["cached_input_tokens"] > counts["input_tokens"]:
            raise InvalidParameterError("usage.cached_input_tokens",
                                        counts["cached_input_tokens"],
                                        "no more than input_tokens")
        if counts["total_tokens"] < counts["input_tokens"] + counts["output_tokens"]:
            raise InvalidParameterError("usage.total_tokens", counts["total_tokens"],
                                        "at least input_tokens + output_tokens")
        for key in totals:
            totals[key] += counts[key]
        batch_names.append(name)
    if totals["cached_input_tokens"] < policy.minimum_total_cached_tokens:
        raise InvalidParameterError(
            "review.cached_input_tokens", totals["cached_input_tokens"],
            "at least %d measured cache-hit tokens. Enabling caching is not evidence that a "
            "request hit it" % policy.minimum_total_cached_tokens)
    return ReviewCostReceipt(
        review_record_sha256=reviewed.review.record_sha256,
        policy_sha256=policy.policy_sha256,
        call_count=len(reviewed.review.calls), batch_names=tuple(batch_names),
        cache_hit_fraction=(totals["cached_input_tokens"] / totals["input_tokens"]
                            if totals["input_tokens"] else 0.0),
        **totals)


def save_cost_receipt(path: Union[os.PathLike[str], str], receipt: ReviewCostReceipt, *,
                      overwrite: bool = False) -> Path:
    """Publish one canonical receipt atomically; overwrite is an explicit exceptional act."""
    if not isinstance(receipt, ReviewCostReceipt):
        raise InvalidParameterError("receipt", type(receipt).__name__, "a ReviewCostReceipt")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise InvalidParameterError("path", str(target),
                                    "a new path, or overwrite=True explicitly")
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise InvalidParameterError("temporary path", str(temporary),
                                    "no stale publication temporary file")
    try:
        temporary.write_bytes(_canonical(receipt.to_mapping()) + b"\n")
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_cost_receipt(path: Union[os.PathLike[str], str]) -> ReviewCostReceipt:
    """Reload and re-verify a published cost receipt."""
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidParameterError("path", str(target),
                                    "a readable canonical TG7.3 cost receipt") from exc
    return ReviewCostReceipt.from_mapping(value)


__all__ = [
    "COST_POLICY_SCHEMA", "COST_RECEIPT_SCHEMA", "GEMINI_PROVIDER", "GEMINI_35_FLASH",
    "GEMINI_BATCH_MODE", "GEMINI_API_ROOT", "GEMINI_CACHE_MINIMUM_TOKENS",
    "GEMINI_MODEL_SOURCE", "GEMINI_CACHE_SOURCE", "GEMINI_BATCH_SOURCE",
    "GEMINI_PRICING_SOURCE", "DOCUMENTATION_REVIEWED_AT", "DEFAULT_ROLE_EFFORTS",
    "build_gemini_batch_request", "GeminiBatchTransport", "ReviewCostPolicy",
    "ReviewCostReceipt", "audit_review_cost", "save_cost_receipt", "load_cost_receipt",
]
