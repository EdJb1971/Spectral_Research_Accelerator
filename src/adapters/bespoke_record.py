"""The bespoke record family (TG17.3, `ed-dev`).

**Order book is not a finance adapter. It is one declaration in this family.**

Three of this programme's four flagship domains reach a public archive with a documented
addressing scheme, a nominal cadence and a licence. The fourth does not: it is whatever record a
researcher has, in a file they hold, measuring something no catalogue here has heard of — a
venue's aggregated trade volume, a factory line's cycle counter, a clinic's appointment log, a
telescope's engineering channel. Treating that as "the finance adapter" would have written the
one example into the code and left every other bespoke record needing a fifth, sixth and seventh
adapter. So this module is the *family*, and ``order_book`` is its first saved instance.

**The fence, and why it is the whole design.** A family that accepts any record is one keystroke
from accepting any claim. The rule that prevents it is TG8.4's, already enforced in
`src/data_layer/tabular_source.py` and reused here without amendment:

    detection may create a required declaration; it may never satisfy one.

Nothing in this module reads a file and concludes what the domain is. `clock_facts` observes
that a clock is irregular; `required_violations` turns that observation into an *obligation*;
the researcher must then have declared `irregular_sampling` on a domain that passed
`onboard_domain` before the record may be read at all. The same holds for aggregate support
(`aggregated_values`), for axes a flat record cannot supply (`assert_domain_admits_channel_table`)
and for the lag floor: a bespoke domain declaring `lag_policy="none"` gets `precedence` added to
its refused operations by construction, so no bespoke record can produce a lead-lag reading
merely because its author did not think about it.

**What a bespoke domain therefore costs to add, and what it does not prove.** A researcher
supplies a declaration, its wording, and a content-addressed record; they write no code. That is
the point of the family. It is emphatically **not** evidence that the `DomainExperimentAdapter`
seam works — a data-driven instance of an existing adapter tests the adapter's parameters, not
the registry's extension point. TG17.3's acceptance criterion is met by a fifth adapter with
genuinely different structural mathematics, registered from outside `src/`; see
`src/tests/test_adapter_registry.py`. Conflating the two would let this module's flexibility
stand in for a seam nobody had exercised.

**Identity is the bytes, never the filename.** The `record_sha256` control refuses a path. Two
files with the same name are not the same record, a file edited in place is a different record
with the same name, and a result that named a filename would be unreproducible the moment either
happened. Until a record is bound by digest the acquisition plan carries an explicit refusal
rather than a plan — which is why TG17.1's flagship recipe preflights as `REFUSED` and says so.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.adapters.standardized_level_adapter import (BENCHMARK_BINDING, LIVE_BINDING,
                                                     build_standardized_level_adapter)
from src.core.builtin_domains import ORDER_BOOK, register_builtin_domains

from src.core.domain import DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.experiment_adapter import (AcquisitionPlan, ControlField, DomainExperimentAdapter,
                                         register_experiment_adapter)
from src.core.onboarding import is_onboarded

register_builtin_domains()

BESPOKE_FAMILY = "bespoke_record"

#: The refusal a researcher sees before a record is bound. It names the remedy rather than the
#: rule, because the rule is only useful to someone who already knows what to do about it.
UNBOUND_REFUSAL = ("select a content-addressed local record. A bespoke domain has no public "
                   "archive to plan against, so its coverage cannot be estimated from a "
                   "catalogue: the record itself is the only thing that establishes what was "
                   "observed and when.")

CONTROLS: Tuple[ControlField, ...] = (
    ControlField(name="record_sha256", label="Record", kind="content_record", required=False,
                 help=("The sha256 of the record this observation reads. A filename is not an "
                       "identity: a file edited in place keeps its name and becomes a "
                       "different record, and a result naming the name would not reproduce.")),
    ControlField(name="value_column", label="Value column", kind="text", required=False,
                 help=("Which column carries the measured value. A bespoke record has no "
                       "catalogue to infer this from, so it is chosen rather than guessed.")),
    ControlField(name="time_column", label="Clock column", kind="text", required=False,
                 help=("Which column carries the clock. Choosing the wrong one produces a "
                       "record that is ordered by something other than time, which no "
                       "downstream refusal can detect.")),
    ControlField(name="time_units", label="Clock units", kind="enum", default="s",
                 choices=("s", "min", "h", "d"),
                 help=("What one unit of the clock column means. A lag is reported as a "
                       "duration, so a clock in minutes read as seconds misstates every "
                       "interval by sixty.")),
)


def bespoke_plan(parameters: Mapping[str, Any],
                 identity: Mapping[str, Any] | None = None) -> AcquisitionPlan:
    """A local record's plan: no network, no catalogue, and no exactness from an extent."""
    # The identity is checked first: TG17.1 froze the record binding as part of *which bytes
    # were acquired* rather than as a configuration choice, and a control that could silently
    # override it would let two runs of one manifest read two different records.
    digest = dict(identity or {}).get("content_sha256") or parameters.get("record_sha256")
    return AcquisitionPlan(
        source_id="channel_table:local", source_version="user-supplied-v1",
        support_kind="irregular_local_record", access="local_file_binding",
        access_means="A local record already published to this server by content digest.",
        coverage_exact=False, native_cadence_seconds=None, estimated_bytes_per_day=0,
        opens_measurement_values=False, network_used=False,
        identity={"licence_scope": "user-declared", "content_sha256": digest,
                  "value_column": parameters.get("value_column"),
                  "time_column": parameters.get("time_column"),
                  "time_units": parameters.get("time_units")},
        refusal=None if digest else UNBOUND_REFUSAL)


def assert_record_admissible(declaration: DomainDeclaration,
                             sample_times_seconds: Sequence[float],
                             supports: Optional[Mapping[str, float]] = None) -> None:
    """The TG8.4 fence, applied to a bespoke record before it may become an observation.

    Every refusal here is imported rather than restated. If `tabular_source` learns a new
    obligation, this path inherits it; a second copy of the rule would be a second place for it
    to fall out of date, and the rule that fell out of date would be the one nobody was
    enforcing.
    """
    from src.data_layer.tabular_source import (assert_domain_admits_channel_table, clock_facts,
                                               required_violations)

    if not is_onboarded(declaration.name):
        raise InvalidParameterError(
            "domain", declaration.name,
            "a domain onboarded through the contract. A bespoke record is exactly the case "
            "where nobody else has checked the declaration, so the contract is the only thing "
            "standing between a convenient column and a claim about the world")
    assert_domain_admits_channel_table(declaration)
    facts = clock_facts(np.asarray(sample_times_seconds, dtype=np.float64))
    missing = [name for name in required_violations(facts)
               if name not in tuple(declaration.violations)]
    if missing:
        raise InvalidParameterError(
            "domain.violations", sorted(declaration.violations),
            "the declared violations this record obliges: %s. The clock in this record is %s, "
            "which is an observation about the file and not a permission: the domain must "
            "already have said this about itself before the record may be read under it "
            "(standard E15, rule R17)"
            % (", ".join(missing),
               "not regularly spaced" if not facts.get("regular") else "irregular"))
    for name, footprint in dict(supports or {}).items():
        if float(footprint) > 1.0 and "aggregated_values" not in tuple(declaration.violations):
            raise InvalidParameterError(
                "domain.violations", sorted(declaration.violations),
                "the declared violation 'aggregated_values', because column %r covers more "
                "than one sample of the parent axis. Declaring the footprint without the "
                "violation sets the lag floor correctly and leaves every other refusal that "
                "depends on it switched off" % name)


def build_bespoke_adapter(declaration: DomainDeclaration, *, accepted_semantics: str,
                          accepted_units: str, adapter_id: Optional[str] = None,
                          fixture_domain: Optional[str] = None,
                          admissible_kernels: tuple = ("exact_support_overlap",),
                          admissible_nulls: tuple = ("independent_native_clock_shift",),
                          ) -> DomainExperimentAdapter:
    """One bespoke domain's adapter. No code is written per domain; a declaration is."""

    def fixture(parameters: Mapping[str, Any]):
        from src.benchmarks.structural_trajectory import known_answer_native

        record = known_answer_native(fixture_domain or declaration.name)
        assert_record_admissible(declaration, record.sample_times_seconds)
        return record

    return build_standardized_level_adapter(
        declaration=declaration,
        adapter_id=adapter_id or "%s.%s" % (declaration.name, BESPOKE_FAMILY),
        accepted_semantics=accepted_semantics, accepted_units=accepted_units,
        controls=CONTROLS, plan=bespoke_plan, admissible_kernels=admissible_kernels,
        admissible_nulls=admissible_nulls,
        fixture_record=fixture if fixture_domain or declaration.name else None,
        live_refusal=(
            "a binding this slice can materialise. %s Once bound, this domain reads its own "
            "record and no catalogue: nothing here will estimate its coverage from a product "
            "description." % UNBOUND_REFUSAL),
        extra_leakage_risks=("bespoke_column_choice",),
        domain_mathematics=(
            "none: the bespoke family supplies no structural mathematics of its own. What it "
            "supplies is the TG8.4 fence — a record's observed clock creates obligations on "
            "the domain declaration and never satisfies them.",))


def register_bespoke_domain(declaration: DomainDeclaration, *, accepted_semantics: str,
                            accepted_units: str,
                            admissible_kernels: tuple = ("exact_support_overlap",),
                            admissible_nulls: tuple = ("independent_native_clock_shift",),
                            replace: bool = False) -> DomainExperimentAdapter:
    """Add a bespoke domain to the running server, through the supported seam.

    The declaration must already have passed `onboard_domain`. This deliberately does not
    onboard it as a side effect: onboarding is atomic and refuses several ways, and a caller
    that had those refusals hidden inside an adapter registration would meet them as a failure
    to add an adapter rather than as a statement about their domain.
    """
    adapter = build_bespoke_adapter(declaration, accepted_semantics=accepted_semantics,
                                    accepted_units=accepted_units,
                                    admissible_kernels=admissible_kernels,
                                    admissible_nulls=admissible_nulls)
    return register_experiment_adapter(adapter, replace=replace)


#: The family's first saved instance. TG17.0 kept order book in the flagship quartet because its
#: irregular aggregated clock, absent physical metric and `lag_policy="none"` falsify more of the
#: abstraction than a second gridded product would. That reasoning is about the declaration, not
#: about markets, and it is why this is a declaration rather than a module of its own.
#: The bespoke family admits only `exact_support_overlap` by default (TG17.4). A record whose
#: clock is whatever the depositor happened to write down has no cadence to snap to and no
#: interval over which "the last value still held" is a statement about the world rather than
#: about the file. A bespoke domain that has earned a wider kernel says so in its own
#: registration; it does not inherit one.
#: The same reasoning applies to its nulls. A trading session is a genuine group, but this
#: domain's clock is whatever the depositor wrote down: the framework cannot tell where one
#: session ends without being told, and a group length it inferred would be a scientific choice
#: nobody made. So only the plain shift is admitted, and a depositor who can state their session
#: boundary says so in their own registration. Nor does it admit the scale/shape null: that
#: family alters no record, but admitting it would claim this domain has a native duration
#: worth comparing shapes across, and a bespoke record's native scale is whatever the
#: depositor wrote down.
ADAPTER = build_bespoke_adapter(ORDER_BOOK, accepted_semantics="aggregated traded volume",
                                accepted_units="shares")

__all__ = ["ADAPTER", "BESPOKE_FAMILY", "CONTROLS", "UNBOUND_REFUSAL",
           "assert_record_admissible", "bespoke_plan", "build_bespoke_adapter",
           "register_bespoke_domain"]
