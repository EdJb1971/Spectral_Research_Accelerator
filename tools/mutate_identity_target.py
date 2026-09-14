"""Bounded T4E.8 slice 3 mutation audit in isolated Python processes; never edits source files.

Same contract as `mutate_spatial_identity.py`: a module is re-executed in its existing
namespace before pytest imports the acceptance tests, each affected module first passes the
same unmutated reload path, and a timeout or import failure does not count as a kill. The
mutants below all weaken the declaration or its admissibility rule -- the machinery whose
only job is to refuse -- because logic that is never exercised by a passing run is exactly
the logic that rots.
"""
import importlib
import json
from pathlib import Path
import subprocess
import sys

TEST = "src/tests/test_identity_target_declaration.py"
A = "src.analysis_engine.spectral_identity_audit"
MUTANTS = [
    ("admissibility_never_refuses", A,
     "if evidence not in specification.admissible_evidence:", "if False:"),
    ("kind_recurrence_admits_proxy_labels", A,
     'admissible_evidence=("external_reference",),',
     'admissible_evidence=("external_reference", "record_derived_proxy"),'),
    ("absent_target_falls_through", A, "if not target:", "if False and not target:"),
    ("absent_evidence_falls_through", A, "if not evidence:", "if False and not evidence:"),
    ("proxy_wording_reworded", A,
     'PROXY_LABEL_BOUNDARY = "Tracked-key proxy labels, not independent measurements or physical ground truth"',
     'PROXY_LABEL_BOUNDARY = "Proxy labels"'),
    ("caveat_never_published", A, "return dict(self.caveats).get(evidence)", "return None"),
    ("proxy_claims_independence", A,
     "    provenance=\"Labels computed from the same record and pipeline whose identity is under test\",\n"
     "    independent_of_record=False,",
     "    provenance=\"Labels computed from the same record and pipeline whose identity is under test\",\n"
     "    independent_of_record=True,"),
    ("declared_boundary_ignored", A, '"label_boundary": label_boundary,',
     '"label_boundary": PROXY_LABEL_BOUNDARY,'),
    ("catalogue_family_unchecked", A, "if not present & families:", "if False:"),
    ("unmatched_counted_as_an_identity", A, "if pattern_id is None:", "if False:"),
    ("catalogue_type_unchecked", A, "if not isinstance(catalogue, PatternCatalogue):", "if False:"),
    ("negative_budget_unbounded", A, "if attempts >= budget:", "if False:"),
]


def child(module_name, mutant_name):
    module = importlib.import_module(module_name)
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    if mutant_name != "baseline":
        _, _, before, after = next(m for m in MUTANTS if m[0] == mutant_name)
        if before not in source:
            raise ValueError("Mutation anchor absent: " + mutant_name)
        source = source.replace(before, after)
    exec(compile(source, str(path), "exec"), module.__dict__)
    import pytest
    return pytest.main([TEST, "-q", "--tb=short"])


def run():
    records = []
    for module in dict.fromkeys(m[1] for m in MUTANTS):
        tasks = [("baseline", module)] + [(m[0], m[1]) for m in MUTANTS if m[1] == module]
        for name, target in tasks:
            result = subprocess.run([sys.executable, "-m", "tools.mutate_identity_target",
                                     "--child", target, name], capture_output=True, text=True,
                                    timeout=900)
            killed = result.returncode not in (0,)
            records.append({"mutant": name, "module": target, "returncode": result.returncode,
                            "killed_by_failing_test": bool(killed and name != "baseline"),
                            "tail": result.stdout.strip().splitlines()[-1:] or [""]})
            print("%-38s %-8s rc=%d" % (name, "KILLED" if killed else "survived",
                                        result.returncode), flush=True)
    baselines = [r for r in records if r["mutant"] == "baseline"]
    mutants = [r for r in records if r["mutant"] != "baseline"]
    summary = {
        "schema": "t4e8-slice3-mutation-audit/v1", "test_file": TEST,
        "baselines_passed": all(r["returncode"] == 0 for r in baselines),
        "mutants": len(mutants),
        "killed": sum(1 for r in mutants if r["killed_by_failing_test"]),
        "survivors": [r["mutant"] for r in mutants if not r["killed_by_failing_test"]],
        "records": records,
        "boundary": ("A kill means a mutated run reached pytest and reported a failing "
                     "assertion. Timeouts and import failures are not kills."),
    }
    Path("measurements/t4e8_identity_target_mutations.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("baselines_passed", "mutants", "killed", "survivors")}, indent=2))
    return 0 if summary["baselines_passed"] and not summary["survivors"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(child(sys.argv[2], sys.argv[3]))
    sys.exit(run())
