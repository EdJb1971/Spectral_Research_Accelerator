"""Bounded T4E.8 mutation audit in isolated Python processes; never edits source files.

Re-executes a module in its existing namespace before pytest imports the acceptance tests.
Each affected module first passes the same unmutated reload path. A timeout or import failure
does not count as a kill: a mutated run must reach pytest and report failing assertions.
"""
import importlib
import json
from pathlib import Path
import subprocess
import sys

TEST = "src/tests/test_spectral_spatial_identity.py"
I = "src.analysis_engine.spectral_invariance"
C = "src.analysis_engine.spectral_clustering"
A = "src.analysis_engine.spectral_identity_audit"
MUTANTS = [
    ("band_normalisation", I, "values[left, right] = float(distance.value)",
     "values[left, right] = float(distance.value) / constellation.features[left].spatial_scale.value"),
    ("constant_pair_shape", I, "values[left, right] = float(distance.value)",
     "values[left, right] = 1.0"),
    ("strengths_comparable", I, "strengths=False, scales=False, requires_scope=True",
     "strengths=True, scales=False, requires_scope=True"),
    ("scales_comparable", I, "strengths=False, scales=False, requires_scope=True",
     "strengths=False, scales=True, requires_scope=True"),
    ("node_order_not_canonical", I, "for order in itertools.permutations(range(size)):",
     "for order in [tuple(range(size))]:"),
    ("source_not_bound", I, "tuple(constellation.features[0].semantic_key)",
     '("any-domain", "any-dataset", "any-variable")'),
    ("mixed_families_compared", C, "if left.family != right.family:", "if False:"),
    ("carried_scales_reenter_point", C, 'scales=blocks["scales"]', 'scales=tuple(signature.scales)'),
    ("split_radius_tie", A, "same > radius", "same >= radius"),
    ("admission_radius_tie", A, "different <= radius", "different < radius"),
    ("auc_ties_count_as_wins", A, "(through - below) / 2", "(through - below)"),
    ("recall_rounds_down", A, "math.ceil(recall * len(values))", "math.floor(recall * len(values))"),
    ("feasibility_always_true", A, "else admission <= max_admission", "else True"),
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
            result = subprocess.run([sys.executable, "-m", "tools.mutate_spatial_identity",
                                     "--child", target, name], capture_output=True, text=True,
                                    timeout=120)
            output = result.stdout + result.stderr
            expected = result.returncode == 0 if name == "baseline" else (
                result.returncode == 1 and " failed" in output and "ERROR collecting" not in output)
            record = {"module": target, "mutation": name, "exit_code": result.returncode,
                      "status": ("BASELINE_PASS" if name == "baseline" else "KILLED") if expected else "AUDIT_FAILED",
                      "output": output}
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "output"}), flush=True)
            if not expected:
                print(output, flush=True)
                break
        if not expected:
            break
    Path("measurements/t4e8_spatial_mutations.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0 if expected else 1


if __name__ == "__main__":
    raise SystemExit(child(sys.argv[2], sys.argv[3]) if len(sys.argv) > 1 and sys.argv[1] == "--child"
                     else run())
