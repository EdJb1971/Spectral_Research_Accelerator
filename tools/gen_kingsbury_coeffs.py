"""Generate the vendored Kingsbury coefficient module from the `dtcwt` reference data.

Run once; the output is committed. The point of vendoring is that the runtime must not
depend on `dtcwt` (or on `pytorch_wavelets`, which imports the deprecated `pkg_resources`),
while the coefficients themselves stay byte-identical to the reference so the oracle
cross-checks are meaningful.
"""
import numpy as np
import dtcwt.coeffs as C

WANT = {
    "near_sym_a": ("h0o", "h1o", "g0o", "g1o"),
    "near_sym_b": ("h0o", "h1o", "g0o", "g1o"),
    "legall": ("h0o", "h1o", "g0o", "g1o"),
    "qshift_a": ("h0a", "h0b", "h1a", "h1b", "g0a", "g0b", "g1a", "g1b"),
    "qshift_b": ("h0a", "h0b", "h1a", "h1b", "g0a", "g0b", "g1a", "g1b"),
    "qshift_c": ("h0a", "h0b", "h1a", "h1b", "g0a", "g0b", "g1a", "g1b"),
    "qshift_d": ("h0a", "h0b", "h1a", "h1b", "g0a", "g0b", "g1a", "g1b"),
}

import os, inspect, glob
data_dir = os.path.join(os.path.dirname(inspect.getfile(C)), "data")

lines = []
lines.append('"""Vendored Kingsbury filter coefficients for the dual-tree complex wavelet transform.')
lines.append("")
lines.append("**Generated file - do not hand-edit.** Produced by `_gen_coeffs.py` from the reference")
lines.append("`dtcwt` package (N. Kingsbury's coefficients, as distributed in dtcwt 0.14.0), and")
lines.append("committed so the *runtime* depends on neither `dtcwt` nor `pytorch_wavelets`.")
lines.append("")
lines.append("Why vendor rather than import:")
lines.append("")
lines.append("*   `pytorch_wavelets` imports `pkg_resources`, an API scheduled for removal. A")
lines.append("    research platform intended to outlast its dependencies should not take a runtime")
lines.append("    dependency on a deprecated loader for what are, in the end, ~80 constants.")
lines.append("*   Both packages remain **test-only oracles**. They are far more valuable as two")
lines.append("    independent validators of our implementation than as the implementation itself,")
lines.append("    and vendoring the coefficients is what keeps that cross-check honest: if we")
lines.append("    imported the filters from the oracle, agreement would be partly circular.")
lines.append("*   Full `repr()` precision is used. Truncated wavelet literals cost real accuracy -")
lines.append("    in T3.5.7 truncated db3 coefficients held orthonormality only to 1e-11 and failed")
lines.append("    a PyWavelets cross-check that full precision passes exactly.")
lines.append('"""')
lines.append("")
lines.append("from typing import Dict, Tuple")
lines.append("")
lines.append("#: Level-1 (non-q-shift) filter sets: odd-length, used for the first decomposition.")
lines.append("LEVEL1_FILTERS: Dict[str, Dict[str, Tuple[float, ...]]] = {")
for name in ("near_sym_a", "near_sym_b", "legall"):
    z = np.load(os.path.join(data_dir, name + ".npz"))
    lines.append("    %r: {" % name)
    for key in WANT[name]:
        arr = np.asarray(z[key], dtype=np.float64).ravel()
        lines.append("        %r: (" % key)
        for i in range(0, len(arr), 3):
            chunk = ", ".join(repr(float(v)) for v in arr[i:i + 3])
            lines.append("            " + chunk + ",")
        lines.append("        ),")
    lines.append("    },")
lines.append("}")
lines.append("")
lines.append("#: Q-shift filter sets for levels >= 2. Tree b's filters are the time-reverse of")
lines.append("#: tree a's, which is what produces the quarter-sample delay difference and hence")
lines.append("#: the approximate Hilbert-pair relationship between the two trees.")
lines.append("QSHIFT_FILTERS: Dict[str, Dict[str, Tuple[float, ...]]] = {")
for name in ("qshift_a", "qshift_b", "qshift_c", "qshift_d"):
    z = np.load(os.path.join(data_dir, name + ".npz"))
    lines.append("    %r: {" % name)
    for key in WANT[name]:
        arr = np.asarray(z[key], dtype=np.float64).ravel()
        lines.append("        %r: (" % key)
        for i in range(0, len(arr), 3):
            chunk = ", ".join(repr(float(v)) for v in arr[i:i + 3])
            lines.append("            " + chunk + ",")
        lines.append("        ),")
    lines.append("    },")
lines.append("}")
lines.append("")
lines.append("#: Provenance of the vendored numbers, so their origin is never in doubt.")
lines.append("COEFFICIENT_PROVENANCE = {")
lines.append('    "source_package": "dtcwt",')
lines.append('    "source_version": %r,' % getattr(__import__("dtcwt"), "__version__", "0.14.0"))
lines.append('    "author": "N. G. Kingsbury",')
lines.append('    "generated_by": "_gen_coeffs.py",')
lines.append('    "precision": "full float64 repr()",')
lines.append("}")
lines.append("")

out = "\n".join(lines)
with open("src/transform_engine/kingsbury_coeffs.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(out)
print("wrote src/transform_engine/kingsbury_coeffs.py (%d lines)" % len(lines))
