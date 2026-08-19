"""Transform registry (T3.5.15 / D15, standard E1).

Replaces the parallel `if/elif transform_type == ...` chains that had grown in *two* places -
`experiment_engine/engine.py` and `api/main.py` - and that had to be edited in lockstep every
time a transform was added. Two consecutive slices added `swt` and `dtcwt` that way, each
leaving a `TODO(T3.5.15)` comment.

Each entry declares its parameters and its **capabilities**, which is what makes the wavelet
bank of T4B.2 possible: a sweep can ask for "every shift-invariant transform" rather than
naming them, and a new transform joins the bank by being registered rather than by anyone
editing a sweep definition.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional

import torch

from src.core.registry import Registry
from src.physical_core.field import PhysicalField

TRANSFORMS: Registry["TransformSpec"] = Registry("transform")


class TransformSpec(NamedTuple):
    """Everything the engine and the API need in order to run one transform."""

    apply: Callable[..., Dict[str, Any]]
    #: (coeffs, original_field) -> PhysicalField
    inverse: Callable[[Dict[str, Any], PhysicalField], PhysicalField]
    #: (coeffs) -> JSON-serialisable summary for an API response
    summarise: Callable[[Dict[str, Any]], Dict[str, Any]]


def register_transform(name: str, description: str = "", params: Optional[Dict] = None,
                       capabilities: Optional[Dict] = None, tags=None):
    return TRANSFORMS.register(name, description=description, params=params,
                               capabilities=capabilities, tags=tags)


def apply_transform(name: str, field: PhysicalField, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a registered transform and return coefficients, reconstruction and a summary.

    The single dispatch point. `engine.py` and `main.py` both call this, so they can no
    longer disagree about what a transform name means - which they previously could, and
    which is the kind of divergence nobody notices until two code paths give different
    answers for the same config.
    """
    spec = TRANSFORMS.get(name)
    coeffs = spec.apply(field, config)
    reconstructed = spec.inverse(coeffs, field)
    mse = float(torch.mean((field.data.to(reconstructed.data.dtype)
                            - reconstructed.data) ** 2))
    return {
        "transform": name,
        "coefficients": coeffs,
        "reconstructed": reconstructed,
        "summary": spec.summarise(coeffs),
        "reconstruction_mse": mse,
    }


def _register_builtins() -> None:
    """Register the transforms that ship with the platform.

    Imported lazily inside the function to keep the module import graph acyclic - the
    transform implementations import `PhysicalField`, which imports `GridSpec`, and a
    top-level import here would close a loop through `engine.py`.
    """
    from src.transform_engine import dtcwt as dtcwt_mod
    from src.transform_engine import stationary as swt_mod
    from src.transform_engine.transforms import SpectralTransformEngine as STE

    # ---------------------------------------------------------------- fft
    def _fft_apply(field, config):
        magnitude, phase = STE.apply_fft2d(field)
        return {"magnitude": magnitude, "phase": phase, "shape": tuple(field.data.shape)}

    register_transform(
        "fft", description="2D real FFT: magnitude and phase.",
        params={}, capabilities={"invertible": True, "shift_invariant_magnitude": True,
                                 "oriented": False, "multiscale": False},
    )(TransformSpec(
        apply=_fft_apply,
        inverse=lambda c, f: STE.inverse_fft2d(c["magnitude"], c["phase"], c["shape"]),
        summarise=lambda c: {"shape": list(c["shape"])},
    ))

    # ---------------------------------------------------------------- dct
    register_transform(
        "dct", description="2D DCT-II with a DCT-III inverse.",
        params={}, capabilities={"invertible": True, "oriented": False, "multiscale": False},
    )(TransformSpec(
        apply=lambda field, config: {"coefficients": STE.apply_dct2d(field)},
        inverse=lambda c, f: STE.inverse_dct2d(c["coefficients"]),
        summarise=lambda c: {"shape": list(c["coefficients"].shape)},
    ))

    # ---------------------------------------------------------------- dwt
    def _dwt_apply(field, config):
        levels = int(config.get("levels", 1))
        out = STE.apply_dwt2d(field, levels=levels)
        out["_levels"] = levels
        out["_shape"] = tuple(field.data.shape)
        return out

    register_transform(
        "dwt", description="Multi-level decimated Haar DWT.",
        params={"levels": "int >= 1"},
        capabilities={"invertible": True, "multiscale": True, "oriented": False,
                      "shift_invariant": False, "parent_grid": False},
    )(TransformSpec(
        apply=_dwt_apply,
        inverse=lambda c, f: STE.inverse_dwt2d(c, levels=c["_levels"],
                                               target_shape=c["_shape"]),
        summarise=lambda c: {"levels": c["_levels"]},
    ))

    # ---------------------------------------------------------------- swt
    def _swt_apply(field, config):
        return swt_mod.apply_swt2d(
            field, levels=int(config.get("levels", 1)),
            wavelet=config.get("wavelet", "haar"), mode=config.get("mode", "periodic"))

    def _swt_summary(c):
        """Full bands plus the R3-compliant energy summary.

        The bands are returned in full rather than reduced to statistics: every SWT band is
        on the parent grid, and a caller inspecting scale structure needs the arrays. Keeping
        the previous payload shape also means the registry refactor changes dispatch only,
        which is what lets the existing tests verify it.
        """
        out = {"LL": c["LL"].tolist(),
               "energy_fractions": swt_mod.swt_energy_fractions(c),
               "meta": dict(c["meta"])}
        for lvl in range(1, c["meta"]["levels"] + 1):
            key = "level_%d" % lvl
            out[key] = {b: c[key][b].tolist() for b in ("LH", "HL", "HH")}
        return out

    register_transform(
        "swt", description="Undecimated (a trous) stationary wavelet transform.",
        params={"levels": "int >= 1", "wavelet": "haar|db2|db3",
                "mode": "periodic|reflect"},
        capabilities={"invertible": True, "multiscale": True, "oriented": False,
                      "shift_invariant": True, "parent_grid": True},
        tags=["wavelet_bank"],
    )(TransformSpec(
        apply=_swt_apply,
        inverse=lambda c, f: swt_mod.inverse_swt2d(c),
        summarise=_swt_summary,
    ))

    # ---------------------------------------------------------------- dtcwt
    def _dtcwt_apply(field, config):
        return dtcwt_mod.apply_dtcwt2d(
            field, levels=int(config.get("levels", 3)),
            level1=config.get("level1", "near_sym_b"),
            qshift=config.get("qshift", "qshift_b"))

    def _dtcwt_summary(c):
        s = dtcwt_mod.subband_energies(c)
        return {
            "levels": c["levels"], "level1": c["level1"], "qshift": c["qshift"],
            "feature_orientations_deg": s["feature_orientations_deg"],
            "wavevector_orientations_deg": s["wavevector_orientations_deg"],
            "orientation_convention": s["orientation_convention"],
            "subband_energy": s["levels"],
            "coefficient_provenance": c["coefficient_provenance"],
        }

    register_transform(
        "dtcwt", description="Kingsbury q-shift dual-tree complex wavelet transform.",
        params={"levels": "int >= 1", "level1": "near_sym_a|near_sym_b|legall",
                "qshift": "qshift_a|qshift_b|qshift_c|qshift_d"},
        capabilities={"invertible": True, "multiscale": True, "oriented": True,
                      "n_orientations": 6, "shift_invariant": "approximate",
                      "parent_grid": False, "complex": True},
        tags=["wavelet_bank"],
    )(TransformSpec(
        apply=_dtcwt_apply,
        inverse=lambda c, f: dtcwt_mod.inverse_dtcwt2d(c),
        summarise=_dtcwt_summary,
    ))

    # ---------------------------------------------------------------- hybrid
    def _hybrid_apply(field, config):
        out = STE.apply_hybrid(
            field, crossover_freq=float(config.get("crossover_freq", 0.5)),
            mixing_weight=float(config.get("mixing_weight", 0.5)))
        out["_shape"] = tuple(field.data.shape)
        return out

    register_transform(
        "hybrid", description="FFT low-pass plus DWT on the high-frequency residual.",
        params={"crossover_freq": "float in (0, 0.5]",
                "mixing_weight": "float in [0, 1]; only used when exact=False"},
        capabilities={"invertible": True, "multiscale": True, "oriented": False},
    )(TransformSpec(
        apply=_hybrid_apply,
        inverse=lambda c, f: STE.inverse_hybrid(c, target_shape=c["_shape"], exact=True),
        summarise=lambda c: {"crossover_freq": c.get("crossover_freq")},
    ))


_register_builtins()
