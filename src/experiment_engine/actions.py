"""Built-in pipeline actions, registered rather than dispatched by an if/elif chain.

Roadmap T3.5.15 (defect D15, standard E1). These seven action bodies previously lived in a
single 270-line `if/elif` inside `ExperimentEngine._execute_action`, with the action names
repeated in two further chains (`_get_node_type_for_action` and `_create_node_summary`) that
had to be kept in step by hand.

Each action now declares its own node type and summary alongside its implementation, so the
three facts about an action live together instead of in three places that could disagree.
Adding an action means adding a file and importing it; `engine.py` is not touched.

The bodies are moved verbatim from `engine.py` - this slice changes *dispatch*, not
behaviour, and the existing engine tests are what verify that.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional

import torch

from src.core.errors import MissingParameterError
from src.core.registry import Registry
from src.analysis_engine.decomposition import ErrorDecompositionEngine
from src.analysis_engine.diagnostics import SpectralSpatialAnalysisEngine
from src.boundary_lab.boundary import BoundaryConditionLab
from src.data_layer.adapters import MeteorologicalDataAdapter
from src.physical_core.field import PhysicalField
from src.synthetic_generator.generator import SyntheticFieldGenerator
from src.synthetic_generator.perturbation import PerturbationEngine
from src.transform_engine import dtcwt as RealDTCWT
from src.transform_engine import stationary as swt_engine
from src.transform_engine.transforms import SpectralTransformEngine

ACTIONS: Registry["ActionSpec"] = Registry("action")


class ActionSpec(NamedTuple):
    """An action's implementation together with how it appears in the lineage graph."""

    run: Callable[[Dict[str, Any], torch.device], Dict[str, Any]]
    #: Lineage node type: "field", "coefficients", "metrics" or "intermediate".
    node_type: str
    #: (step_result, resolved_args) -> dict merged into the lineage node summary.
    summarise: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def register_action(name: str, description: str = "", params: Optional[Dict] = None,
                    node_type: str = "intermediate", capabilities: Optional[Dict] = None,
                    tags=None):
    """Register a pipeline action. The decorated callable takes ``(args, device)``."""

    def decorator(fn):
        summarise = getattr(fn, "_summarise", lambda result, args: {})
        ACTIONS.register(
            name, description=description or (fn.__doc__ or "").strip().split("\n")[0],
            params=params, capabilities=capabilities, tags=tags,
        )(ActionSpec(run=fn, node_type=node_type, summarise=summarise))
        return fn

    return decorator


def execute(action: str, args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Single dispatch point. Unknown names raise `UnknownNameError`, which lists the rest."""
    return ACTIONS.get(action).run(args, device)


def node_type_for(action: str) -> str:
    return ACTIONS.get(action).node_type



@register_action(
    'generate_synthetic',
    description='Generate a synthetic field (sinusoid, vortex, front, turbulence).',
    params={'type': 'sinusoid|vortex|front|turbulence', 'height': 'int', 'width': 'int', 'params': 'dict, field-type specific'},
    node_type='field',
)
def generate_synthetic(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Generate a synthetic field (sinusoid, vortex, front, turbulence)."""
    field_type = args.get("type", "sinusoid").lower()
    height = args.get("height", 32)
    width = args.get("width", 32)
    params = args.get("params", {})
    
    if field_type == "sinusoid":
        freqs = params.get("frequencies", [(2.0, 2.0)])
        amps = params.get("amplitudes", [1.0])
        phases = params.get("phases", [(0.0, 0.0)])
        field = SyntheticFieldGenerator.generate_sinusoid(height, width, freqs, amps, phases)
    elif field_type == "vortex":
        centers = params.get("centers", [(0.5, 0.5)])
        amps = params.get("amplitudes", [1.0])
        radii = params.get("core_radii", [0.1])
        field = SyntheticFieldGenerator.generate_vortex(height, width, centers, amps, radii)
    elif field_type == "front":
        angle = params.get("angle", 0.0)
        offset = params.get("offset", 0.0)
        width_param = params.get("width_param", 0.1)
        amp = params.get("amplitude", 1.0)
        field = SyntheticFieldGenerator.generate_front(height, width, angle, offset, width_param, amp)
    else:
        raise ValueError(f"Unsupported synthetic field type: {field_type}")
        
    return {
        "field_data": field.data.tolist(),
        "coords": {k: v.tolist() for k, v in field.coords.items()},
        "metadata": field.metadata
    }


@register_action(
    'apply_transform',
    description='Apply a registered spectral transform and reconstruct.',
    params={'field': 'PhysicalField or {step.key} reference', 'transform_type': 'any registered transform', 'config': 'dict passed to the transform'},
    node_type='coefficients',
)
def apply_transform(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Apply a registered spectral transform and reconstruct.

    Dispatch goes through `transform_engine.registry`, which is the single definition of
    what each transform name means. This body previously carried its own six-branch
    `if/elif` - a third copy of the same chain, inherited when the action was moved out of
    `engine.py` and easy to miss precisely because the move was mechanical.
    """
    from src.transform_engine import registry as transform_registry

    field_data = args.get("field_data", args.get("field"))
    if field_data is None:
        raise MissingParameterError(
            "field_data", action="apply_transform",
            required=["field_data", "transform_type"])
    transform_type = str(args.get("transform_type", "fft")).lower()
    config = args.get("config", {}) or {}

    data_tensor = torch.as_tensor(field_data, dtype=torch.float32, device=device)
    field = PhysicalField(data_tensor)

    result = transform_registry.apply_transform(transform_type, field, config)
    reconstructed = result["reconstructed"]
    max_err = float(torch.max(torch.abs(
        field.data.to(reconstructed.data.dtype) - reconstructed.data)))

    return {
        "field_data": reconstructed.data.tolist(),
        "coefficients": result["summary"],
        "transform_type": transform_type,
        "metrics": {
            "mean_squared_error": result["reconstruction_mse"],
            "max_absolute_error": max_err,
        },
    }



@register_action(
    'perturb_field',
    description='Rotate, translate or add noise to a field.',
    params={'field': 'field reference', 'perturbation': 'rotate|translate|noise', 'params': 'dict; noise accepts `seed` for reproducibility'},
    node_type='metrics',
)
def perturb_field(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Rotate, translate or add noise to a field."""
    field_data = args.get("field_data")
    perturbations = args.get("perturbations", [])
    
    original_tensor = torch.tensor(field_data, dtype=torch.float32, device=device)
    original_field = PhysicalField(original_tensor)
    
    current_field = original_field
    for pert in perturbations:
        p_type = pert.get("type", "").lower()
        if p_type == "rotation":
            angle = pert.get("angle")
            if angle is None:
                raise ValueError("Rotation perturbation requires an 'angle' parameter.")
            current_field = PerturbationEngine.rotate(current_field, angle)
        elif p_type == "translation":
            shift_x = pert.get("shift_x")
            shift_y = pert.get("shift_y")
            if shift_x is None or shift_y is None:
                raise ValueError("Translation perturbation requires 'shift_x' and 'shift_y' parameters.")
            current_field = PerturbationEngine.translate(current_field, shift_x, shift_y)
        elif p_type == "noise":
            noise_type = pert.get("noise_type", "gaussian")
            level = pert.get("level", 0.1)
            current_field = PerturbationEngine.add_noise(current_field, noise_type, level)
        else:
            raise ValueError(f"Unsupported perturbation type: {p_type}")
            
    metrics = PerturbationEngine.compute_sensitivity_metrics(original_field, current_field)
    return {
        "perturbed_field": current_field.data.tolist(),
        "metrics": metrics
    }


@register_action(
    'analyze_boundary',
    description='Boundary-treatment artefact analysis.',
    params={'field': 'field reference', 'treatment': 'periodic|zero|reflect|replicate', 'pad_width': 'int', 'window_type': 'none|tukey|hann'},
    node_type='metrics',
)
def analyze_boundary(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Boundary-treatment artefact analysis."""
    field_data = args.get("field_data")
    treatment = args.get("treatment", "zero")
    pad_width = args.get("pad_width", 4)
    window_type = args.get("window_type")
    window_alpha = args.get("window_alpha", 0.1)
    ref_data = args.get("reference_field_data")
    
    data_tensor = torch.tensor(field_data, dtype=torch.float32, device=device)
    field = PhysicalField(data_tensor)
    
    ref_field = None
    if ref_data is not None:
        ref_tensor = torch.tensor(ref_data, dtype=torch.float32, device=device)
        ref_field = PhysicalField(ref_tensor)
        
    analysis = BoundaryConditionLab.analyze_boundary_artefacts(
        field=field,
        treatment=treatment,
        pad_width=pad_width,
        window_type=window_type,
        window_alpha=window_alpha,
        reference_field=ref_field
    )
    return analysis


@register_action(
    'slice_dataset',
    description='Crop a region/variable/time out of a registered dataset.',
    params={'dataset_id': 'registered dataset id', 'variable': 'str', 'region': 'dict with lat/lon bounds', 'time_index': 'int'},
    node_type='field',
)
def slice_dataset(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Crop a region/variable/time out of a registered dataset."""
    dataset_id = args.get("dataset_id")
    variable = args.get("variable")
    time = args.get("time")
    level = args.get("level")
    lat_range = args.get("lat_range")
    lon_range = args.get("lon_range")
    
    field = MeteorologicalDataAdapter.slice_dataset(
        dataset_id=dataset_id,
        variable=variable,
        time=time,
        level=level,
        lat_range=lat_range,
        lon_range=lon_range
    )
    return {
        "field_data": field.data.tolist(),
        "coords": {k: v.tolist() for k, v in field.coords.items()},
        "metadata": field.metadata
    }


@register_action(
    'compute_diagnostics',
    description='Area-weighted error metrics, physical spectra, wavelet energy.',
    params={'forecast': 'field reference', 'ground_truth': 'field reference'},
    node_type='metrics',
)
def compute_diagnostics(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Area-weighted error metrics, physical spectra, wavelet energy."""
    forecast_data = args.get("forecast_data")
    gt_data = args.get("ground_truth_data")
    
    f_tensor = torch.tensor(forecast_data, dtype=torch.float32, device=device)
    g_tensor = torch.tensor(gt_data, dtype=torch.float32, device=device)
    
    f_field = PhysicalField(f_tensor)
    g_field = PhysicalField(g_tensor)
    
    diagnostics = SpectralSpatialAnalysisEngine.compute_diagnostics(f_field, g_field)
    return diagnostics


@register_action(
    'decompose_errors',
    description='Decompose error by scale, boundary distance and lead time.',
    params={'forecast': 'field reference', 'ground_truth': 'field reference', 'boundary_width': 'int'},
    node_type='metrics',
)
def decompose_errors(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Decompose error by scale, boundary distance and lead time."""
    f_data = args.get("forecast_data")
    g_data = args.get("ground_truth_data")
    f_series = args.get("forecast_series")
    g_series = args.get("ground_truth_series")
    lead_times = args.get("lead_times")
    boundary_width = args.get("boundary_width", 8)
    
    scale_decomp = None
    boundary_decomp = None
    lead_time_decomp = None
    
    if f_data is not None and g_data is not None:
        f_tensor = torch.tensor(f_data, dtype=torch.float32, device=device)
        g_tensor = torch.tensor(g_data, dtype=torch.float32, device=device)
        
        scale_decomp = ErrorDecompositionEngine.decompose_by_scale(f_tensor, g_tensor)
        boundary_decomp = ErrorDecompositionEngine.decompose_by_boundary(
            f_tensor, g_tensor, boundary_width=boundary_width
        )
        
    if f_series is not None and g_series is not None and lead_times is not None:
        f_tensors = [torch.tensor(f, dtype=torch.float32, device=device) for f in f_series]
        g_tensors = [torch.tensor(g, dtype=torch.float32, device=device) for g in g_series]
        
        lead_time_decomp = ErrorDecompositionEngine.decompose_by_lead_time(
            f_tensors, g_tensors, lead_times
        )
        
    return {
        "scale_decomposition": scale_decomp,
        "boundary_decomposition": boundary_decomp,
        "lead_time_decomposition": lead_time_decomp
    }

# --------------------------------------------------------------------- lineage summaries
#
# Each action's lineage summary lives next to its implementation. These were the last
# `elif action == ...` chain in `engine.py`: a third place where the seven action names were
# repeated and had to be kept in step with the other two by hand.


def _shape_of(step_result):
    data = step_result.get("field_data")
    if not data:
        return None
    return [len(data), len(data[0])]


def _summarise_generate_synthetic(result, args):
    return {"type": args.get("type"), "shape": _shape_of(result),
            "metadata": result.get("metadata", {})}


def _summarise_slice_dataset(result, args):
    return {"dataset_id": args.get("dataset_id"), "variable": args.get("variable"),
            "shape": _shape_of(result), "metadata": result.get("metadata", {})}


def _summarise_apply_transform(result, args):
    return {"transform_type": args.get("transform_type"),
            "metrics": result.get("metrics", {})}


def _summarise_perturb_field(result, args):
    return {"perturbations": args.get("perturbations"), "metrics": result.get("metrics", {})}


def _summarise_analyze_boundary(result, args):
    return {"treatment": args.get("treatment"), "pad_width": args.get("pad_width"),
            "spectral_leakage": result.get("spectral_leakage")}


def _summarise_compute_diagnostics(result, args):
    return {"spatial_metrics": result.get("spatial_metrics", {}),
            "gradient_errors": result.get("gradient_errors", {})}


def _summarise_decompose_errors(result, args):
    return {"scale_decomposition": result.get("scale_decomposition"),
            "lead_time_decomposition": result.get("lead_time_decomposition")}


_SUMMARISERS = {
    "generate_synthetic": _summarise_generate_synthetic,
    "slice_dataset": _summarise_slice_dataset,
    "apply_transform": _summarise_apply_transform,
    "perturb_field": _summarise_perturb_field,
    "analyze_boundary": _summarise_analyze_boundary,
    "compute_diagnostics": _summarise_compute_diagnostics,
    "decompose_errors": _summarise_decompose_errors,
}

for _name, _fn in _SUMMARISERS.items():
    _entry = ACTIONS.entry(_name)
    ACTIONS.register(_name, description=_entry.description, params=_entry.params,
                     capabilities=_entry.capabilities, tags=_entry.tags, replace=True)(
        ActionSpec(run=_entry.value.run, node_type=_entry.value.node_type, summarise=_fn))


def summarise(action: str, step_result: dict, resolved_args: dict) -> dict:
    """Lineage node summary for one step, from the action's own summariser."""
    out = {"action": action}
    try:
        spec = ACTIONS.get(action)
    except Exception:
        return out
    extra = spec.summarise(step_result, resolved_args) or {}
    out.update({k: v for k, v in extra.items() if v is not None or k in extra})
    return out


@register_action(
    'slice_sequence',
    description='Crop a dataset over time into a FieldSequence, stored as an artifact.',
    params={'dataset_id': 'registered dataset id', 'variable': 'str',
            'time_range': '[start, end] ISO timestamps', 'level': 'float hPa',
            'lat_range': '[min, max]', 'lon_range': '[min, max]'},
    node_type='field',
)
def slice_sequence(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Crop a dataset over time (roadmap T4A.4).

    **The result is a handle, not a payload.** A 100-frame 64x64 sequence is 3.3 MB of floats;
    as JSON-encoded nested lists in a `step_outputs` dict and again in a lineage row it is the
    scaling wall `architecture.md` has flagged since the first audit. The array goes to the
    content-addressed store and what travels between steps is a reference plus a summary -
    which `resolve_value` dereferences automatically, so a downstream action written before
    the store existed still receives an array.
    """
    from src.artifact_store import store as artifact_store
    from src.data_layer.adapters import MeteorologicalDataAdapter

    time_range = args.get("time_range")
    if time_range is not None:
        time_range = tuple(time_range)
    lat_range = args.get("lat_range")
    lon_range = args.get("lon_range")

    sequence = MeteorologicalDataAdapter.slice_sequence(
        dataset_id=args["dataset_id"],
        variable=args["variable"],
        time_range=time_range,
        level=args.get("level"),
        lat_range=tuple(lat_range) if lat_range else None,
        lon_range=tuple(lon_range) if lon_range else None,
        max_frames=int(args.get("max_frames", 2000)),
    )
    handle = artifact_store.get_store().put(
        sequence, name="%s_%s" % (args["dataset_id"], args["variable"]))
    return {
        "sequence_ref": handle.ref,
        "handle": handle.to_dict(),
        "summary": sequence.summary(),
    }


def _summarise_slice_sequence(result, args):
    """Handle plus summary in the lineage row - never the frames."""
    return {"dataset_id": args.get("dataset_id"), "variable": args.get("variable"),
            "time_range": args.get("time_range"),
            "sequence_ref": result.get("sequence_ref"),
            "sequence": result.get("summary", {})}


# `ActionSpec` is frozen, so the summariser is attached by re-registering the entry the way
# the module's existing loop does it, rather than by mutating the spec in place.
_SUMMARISERS["slice_sequence"] = _summarise_slice_sequence
_entry = ACTIONS.entry("slice_sequence")
ACTIONS.register("slice_sequence", description=_entry.description, params=_entry.params,
                 capabilities=_entry.capabilities, tags=_entry.tags, replace=True)(
    ActionSpec(run=_entry.value.run, node_type=_entry.value.node_type,
               summarise=_summarise_slice_sequence))


# ============================================================ Phase 4B: the wavelet bank

def _sequence_from(value: Any, parameter: str = "sequence"):
    """Accept a `FieldSequence`, an artifact ref, or a `{step.sequence_ref}` result.

    A raw numpy array is refused *by name*. `resolve_value` dereferences a literal
    ``artifact://...`` into an array before an action ever sees it, and an array has no time
    axis and no grid - the two things this action needs most. Silently accepting one would
    mean inventing a cadence, so the refusal names the fix instead.
    """
    from src.artifact_store import store as artifact_store
    from src.core.errors import InvalidParameterError
    from src.physical_core.sequence import FieldSequence

    if isinstance(value, FieldSequence):
        return value
    if isinstance(value, dict):
        for key in ("sequence_ref", "ref"):
            if key in value:
                return _sequence_from(value[key], parameter)
    if isinstance(value, str) and artifact_store.is_ref(value):
        return artifact_store.get_store().load_sequence(value)
    raise InvalidParameterError(
        parameter, type(value).__name__,
        "a FieldSequence or an artifact reference. A bare array is not a sequence: it has no "
        "time coordinate and no grid, and reconstructing them would mean assuming a cadence "
        "the record may not have. Reference the producing step as '{step.sequence_ref}', "
        "which passes the reference rather than the dereferenced payload")


@register_action(
    'decompose_bank',
    description='Decompose a FieldSequence over a wavelet bank into CoefficientField artifacts.',
    params={'sequence': 'a {step.sequence_ref} from slice_sequence, or a FieldSequence',
            'wavelet_bank': 'declarative block: families, scales, orientations, levels_hpa',
            'keep_native': 'bool; retain native coefficients so the result can be inverted'},
    node_type='coefficients',
)
def decompose_bank(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Run every combination of a `WaveletBank` and store each result (roadmap T4B.3).

    One artifact per combination, and **references in the step output**. A single
    `CoefficientField` over 10 frames of 64x64 at 4 scales and 6 orientations is ~10M complex
    numbers; a bank of eight combinations is eight of those, which is precisely the payload
    the pre-4A lineage seam could not carry.
    """
    from src.artifact_store import store as artifact_store
    from src.core.errors import InvalidParameterError
    from src.transform_engine.bank import WaveletBank, select_orientations
    from src.transform_engine.coefficient_field import decompose_sequence

    bank = WaveletBank.from_config(args.get("wavelet_bank") or {})
    keep_native = bool(args.get("keep_native", False))

    # T4B.4: when the bank has a vertical dimension it needs one sequence *per level*.
    # Decomposing a single sequence once per level and labelling the results 850 hPa and
    # 500 hPa would produce identical coefficient arrays under different labels - a fabricated
    # vertical structure that every cross-level statistic downstream would then measure. So a
    # bank with `levels_hpa` requires `sequences` and refuses to guess.
    sequences: Dict[Any, Any] = {}
    if bank.levels_hpa:
        supplied = args.get("sequences")
        if not isinstance(supplied, dict):
            raise InvalidParameterError(
                "sequences", type(supplied).__name__,
                "a mapping of pressure level to sequence reference, e.g. "
                "{'850': '{slice850.sequence_ref}', '500': '{slice500.sequence_ref}'}. This "
                "bank declares levels_hpa=%s, and decomposing one sequence repeatedly would "
                "label identical coefficients with different pressures"
                % (list(bank.levels_hpa),))
        sequences = {float(k): _sequence_from(v, "sequences[%s]" % k)
                     for k, v in supplied.items()}
        missing = [lev for lev in bank.levels_hpa if float(lev) not in sequences]
        if missing:
            raise InvalidParameterError(
                "sequences", sorted(sequences), "a sequence for every declared level; "
                "missing %s" % (missing,))
    else:
        sequences = {None: _sequence_from(args.get("sequence"), "sequence")}

    store = artifact_store.get_store()
    results = []
    for combination in bank.combinations():
        family = combination["wavelet_family"]
        level_hpa = combination.get("level_hpa")
        sequence = sequences[float(level_hpa)] if level_hpa is not None else sequences[None]
        config = {k: v for k, v in combination.items()
                  if k not in ("wavelet_family", "orientations", "level_hpa")}
        field = decompose_sequence(sequence, family=family, config=config,
                                   keep_native=keep_native)
        field = select_orientations(field, combination.get("orientations"))
        if level_hpa is not None:
            field.level = float(level_hpa)
        handle = store.put(field, name="bank_%s_l%s" % (family, combination.get("levels")))
        results.append({
            "wavelet_family": family,
            "levels": combination.get("levels"),
            "level_hpa": combination.get("level_hpa"),
            "coefficients_ref": handle.ref,
            "handle": handle.to_dict(),
            "summary": field.summary(),
        })

    return {"bank": bank.summary(), "n_combinations": len(results), "results": results,
            "refs": [r["coefficients_ref"] for r in results]}


def _summarise_decompose_bank(result, args):
    """Refs and per-combination statistics in the lineage row - never the coefficients."""
    return {
        "bank": result.get("bank"),
        "n_combinations": result.get("n_combinations"),
        "combinations": [
            {"wavelet_family": r["wavelet_family"], "levels": r["levels"],
             "level_hpa": r["level_hpa"], "coefficients_ref": r["coefficients_ref"],
             "shape": r["summary"]["shape"],
             "resampled_to_parent": r["summary"]["resampled_to_parent"],
             "mean_energy_fraction": r["summary"]["mean_energy_fraction"]}
            for r in result.get("results", [])],
    }


@register_action(
    'extract_scale_signature',
    description='Per (time, scale) energy and energy fraction from a CoefficientField.',
    params={'coefficients': 'a {step.coefficients_ref} from decompose_bank',
            'normalise': 'bool; report energy fractions as well as raw energy'},
    node_type='metrics',
)
def extract_scale_signature(args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Collapse a `CoefficientField` to a per-`(time, scale)` energy signature (T4B.3).

    **Scope, so the roadmap is not over-claimed:** this is the energy half of rule R3 - energy
    per scale, and the fraction of each frame's energy that scale carries. The full
    `ScaleSignature` of T4C.1 adds participation ratio, the Gini coefficient and threshold
    counts. This action is the plumbing that 4C extends, not 4C delivered early.

    Energy *fractions* rather than raw energy are what make signatures comparable: raw energy
    scales with the amplitude of the field, so the same structure in two fields recorded in
    different units would produce two different signatures.
    """
    from src.artifact_store import store as artifact_store
    from src.core.errors import InvalidParameterError
    from src.transform_engine.coefficient_field import CoefficientField

    value = args.get("coefficients")
    if isinstance(value, dict) and "coefficients_ref" in value:
        value = value["coefficients_ref"]
    if isinstance(value, CoefficientField):
        field = value
    elif isinstance(value, str) and artifact_store.is_ref(value):
        field = artifact_store.get_store().load_coefficient_field(value)
    else:
        raise InvalidParameterError(
            "coefficients", type(value).__name__,
            "a CoefficientField or an artifact reference. Reference the producing step as "
            "'{step.coefficients_ref}', which passes the reference rather than the "
            "dereferenced array - an array alone has lost its scale and orientation labels")

    energy = field.energy()
    fractions = field.energy_fractions()
    per_scale = fractions.sum(dim=2)
    dominant = torch.argmax(per_scale, dim=1)

    return {
        "scales": [str(s) for s in field.scales],
        "orientations": [str(o) for o in field.orientations],
        "times_seconds": [float(t) for t in field.times],
        "energy": energy.tolist(),
        "energy_fraction": fractions.tolist(),
        "energy_fraction_per_scale": per_scale.tolist(),
        "dominant_scale_per_time": [field.scales[int(i)] for i in dominant],
        "wavelet_family": field.wavelet_family,
        "orientation_convention": field.orientation_convention,
        "scale_wavelength_bands": field.scale_wavelength_bands(),
        "scope": ("energy and energy fraction only; participation ratio, Gini and threshold "
                  "counts are T4C.1"),
    }


def _summarise_extract_scale_signature(result, args):
    """Small enough to embed whole: `(time, scale)` is the one reduction that always fits."""
    return {"wavelet_family": result.get("wavelet_family"),
            "scales": result.get("scales"),
            "energy_fraction_per_scale": result.get("energy_fraction_per_scale"),
            "dominant_scale_per_time": result.get("dominant_scale_per_time"),
            "scope": result.get("scope")}


for _bank_name, _bank_fn in (("decompose_bank", _summarise_decompose_bank),
                             ("extract_scale_signature", _summarise_extract_scale_signature)):
    _SUMMARISERS[_bank_name] = _bank_fn
    _bank_entry = ACTIONS.entry(_bank_name)
    ACTIONS.register(_bank_name, description=_bank_entry.description,
                     params=_bank_entry.params, capabilities=_bank_entry.capabilities,
                     tags=_bank_entry.tags, replace=True)(
        ActionSpec(run=_bank_entry.value.run, node_type=_bank_entry.value.node_type,
                   summarise=_bank_fn))
