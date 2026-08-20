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
