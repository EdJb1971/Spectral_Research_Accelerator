import uuid
import itertools
import datetime
import re
import torch
from typing import Dict, Any, List, Optional, Tuple
from src.database.models import Experiment, ExperimentRun, LineageNode, LineageEdge
from src.synthetic_generator.generator import SyntheticFieldGenerator
from src.synthetic_generator.perturbation import PerturbationEngine
from src.boundary_lab.boundary import BoundaryConditionLab
from src.transform_engine.transforms import SpectralTransformEngine
from src.transform_engine import dtcwt as RealDTCWT
from src.transform_engine import stationary as swt_engine
from src.data_layer.adapters import MeteorologicalDataAdapter
from src.analysis_engine.diagnostics import SpectralSpatialAnalysisEngine
from src.analysis_engine.decomposition import ErrorDecompositionEngine
from src.physical_core.field import PhysicalField

def get_execution_device(run_idx: int = 0) -> torch.device:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_id = run_idx % num_gpus
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def resolve_value(val: Any, params: Dict[str, Any], step_outputs: Dict[str, Any]) -> Any:
    if isinstance(val, str):
        if val.startswith("{") and val.endswith("}") and val.count("{") == 1 and val.count("}") == 1:
            ref = val[1:-1]
            if "." in ref:
                step_name, key = ref.split(".", 1)
                if step_name in step_outputs and isinstance(step_outputs[step_name], dict) and key in step_outputs[step_name]:
                    return step_outputs[step_name][key]
            elif ref in params:
                return params[ref]
        
        def replacer(match):
            placeholder = match.group(1)
            if "." in placeholder:
                step_name, key = placeholder.split(".", 1)
                if step_name in step_outputs and isinstance(step_outputs[step_name], dict) and key in step_outputs[step_name]:
                    return str(step_outputs[step_name][key])
            elif placeholder in params:
                return str(params[placeholder])
            return match.group(0)
            
        if "{" in val:
            return re.sub(r"\{([^}]+)\}", replacer, val)
        return val
    elif isinstance(val, list):
        return [resolve_value(item, params, step_outputs) for item in val]
    elif isinstance(val, dict):
        return {k: resolve_value(v, params, step_outputs) for k, v in val.items()}
    return val

class DeclarativeExperimentEngine:
    @staticmethod
    def expand_parameter_matrix(matrix: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not matrix:
            return [{}]
        
        total_combinations = 1
        for val_list in matrix.values():
            if isinstance(val_list, list):
                if len(val_list) > 0:
                    total_combinations *= len(val_list)
        
        if total_combinations > 1000:
            raise ValueError(f"Parameter matrix expansion exceeds maximum limit of 1000 combinations (got {total_combinations}).")
            
        keys, values = zip(*matrix.items())
        experiments = []
        for combination in itertools.product(*values):
            experiments.append(dict(zip(keys, combination)))
        return experiments

    @classmethod
    def execute_experiment(cls, experiment_id: str, session_factory=None):
        """Execute all runs for an experiment.

        session_factory: optional callable returning a SQLAlchemy Session. Defaults to
        the application's SessionLocal. Injectable so that tests (and, later, worker
        processes with their own per-worker sessions) can supply their own binding
        rather than being silently ignored.
        """
        if session_factory is None:
            from src.database.session import SessionLocal as session_factory
        db = session_factory()
        try:
            experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not experiment:
                return
            
            experiment.status = "RUNNING"
            db.commit()

            config = experiment.config
            parameter_matrix = config.get("parameter_matrix", {})
            pipeline = config.get("pipeline", [])
            metadata = config.get("metadata", {})
            code_revision = metadata.get("code_revision", "unknown")
            dataset_version = metadata.get("dataset_version", "unknown")

            expanded_runs = cls.expand_parameter_matrix(parameter_matrix)

            # Create Code Revision Node
            code_node_id = str(uuid.uuid4())
            code_node = LineageNode(
                id=code_node_id,
                experiment_id=experiment_id,
                name="code_revision",
                type="code_revision",
                value={"revision": code_revision}
            )
            db.add(code_node)
            db.commit()

            all_runs_successful = True

            for idx, run_params in enumerate(expanded_runs):
                run_id = str(uuid.uuid4())
                run = ExperimentRun(
                    id=run_id,
                    experiment_id=experiment_id,
                    parameters=run_params,
                    status="RUNNING",
                    created_at=datetime.datetime.now(datetime.timezone.utc)
                )
                db.add(run)
                db.commit()

                device = get_execution_device(idx)
                step_outputs = {}
                step_nodes = {}
                run_successful = True

                try:
                    for step in pipeline:
                        step_name = step.get("name")
                        action = step.get("action")
                        args = step.get("args", {})

                        resolved_args = resolve_value(args, run_params, step_outputs)
                        step_result = cls._execute_action(action, resolved_args, device)
                        step_outputs[step_name] = step_result

                        node_id = str(uuid.uuid4())
                        node_type = cls._get_node_type_for_action(action)
                        node_value = cls._create_node_summary(action, step_result, resolved_args)
                        
                        node = LineageNode(
                            id=node_id,
                            experiment_id=experiment_id,
                            run_id=run_id,
                            name=step_name,
                            type=node_type,
                            value=node_value
                        )
                        db.add(node)
                        db.commit()
                        step_nodes[step_name] = node_id

                        edge_id = str(uuid.uuid4())
                        edge = LineageEdge(
                            id=edge_id,
                            source_id=code_node_id,
                            target_id=node_id,
                            relation="executed_by"
                        )
                        db.add(edge)

                        inputs = cls._find_step_dependencies(args)
                        for inp_step in inputs:
                            if inp_step in step_nodes:
                                edge_id = str(uuid.uuid4())
                                edge = LineageEdge(
                                    id=edge_id,
                                    source_id=step_nodes[inp_step],
                                    target_id=node_id,
                                    relation="input_to"
                                )
                                db.add(edge)

                        if action == "slice_dataset":
                            ds_node_id = str(uuid.uuid4())
                            ds_node = LineageNode(
                                id=ds_node_id,
                                experiment_id=experiment_id,
                                run_id=run_id,
                                name=f"dataset_{resolved_args.get('dataset_id')}",
                                type="dataset",
                                value={
                                    "dataset_id": resolved_args.get("dataset_id"),
                                    "variable": resolved_args.get("variable"),
                                    "version": dataset_version
                                }
                            )
                            db.add(ds_node)
                            db.commit()

                            edge_id = str(uuid.uuid4())
                            edge = LineageEdge(
                                id=edge_id,
                                source_id=ds_node_id,
                                target_id=node_id,
                                relation="sliced_from"
                            )
                            db.add(edge)

                        db.commit()

                    run.status = "COMPLETED"
                    run.results = cls._summarize_run_results(step_outputs)
                    run.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    db.commit()

                except Exception as e:
                    run_successful = False
                    all_runs_successful = False
                    run.status = "FAILED"
                    run.error_message = str(e)
                    run.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    db.commit()

            experiment.status = "COMPLETED" if all_runs_successful else "FAILED"
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _execute_action(action: str, args: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        if action == "generate_synthetic":
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

        elif action == "apply_transform":
            field_data = args.get("field_data")
            transform_type = args.get("transform_type", "fft").lower()
            config = args.get("config", {})
            
            data_tensor = torch.tensor(field_data, dtype=torch.float32, device=device)
            field = PhysicalField(data_tensor)
            
            if transform_type == "fft":
                magnitude, phase = SpectralTransformEngine.apply_fft2d(field)
                reconstructed = SpectralTransformEngine.inverse_fft2d(magnitude, phase, field.data.shape)
                coefficients = {
                    "magnitude": magnitude.tolist(),
                    "phase": phase.tolist()
                }
            elif transform_type == "dct":
                coeffs = SpectralTransformEngine.apply_dct2d(field)
                reconstructed = SpectralTransformEngine.inverse_dct2d(coeffs)
                coefficients = {
                    "coefficients": coeffs.tolist()
                }
            elif transform_type == "dwt":
                levels = config.get("levels", 1)
                coeffs = SpectralTransformEngine.apply_dwt2d(field, levels=levels)
                reconstructed = SpectralTransformEngine.inverse_dwt2d(coeffs, levels=levels, target_shape=field.data.shape)
                serialized_coeffs = {"LL": coeffs["LL"].tolist()}
                for lvl in range(1, levels + 1):
                    serialized_coeffs[f"level_{lvl}"] = {
                        k: v.tolist() for k, v in coeffs[f"level_{lvl}"].items()
                    }
                coefficients = serialized_coeffs
            elif transform_type == "dtcwt":
                # T3.5.6 / D1: the real Kingsbury q-shift DTCWT. The previous
                # `SpectralTransformEngine.apply_dtcwt2d` ran four copies of one Haar filter
                # bank and is retained only as a documented artefact (see transforms.py).
                # TODO(T3.5.15): this branch is registry debt.
                levels = int(config.get("levels", 3))
                coeffs = RealDTCWT.apply_dtcwt2d(
                    field, levels=levels,
                    level1=config.get("level1", "near_sym_b"),
                    qshift=config.get("qshift", "qshift_b"))
                reconstructed = RealDTCWT.inverse_dtcwt2d(coeffs)
                orientation = RealDTCWT.subband_energies(coeffs)
                serialized_coeffs = {"LL": coeffs["LL"].tolist()}
                for lvl in range(1, levels + 1):
                    serialized_coeffs[f"level_{lvl}"] = {
                        "LH_real": coeffs[f"level_{lvl}"]["LH_real"].tolist(),
                        "LH_imag": coeffs[f"level_{lvl}"]["LH_imag"].tolist(),
                        "HL_real": coeffs[f"level_{lvl}"]["HL_real"].tolist(),
                        "HL_imag": coeffs[f"level_{lvl}"]["HL_imag"].tolist(),
                        "HH_real": coeffs[f"level_{lvl}"]["HH_real"].tolist(),
                        "HH_imag": coeffs[f"level_{lvl}"]["HH_imag"].tolist()
                    }
                coefficients = serialized_coeffs
            elif transform_type == "swt":
                # Undecimated / stationary transform (T3.5.7). Unlike 'dwt' every band
                # stays on the parent grid, which is what Phase 4 needs. NOTE: this is
                # another branch on the if/elif chain that defect D15 is about; it moves
                # onto @register_transform in T3.5.15.
                levels = config.get("levels", 1)
                wavelet = config.get("wavelet", "haar")
                mode = config.get("mode", "periodic")
                coeffs_swt = swt_engine.apply_swt2d(field, levels=levels, wavelet=wavelet, mode=mode)
                if mode == "periodic":
                    reconstructed = swt_engine.inverse_swt2d(coeffs_swt)
                else:
                    reconstructed = field  # reflect mode is analysis-only; report no recon error
                serialized_coeffs = {"LL": coeffs_swt["LL"].tolist()}
                for lvl in range(1, levels + 1):
                    serialized_coeffs["level_%d" % lvl] = {
                        k: v.tolist() for k, v in coeffs_swt["level_%d" % lvl].items()
                    }
                serialized_coeffs["energy_fractions"] = swt_engine.swt_energy_fractions(coeffs_swt)
                serialized_coeffs["meta"] = coeffs_swt["meta"]
                coefficients = serialized_coeffs
            elif transform_type == "hybrid":
                crossover_freq = config.get("crossover_freq", 0.5)
                mixing_weight = config.get("mixing_weight", 0.5)
                coeffs = SpectralTransformEngine.apply_hybrid(field, crossover_freq, mixing_weight)
                reconstructed = SpectralTransformEngine.inverse_hybrid(coeffs, field.data.shape)
                serialized_dwt = {"LL": coeffs["dwt_coeffs"]["LL"].tolist()}
                for lvl in range(1, 2):
                    serialized_dwt[f"level_{lvl}"] = {
                        k: v.tolist() for k, v in coeffs["dwt_coeffs"][f"level_{lvl}"].items()
                    }
                coefficients = {
                    "fft_mag": coeffs["fft_mag"].tolist(),
                    "fft_phase": coeffs["fft_phase"].tolist(),
                    "fft_mag_filtered": coeffs["fft_mag_filtered"].tolist(),
                    "dwt_coeffs": serialized_dwt,
                    "mixing_weight": coeffs["mixing_weight"],
                    "crossover_freq": coeffs["crossover_freq"]
                }
            else:
                raise ValueError(f"Unsupported transform type: {transform_type}")
                
            mse = torch.mean((field.data - reconstructed.data) ** 2).item()
            max_err = torch.max(torch.abs(field.data - reconstructed.data)).item()
            
            return {
                "reconstructed_field": reconstructed.data.tolist(),
                "coefficients": coefficients,
                "metrics": {
                    "mean_squared_error": mse,
                    "max_absolute_error": max_err
                }
            }

        elif action == "perturb_field":
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

        elif action == "analyze_boundary":
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

        elif action == "slice_dataset":
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

        elif action == "compute_diagnostics":
            forecast_data = args.get("forecast_data")
            gt_data = args.get("ground_truth_data")
            
            f_tensor = torch.tensor(forecast_data, dtype=torch.float32, device=device)
            g_tensor = torch.tensor(gt_data, dtype=torch.float32, device=device)
            
            f_field = PhysicalField(f_tensor)
            g_field = PhysicalField(g_tensor)
            
            diagnostics = SpectralSpatialAnalysisEngine.compute_diagnostics(f_field, g_field)
            return diagnostics

        elif action == "decompose_errors":
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

        else:
            raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _get_node_type_for_action(action: str) -> str:
        if action in ["generate_synthetic", "slice_dataset"]:
            return "field"
        elif action == "apply_transform":
            return "coefficients"
        elif action in ["compute_diagnostics", "decompose_errors", "perturb_field", "analyze_boundary"]:
            return "metrics"
        return "intermediate"

    @staticmethod
    def _create_node_summary(action: str, step_result: Dict[str, Any], resolved_args: Dict[str, Any]) -> Dict[str, Any]:
        summary = {"action": action}
        if action == "generate_synthetic":
            summary["type"] = resolved_args.get("type")
            summary["shape"] = [len(step_result["field_data"]), len(step_result["field_data"][0])]
            summary["metadata"] = step_result.get("metadata", {})
        elif action == "slice_dataset":
            summary["dataset_id"] = resolved_args.get("dataset_id")
            summary["variable"] = resolved_args.get("variable")
            summary["shape"] = [len(step_result["field_data"]), len(step_result["field_data"][0])]
            summary["metadata"] = step_result.get("metadata", {})
        elif action == "apply_transform":
            summary["transform_type"] = resolved_args.get("transform_type")
            summary["metrics"] = step_result.get("metrics", {})
        elif action == "perturb_field":
            summary["perturbations"] = resolved_args.get("perturbations")
            summary["metrics"] = step_result.get("metrics", {})
        elif action == "analyze_boundary":
            summary["treatment"] = resolved_args.get("treatment")
            summary["pad_width"] = resolved_args.get("pad_width")
            summary["spectral_leakage"] = step_result.get("spectral_leakage")
        elif action == "compute_diagnostics":
            summary["spatial_metrics"] = step_result.get("spatial_metrics", {})
            summary["gradient_errors"] = step_result.get("gradient_errors", {})
        elif action == "decompose_errors":
            summary["scale_decomposition"] = step_result.get("scale_decomposition")
            summary["lead_time_decomposition"] = step_result.get("lead_time_decomposition")
        return summary

    @staticmethod
    def _find_step_dependencies(args: Any) -> List[str]:
        dependencies = []
        if isinstance(args, str):
            if args.startswith("{") and args.endswith("}"):
                ref = args[1:-1]
                if "." in ref:
                    step_name, _ = ref.split(".", 1)
                    dependencies.append(step_name)
        elif isinstance(args, list):
            for item in args:
                dependencies.extend(DeclarativeExperimentEngine._find_step_dependencies(item))
        elif isinstance(args, dict):
            for v in args.values():
                dependencies.extend(DeclarativeExperimentEngine._find_step_dependencies(v))
        return list(set(dependencies))

    @staticmethod
    def _summarize_run_results(step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        summary = {}
        for step_name, output in step_outputs.items():
            if "metrics" in output:
                summary[f"{step_name}_metrics"] = output["metrics"]
            elif "spatial_metrics" in output:
                summary[f"{step_name}_spatial_metrics"] = output["spatial_metrics"]
            elif "scale_decomposition" in output and output["scale_decomposition"] is not None:
                summary[f"{step_name}_scale_decomposition"] = output["scale_decomposition"]
        return summary
