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
from src.experiment_engine import actions
from src.core.errors import (
    PipelineStepError,
    ReferenceResolutionError,
    SpectralEarthError,
)

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
        """Dispatch one pipeline step through the action registry (T3.5.15, D15).

        This was a 270-line `if/elif` chain that had to be edited to add an action, with
        the same action names repeated in two further chains below. Dispatch now goes
        through `experiment_engine.actions.ACTIONS`, so a new action is a new file plus an
        import - `engine.py` is not touched - and an unknown name raises
        `UnknownNameError`, which lists the valid actions and offers a `did you mean`
        instead of the old bare `ValueError`.
        """
        return actions.execute(action, args, device)

    @staticmethod
    def _get_node_type_for_action(action: str) -> str:
        """Lineage node type, declared by the action itself rather than by a parallel chain."""
        try:
            return actions.node_type_for(action)
        except Exception:
            return "intermediate"

    @staticmethod
    def _create_node_summary(action: str, step_result: Dict[str, Any],
                             resolved_args: Dict[str, Any]) -> Dict[str, Any]:
        """Lineage summary, produced by the action itself (T3.5.15).

        This was the last `elif action == ...` chain in this file, and the third place
        the seven action names were listed. All three now come from one registration.
        """
        return actions.summarise(action, step_result, resolved_args)

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
