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
from src.core.executor import (
    derive_seeds,
    describe as executor_describe,
    get_executor,
)
from src.core import device as device_policy
from src.benchmarks.seeding import DEFAULT_ROOT_SEED
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

def _resolvable_names(params: Dict[str, Any], step_outputs: Dict[str, Any]) -> List[str]:
    """Everything a `{...}` reference could legitimately name right now."""
    names = list(params)
    for step, out in step_outputs.items():
        if isinstance(out, dict):
            names.extend("%s.%s" % (step, k) for k in out)
    return names


def resolve_value(val: Any, params: Dict[str, Any], step_outputs: Dict[str, Any],
                  step_name: Optional[str] = None) -> Any:
    if isinstance(val, str):
        if val.startswith("{") and val.endswith("}") and val.count("{") == 1 and val.count("}") == 1:
            ref = val[1:-1]
            if "." in ref:
                src_step, key = ref.split(".", 1)
                if src_step in step_outputs and isinstance(step_outputs[src_step], dict) and key in step_outputs[src_step]:
                    return step_outputs[src_step][key]
            elif ref in params:
                return params[ref]
            # An unresolvable reference used to fall through and stay a literal string, so
            # the failure surfaced much later as something like
            # `torch.tensor("{gen.field}") -> must be real number, not NoneType`. Naming the
            # reference and listing what *is* resolvable turns that into a one-line fix.
            raise ReferenceResolutionError(
                val, _resolvable_names(params, step_outputs), step=step_name)
        
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

def execute_run_payload(item, seed=None):
    """Compute one sweep run. **Pure with respect to the database** (T3.5.19).

    This is the unit the Executor distributes, so it must be a module-level function (the
    `process` backend pickles it) and it must not touch a session: a SQLAlchemy connection
    cannot cross a process boundary. It returns everything the parent needs in order to
    persist the run, and the parent - which owns the one session - writes it.

    That split is what makes parallel and serial byte-identical. Workers do arithmetic;
    ordering, identity and persistence stay in one place.
    """
    from src.core import device as device_policy
    from src.experiment_engine import actions

    pipeline = item["pipeline"]
    run_params = item["run_params"]
    run_idx = item["run_idx"]
    dataset_version = item.get("dataset_version", "unknown")

    dev = device_policy.select_device(item.get("device"), run_idx=run_idx)
    if seed is not None:
        device_policy.enable_determinism(seed)

    step_outputs = {}
    steps = []
    # Tracked outside the loop so a failure names the step that *failed*, not the last one
    # that succeeded. The first version used `steps[-1]`, which is the last **appended**
    # step - so a failure in step 2 was reported against step 1. A misattributed error is
    # worse than a vague one: it sends the reader to the wrong place with confidence.
    current_step = "<none>"
    current_action = "<none>"
    try:
        for step in pipeline:
            step_name = step.get("name")
            action = step.get("action")
            args = step.get("args", {})
            current_step, current_action = step_name, action
            resolved_args = resolve_value(args, run_params, step_outputs,
                                          step_name=step_name)
            step_result = actions.execute(action, resolved_args, dev)
            step_outputs[step_name] = step_result
            steps.append({
                "step_name": step_name,
                "action": action,
                "resolved_args": resolved_args,
                "node_type": actions.node_type_for(action),
                "node_value": actions.summarise(action, step_result, resolved_args),
                "dependencies": DeclarativeExperimentEngine._find_step_dependencies(args),
                "dataset_id": resolved_args.get("dataset_id") if action == "slice_dataset" else None,
                "variable": resolved_args.get("variable") if action == "slice_dataset" else None,
                "dataset_version": dataset_version,
            })
        return {
            "ok": True,
            "run_idx": run_idx,
            "seed": seed,
            "steps": steps,
            "results": DeclarativeExperimentEngine._summarize_run_results(step_outputs),
            "device": str(dev),
        }
    except Exception as exc:
        # Returned rather than raised: one failing run in a 20-run sweep must not discard
        # the other 19, and the failing step has to be named (T3.5.14 / D14).
        failing, failing_action = current_step, current_action
        return {
            "ok": False,
            "run_idx": run_idx,
            "seed": seed,
            "steps": steps,
            "error": "Step %r (action %r) failed in run %d: %s: %s" % (
                failing, failing_action, run_idx, type(exc).__name__, exc),
            "error_type": type(exc).__name__,
            "failing_step": failing,
            "failing_action": failing_action,
            # The traceback is kept on the payload (not persisted to the API response) so a
            # sweep failure can be diagnosed without re-running it - which for a long sweep
            # may not be cheap.
            "traceback": __import__("traceback").format_exc(),
            "device": str(dev),
        }


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

            # --- compute: distributed through the Executor seam (T3.5.19) -------------
            exec_cfg = config.get("execution", {}) or {}
            backend = exec_cfg.get("backend", "serial")
            n_workers = int(exec_cfg.get("n_workers", 1))
            root_seed = int(exec_cfg.get("seed", DEFAULT_ROOT_SEED))
            executor = get_executor(backend, n_workers,
                                    threads_per_worker=exec_cfg.get("threads_per_worker"))
            seeds = derive_seeds(root_seed, len(expanded_runs))
            exec_record = executor_describe(executor)
            exec_record["root_seed"] = root_seed
            exec_record.update(device_policy.describe(n_workers=n_workers))

            items = [{"pipeline": pipeline, "run_params": params, "run_idx": i,
                      "dataset_version": dataset_version,
                      "device": exec_cfg.get("device")}
                     for i, params in enumerate(expanded_runs)]
            task_results = executor.map(execute_run_payload, items, seeds)

            # --- persist: serial, one session, submission order ----------------------
            for idx, task in enumerate(task_results):
                run_params = expanded_runs[idx]
                payload = task.value if task.ok else {
                    "ok": False, "run_idx": idx, "seed": task.seed, "steps": [],
                    "error": "%s: %s" % (task.error_type, task.error),
                    "error_type": task.error_type,
                }
                run_id = str(uuid.uuid4())
                run = ExperimentRun(
                    id=run_id,
                    experiment_id=experiment_id,
                    parameters=run_params,
                    status="RUNNING",
                    seed=payload.get("seed"),
                    execution=exec_record,
                    created_at=datetime.datetime.now(datetime.timezone.utc)
                )
                db.add(run)
                db.commit()

                step_nodes = {}
                for step in payload.get("steps", []):
                    node_id = str(uuid.uuid4())
                    node = LineageNode(
                        id=node_id,
                        experiment_id=experiment_id,
                        run_id=run_id,
                        name=step["step_name"],
                        type=step["node_type"],
                        value=step["node_value"]
                    )
                    db.add(node)
                    db.commit()
                    step_nodes[step["step_name"]] = node_id

                    db.add(LineageEdge(id=str(uuid.uuid4()), source_id=code_node_id,
                                       target_id=node_id, relation="executed_by"))
                    for inp_step in step["dependencies"]:
                        if inp_step in step_nodes:
                            db.add(LineageEdge(id=str(uuid.uuid4()),
                                               source_id=step_nodes[inp_step],
                                               target_id=node_id, relation="input_to"))

                    if step.get("dataset_id"):
                        ds_node_id = str(uuid.uuid4())
                        db.add(LineageNode(
                            id=ds_node_id, experiment_id=experiment_id, run_id=run_id,
                            name="dataset_%s" % step["dataset_id"], type="dataset",
                            value={"dataset_id": step["dataset_id"],
                                   "variable": step.get("variable"),
                                   "version": step.get("dataset_version")}))
                        db.commit()
                        db.add(LineageEdge(id=str(uuid.uuid4()), source_id=ds_node_id,
                                           target_id=node_id, relation="sliced_from"))
                    db.commit()

                if payload.get("ok"):
                    run.status = "COMPLETED"
                    run.results = payload["results"]
                else:
                    all_runs_successful = False
                    run.status = "FAILED"
                    run.error_message = payload.get("error")
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
