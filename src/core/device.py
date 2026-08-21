"""Device and thread policy (roadmap T3.5.21 / defect D18, standard E9).

The previous `get_execution_device` did `run_idx % num_gpus` and returned CPU otherwise. Two
problems: it never considered Apple MPS, and the round-robin was cosmetic because sweeps ran
strictly sequentially - one run at a time cannot use four GPUs.

**Threads are the part people get wrong, and it is the one that matters on a laptop.** PyTorch
defaults to one intra-op thread per physical core. Running 8 sweep runs in parallel, each
spawning 8 BLAS threads, gives 64 threads fighting over 8 cores: the machine is fully loaded,
every thread is slower than it would be alone, and total throughput *falls*. That is why
:func:`thread_budget` exists and why the executor sets it inside each worker rather than
leaving the default. `E10`'s cost model depends on this being explicit rather than emergent.

**Determinism is a declared property, not a hope.** `describe()` records the device, the
thread counts and the relevant PyTorch flags into the run's provenance, so a result that
cannot be reproduced can at least be *explained*.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

#: Environment override, e.g. `SPECTRAL_DEVICE=cpu` to force CPU on a CUDA machine.
DEVICE_ENV_VAR = "SPECTRAL_DEVICE"
PROFILE_ENV_VAR = "SPECTRAL_PROFILE"
THREADS_ENV_VAR = "SPECTRAL_THREADS"
VALID_PROFILES = ("auto", "cpu", "accelerator", "hpc")


@dataclass(frozen=True)
class ExecutionProfile:
    """Resolved execution placement, suitable for run provenance."""

    requested: str
    device: str
    scheduler: Optional[str]
    job_id: Optional[str]
    local_rank: int
    fallback_used: bool

    def to_provenance(self) -> Dict[str, Any]:
        return asdict(self)


def scheduler_context(environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Detect an active batch allocation without importing a scheduler client.

    This deliberately recognises allocation environment variables only; it does not submit,
    cancel or inspect jobs.  The laptop path therefore has no cluster dependency.
    """
    env = os.environ if environ is None else environ
    if env.get("SLURM_JOB_ID"):
        return {
            "scheduler": "slurm",
            "job_id": env["SLURM_JOB_ID"],
            "local_rank": int(env.get("SLURM_LOCALID", env.get("LOCAL_RANK", "0"))),
        }
    if env.get("PBS_JOBID"):
        return {
            "scheduler": "pbs",
            "job_id": env["PBS_JOBID"],
            "local_rank": int(env.get("OMPI_COMM_WORLD_LOCAL_RANK", env.get("LOCAL_RANK", "0"))),
        }
    if env.get("LSB_JOBID"):
        return {
            "scheduler": "lsf",
            "job_id": env["LSB_JOBID"],
            "local_rank": int(env.get("OMPI_COMM_WORLD_LOCAL_RANK", env.get("LOCAL_RANK", "0"))),
        }
    return {"scheduler": None, "job_id": None, "local_rank": 0}


def available_devices() -> Dict[str, Any]:
    """What this machine actually offers. Cheap; safe to call in a health endpoint."""
    # PyTorch intentionally exposes AMD ROCm devices through the torch.cuda API.  Preserve the
    # API-facing ``cuda`` fields for compatibility, but record the actual runtime so provenance
    # never labels an AMD run as NVIDIA CUDA merely because both use torch.device("cuda").
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        accelerator_runtime = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    elif bool(getattr(torch.backends, "mps", None)
              and torch.backends.mps.is_available()):
        accelerator_runtime = "mps"
    else:
        accelerator_runtime = None
    info: Dict[str, Any] = {
        "cpu": True,
        "cuda": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "mps": bool(getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()),
        "accelerator_runtime": accelerator_runtime,
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
    }
    if info["cuda"]:
        info["cuda_devices"] = [torch.cuda.get_device_name(i)
                                for i in range(info["cuda_device_count"])]
    return info


def _select_device_raw(prefer: Optional[str] = None, run_idx: int = 0) -> torch.device:
    """Choose a device without interpreting an execution profile.

    ``run_idx`` spreads concurrent runs across multiple CUDA/ROCm devices. It is only meaningful
    once runs actually execute concurrently (T3.5.19); before that it was decorative, which
    is worth saying plainly rather than leaving a reader to assume multi-GPU was working.
    """
    requested = prefer or os.environ.get(DEVICE_ENV_VAR)
    if requested:
        requested = requested.strip().lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "%s=%r was requested but CUDA/ROCm is not available on this machine. "
                    "Available: %s. Unset the variable to fall back automatically."
                    % (DEVICE_ENV_VAR, requested, available_devices()))
            return torch.device(requested)
        if requested == "mps":
            if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "%s=mps was requested but MPS is not available on this machine."
                    % DEVICE_ENV_VAR)
            return torch.device("mps")
        raise ValueError(
            "unknown device %r; expected 'cpu', 'mps', or 'cuda[:N]'" % requested)

    if torch.cuda.is_available():
        return torch.device("cuda:%d" % (run_idx % max(1, torch.cuda.device_count())))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_device(prefer: Optional[str] = None, run_idx: int = 0) -> torch.device:
    """Choose a device under the active portable execution profile.

    An explicit function argument or ``SPECTRAL_DEVICE`` has highest priority. Otherwise
    ``SPECTRAL_PROFILE`` controls whether CPU fallback is allowed and whether a scheduler
    allocation is mandatory.
    """
    if prefer is not None or os.environ.get(DEVICE_ENV_VAR):
        return _select_device_raw(prefer, run_idx)
    profile = os.environ.get(PROFILE_ENV_VAR, "auto").strip().lower()
    if profile not in VALID_PROFILES:
        raise ValueError(
            "unknown execution profile %r; expected one of %s"
            % (profile, ", ".join(VALID_PROFILES))
        )
    if profile == "cpu":
        return torch.device("cpu")
    if profile == "hpc":
        context = scheduler_context()
        if context["scheduler"] is None:
            raise RuntimeError(
                "SPECTRAL_PROFILE=hpc requires an active Slurm, PBS or LSF allocation; "
                "run through the scheduler rather than on a login node, or use auto locally"
            )
        run_idx = context["local_rank"]
    device = _select_device_raw(run_idx=run_idx)
    if profile == "accelerator" and device.type == "cpu":
        raise RuntimeError(
            "SPECTRAL_PROFILE=accelerator requested a GPU/MPS device, but only CPU is "
            "available. Install a supported PyTorch accelerator build or use auto/cpu."
        )
    return device


def resolve_profile(profile: Optional[str] = None) -> ExecutionProfile:
    """Resolve ``auto|cpu|accelerator|hpc`` into one explicit execution placement.

    ``auto`` is the dependency-free default: fastest available local device, CPU included.
    ``accelerator`` refuses a CPU fallback. ``hpc`` additionally refuses to run outside an
    active scheduler allocation, protecting shared login nodes from accidental workloads.
    An explicit ``SPECTRAL_DEVICE`` still wins inside the selected profile.
    """
    requested = (profile or os.environ.get(PROFILE_ENV_VAR, "auto")).strip().lower()
    if requested not in VALID_PROFILES:
        raise ValueError(
            "unknown execution profile %r; expected one of %s"
            % (requested, ", ".join(VALID_PROFILES))
        )
    scheduler = scheduler_context()
    explicit_device = os.environ.get(DEVICE_ENV_VAR)

    if requested == "cpu":
        if explicit_device and explicit_device.strip().lower() != "cpu":
            raise RuntimeError(
                "%s=cpu conflicts with %s=%r; remove one override"
                % (PROFILE_ENV_VAR, DEVICE_ENV_VAR, explicit_device)
            )
        device = torch.device("cpu")
        fallback = False
    elif requested == "hpc":
        if scheduler["scheduler"] is None:
            raise RuntimeError(
                "SPECTRAL_PROFILE=hpc requires an active Slurm, PBS or LSF allocation; "
                "run through the scheduler rather than on a login node, or use auto locally"
            )
        device = _select_device_raw(run_idx=scheduler["local_rank"])
        fallback = device.type == "cpu"
    else:
        device = _select_device_raw(run_idx=scheduler["local_rank"])
        fallback = device.type == "cpu"
        if requested == "accelerator" and fallback:
            raise RuntimeError(
                "SPECTRAL_PROFILE=accelerator requested a GPU/MPS device, but only CPU is "
                "available. Install a supported PyTorch accelerator build or use auto/cpu."
            )

    return ExecutionProfile(
        requested=requested,
        device=str(device),
        scheduler=scheduler["scheduler"],
        job_id=scheduler["job_id"],
        local_rank=int(scheduler["local_rank"]),
        fallback_used=fallback,
    )


def thread_budget(n_workers: int, total_threads: Optional[int] = None) -> int:
    """Intra-op threads each worker should use, so `n_workers` do not oversubscribe.

    Without this, `n_workers` processes each take the PyTorch default of one thread per
    core. The result is `n_workers x n_cores` threads on `n_cores` cores - every one of them
    slower, and total throughput *lower* than with fewer workers. The oversubscribed case is
    measured explicitly in `VERIFICATION.md` rather than asserted, because the effect is
    counter-intuitive enough that a number is more convincing than an argument.
    """
    if total_threads is None:
        total_threads = int(os.environ.get(THREADS_ENV_VAR, 0)) or os.cpu_count() or 1
    return max(1, total_threads // max(1, n_workers))


def configure_threads(n_threads: int) -> Dict[str, int]:
    """Set intra-op threads for this process, and report what was actually applied.

    Also sets the BLAS environment variables, which PyTorch does *not* control: NumPy and
    SciPy spawn their own pools, and on a `process` backend those inherit the parent's
    settings unless told otherwise. They are read at library import, so setting them here
    only affects workers started afterwards - which is exactly the process-backend case.
    """
    n_threads = max(1, int(n_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    try:
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        # Raises if a parallel region has already started; harmless, and not worth
        # failing a run over.
        pass
    return {"torch_num_threads": torch.get_num_threads(), "requested": n_threads}


def enable_determinism(seed: Optional[int] = None) -> Dict[str, Any]:
    """Put the process into the most reproducible mode available, and say what that is.

    Deterministic algorithms are *requested*, not guaranteed: some CUDA kernels have no
    deterministic implementation, and `warn_only=True` keeps those usable rather than
    aborting a run. The returned record states which mode was actually achieved, so a
    reproducibility claim can be checked against it instead of assumed.
    """
    record: Dict[str, Any] = {"requested_deterministic": True}
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        record["seed"] = int(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        record["deterministic_algorithms"] = "warn_only"
    except Exception as exc:
        record["deterministic_algorithms"] = "unavailable: %s" % type(exc).__name__
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["cudnn_deterministic"] = True
    return record


def describe(device: Optional[torch.device] = None,
             n_workers: int = 1) -> Dict[str, Any]:
    """Provenance record for how and where a run executed (standard E4/E5)."""
    if device is None:
        resolved = resolve_profile()
        device = torch.device(resolved.device)
    else:
        context = scheduler_context()
        resolved = ExecutionProfile(
            requested="explicit",
            device=str(device),
            scheduler=context["scheduler"],
            job_id=context["job_id"],
            local_rank=int(context["local_rank"]),
            fallback_used=device.type == "cpu",
        )
    return {
        "device": str(device),
        "device_type": device.type,
        "available": available_devices(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "n_workers": n_workers,
        "thread_budget_per_worker": thread_budget(n_workers),
        "env_override": os.environ.get(DEVICE_ENV_VAR),
        "execution_profile": resolved.to_provenance(),
    }


def to_device(obj: Any, device: torch.device) -> Any:
    """Move a tensor, a `PhysicalField`, or a (possibly nested) container of them.

    Exists because the codebase mixes CPU-constructed tensors with a selected device in
    places (part of D18). A single helper is easier to audit than scattered `.to()` calls.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if hasattr(obj, "data") and isinstance(getattr(obj, "data", None), torch.Tensor):
        obj.data = obj.data.to(device)
        return obj
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    return obj
