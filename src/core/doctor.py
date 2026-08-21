"""Portable execution capability report.

Run ``python -m src.core.doctor`` on a home laptop, local GPU workstation or allocated HPC
node.  It performs only small local tensor operations and never contacts or submits to a
cluster.  JSON output is stable enough to attach to a support request or experiment record.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any, Dict

import torch

from src.core.device import available_devices, resolve_profile, scheduler_context


def _smoke(device: torch.device) -> Dict[str, Any]:
    """Exercise allocation, FFT and backward propagation on one device."""
    try:
        values = torch.randn(2, 2, 16, 16, device=device, requires_grad=True)
        coefficients = torch.fft.rfft2(values)
        reconstructed = torch.fft.irfft2(coefficients, s=values.shape[-2:])
        reconstructed.square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "status": "PASS",
            "device": str(device),
            "max_reconstruction_error": float(
                torch.max(torch.abs(reconstructed.detach() - values.detach())).cpu()
            ),
            "finite_gradient": bool(torch.isfinite(values.grad).all().cpu()),
        }
    except Exception as exc:  # the report must explain a broken backend, not crash silently
        return {
            "status": "FAIL",
            "device": str(device),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_report(profile: str = "auto") -> Dict[str, Any]:
    """Return the complete capability and smoke-test report."""
    resolved = resolve_profile(profile)
    devices = available_devices()
    smoke = {"cpu": _smoke(torch.device("cpu"))}
    if devices["cuda"]:
        for index in range(devices["cuda_device_count"]):
            name = "cuda:%d" % index
            smoke[name] = _smoke(torch.device(name))
    if devices["mps"]:
        smoke["mps"] = _smoke(torch.device("mps"))
    required_smoke = {"cpu", resolved.device}
    ready = all(
        smoke[name]["status"] == "PASS" for name in required_smoke if name in smoke
    ) and all(name in smoke for name in required_smoke)
    return {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "pytorch": {"version": torch.__version__},
        "devices": devices,
        "scheduler": scheduler_context(),
        "resolved_profile": resolved.to_provenance(),
        "smoke": smoke,
        "ready": ready,
        "all_detected_backends_ready": all(
            result["status"] == "PASS" for result in smoke.values()
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Report SpectralEarth CPU/GPU/HPC execution readiness."
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--profile", choices=("auto", "cpu", "accelerator", "hpc"), default="auto"
    )
    parser.add_argument(
        "--require-accelerator", action="store_true",
        help="return a non-zero exit status when no accelerator is usable",
    )
    args = parser.parse_args(argv)
    try:
        report = build_report(args.profile)
    except (RuntimeError, ValueError) as exc:
        print("SpectralEarth doctor: %s" % exc, file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("SpectralEarth execution doctor")
        print("  platform: %s %s" % (
            report["platform"]["system"], report["platform"]["machine"]))
        print("  PyTorch:  %s" % report["pytorch"]["version"])
        print("  profile:  %(requested)s -> %(device)s" % report["resolved_profile"])
        runtime = report["devices"]["accelerator_runtime"] or "none"
        print("  runtime:  %s" % runtime)
        for name, result in report["smoke"].items():
            print("  smoke %-8s %s" % (name + ":", result["status"]))

    accelerator_pass = any(
        name != "cpu" and result["status"] == "PASS"
        for name, result in report["smoke"].items()
    )
    if args.require_accelerator and not accelerator_pass:
        return 3
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
