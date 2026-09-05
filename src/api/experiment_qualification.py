"""HTTP surface for the TG17.10 release qualification ledger."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request

from src.core.experiment_qualification import (execute_offline_qualification,
                                               qualification_plan)


router = APIRouter(prefix="/api/v1/experiment-qualification",
                   tags=["experiment-qualification"])


def _root(request: Request) -> Path:
    value = getattr(request.app.state, "experiment_run_dir", None)
    return Path(value if value is not None else os.getenv("EXPERIMENT_RUN_DIR",
                                                          "data/experiment_runs"))


@router.get("")
async def plan() -> Dict[str, Any]:
    """Return the complete gate, including everything this process cannot certify."""
    return qualification_plan()


@router.post("/rehearse")
async def rehearse(request: Request) -> Dict[str, Any]:
    """Run deterministic apparatus checks only; network and scientific values stay closed."""
    return execute_offline_qualification(_root(request))


__all__ = ["router"]
