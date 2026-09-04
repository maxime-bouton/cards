from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.estimators.base_estimator import BaseEstimator
from cards.io.io_manager import IOManager


@dataclass
class AnalysisArtifacts:
    original: xp.ndarray
    observations: xp.ndarray
    reduced: dict[str, xp.ndarray]
    global_shapes: dict[str, tuple[int, ...]]
    slices: dict[str, tuple[slice, ...]]
    initialisation: xp.ndarray | None
    potential: xp.ndarray | None
    time: dict[str | xp.ndarray] | None


@dataclass
class AnalysisResults:
    artifacts: AnalysisArtifacts
    metrics: dict[str, dict[str, float]]


class AnalysisHook[G, O](Protocol):
    def run_analysis(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        cfg: dict,
        geometry: G,
        obs: O,
        estimators: list[BaseEstimator],
        burnin: int,
        ckpt_dir: Path,
        obs_path: Path,
    ) -> AnalysisResults: ...

    def save_results(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        results: AnalysisResults,
        save_path: Path,
    ) -> None: ...

    def visualize_results(
        self,
        ctx: ExecutionContext,
        io_mng: IOManager,
        results: AnalysisResults,
        save_path: Path,
    ) -> None: ...
