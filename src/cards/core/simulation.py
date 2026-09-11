# authors: M. Bouton, S. Despierres, P.-A. Thouvenin, P. Chainais, A. Repetti
#
# reference: M. Bouton, P.-A. Thouvenin, A. Repetti, P. Chainais. A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems. IEEE Transactions on Computational Imaging, 2026, 12, pp.839-849. (https://dx.doi.org/10.1109/TCI.2026.3685151)

import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from cards.core.execution_context import ExecutionContext
from cards.core.validation import DefaultSimulationConfig, SimulationConfig
from cards.estimators.base_estimator import BaseEstimator
from cards.hooks.analysis_hook import AnalysisHook
from cards.hooks.geometry_hook import GeometryHook
from cards.hooks.mcmc_hook import McmcHook
from cards.hooks.observations_hook import ObservationsHook
from cards.hooks.paths_hook import PathsHook
from cards.io.io_manager import IOManager
from cards.io.path_builder import PathBuilder
from cards.io.utils import parse_args, read_json
from cards.logger import build_logger
from cards.samplers import SamplerParameters
from cards.samplers.sampler import Sampler


class Simulation[G, O]:
    def __init__(
        self,
        mode: Literal["serial", "mpi"],
        device: Literal["cpu", "gpu"],
        cfg: dict | str | Path | SimulationConfig,
        geometry_hk: GeometryHook[G],
        obs_hk: ObservationsHook[G, O],
        mcmc_hk: McmcHook[G, O] | None = None,
        analysis_hk: AnalysisHook[G, O] | None = None,
        paths_hk: PathsHook | None = None,
    ) -> None:

        if analysis_hk is not None and mcmc_hk is None:
            raise ValueError(
                "An AnalysisHook has been provided without a McmcHook. "
                "The McmcHook is required to build the estimators for analysis."
            )

        self.geometry_hk = geometry_hk
        self.obs_hk = obs_hk
        self.mcmc_hk = mcmc_hk
        self.analysis_hk = analysis_hk
        self.paths_hk = paths_hk

        self.ctx = ExecutionContext(mode, device)

        if isinstance(cfg, SimulationConfig):
            self.cfg = cfg
        else:
            if isinstance(cfg, dict):
                raw_dict = cfg
                anchor_dir = Path.cwd()
            else:
                cfg_path = Path(cfg).resolve()
                raw_dict = read_json(cfg_path)
                anchor_dir = cfg_path.parent
            self.cfg = DefaultSimulationConfig.model_validate(
                raw_dict, context={"anchor_dir": anchor_dir}
            )

        fn_obs_rel_path = paths_hk.fn_obs_rel_path if paths_hk else None
        fn_ckpt_rel_path = paths_hk.fn_ckpt_rel_path if paths_hk else None

        self.paths = PathBuilder(self.cfg, self.ctx, fn_obs_rel_path, fn_ckpt_rel_path)

        self.io_mng = IOManager(self.ctx)
        self.log = build_logger(self.ctx.rank, self.paths.get_log_path())

    @classmethod
    def from_cli(
        cls,
        geom_hk: GeometryHook[G],
        obs_hk: ObservationsHook[G, O],
        mcmc_hk: McmcHook[G, O] | None = None,
        analysis_hk: AnalysisHook[G, O] | None = None,
        paths_hk: PathsHook | None = None,
    ) -> "Simulation":
        args = parse_args()
        # ! only for debugging
        # args.config = "examples/new_config_tv.json"
        # args.mode = "mpi"
        # args.device = "cpu"
        # !
        return cls(
            args.mode,
            args.device,
            args.config,
            geom_hk,
            obs_hk,
            mcmc_hk,
            analysis_hk,
            paths_hk,
        )

    def run(self) -> None:
        try:
            geom = self._run_geometry_phase()
            obs = self._run_observations_phase(geom)

            should_run_mcmc = self.mcmc_hk is not None
            should_run_analysis = self.analysis_hk is not None
            if not (should_run_mcmc or should_run_analysis):
                self._log_phase("END")
                return

            has_src_ctx = self.cfg.analysis.source_context is not None
            skip_sampling = should_run_analysis and has_src_ctx

            if self.mcmc_hk is not None:
                estims = self._run_mcmc_phase(self.mcmc_hk, geom, obs, skip_sampling)

                if self.analysis_hk is not None:
                    self._run_analysis_phase(self.analysis_hk, geom, obs, estims)

            self._log_phase("END")

        except Exception as e:
            self.log.critical(f"Simulation pipeline aborted due to an error: {e}")
            if self.ctx.comm:
                self.log.critical(
                    f"Abort triggered by rank {self.ctx.rank} to prevent deadlock."
                )
                self.ctx.comm.Abort(1)
            raise

    def _run_geometry_phase(self) -> G:
        self._log_phase("GEOMETRY")
        obs_path = self.paths.get_obs_path()
        with self._log_step("Compute geometry"):
            return self.geometry_hk.build_geometry(
                self.ctx,
                self.io_mng,
                self.cfg,
                obs_path,
            )

    def _run_observations_phase(self, geometry: G) -> O:
        self._log_phase("OBSERVATIONS")
        obs_path = self.paths.get_obs_path()
        if not obs_path.exists():
            with self._log_step(f"Generate observations to `{obs_path}`"):
                obs = self.obs_hk.generate_observations(
                    self.ctx, self.io_mng, self.cfg, geometry
                )
            with self._log_step(f"Save observations to `{obs_path}`"):
                obs_path.parent.mkdir(parents=True, exist_ok=True)
                self.obs_hk.save_observations(
                    self.ctx,
                    self.io_mng,
                    geometry,
                    obs,
                    obs_path,
                )
        else:
            self.log.warning(f"  │   Reuse existing observations from `{obs_path}`")
            with self._log_step(f"Load observations from `{obs_path}`"):
                obs = self.obs_hk.load_observations(
                    self.ctx, self.io_mng, geometry, obs_path
                )
        return obs

    def _run_mcmc_phase(
        self,
        mcmc_hk: McmcHook[G, O],
        geometry: G,
        obs: O,
        skip_sampling: bool = False,
    ) -> list[BaseEstimator]:
        self._log_phase("MCMC")
        with self._log_step("Build estimators"):
            vars_, estimators = mcmc_hk.build_estimators(geometry, obs)

        if skip_sampling:
            self.log.warning(
                f"  │   Source context provided. Skipping MCMC sampling. "
                f"Fetching checkpoints from `{self.cfg.analysis.source_context}`."
            )
            return estimators

        with self._log_step("Build model"):
            model = mcmc_hk.build_model(self.ctx, self.cfg, geometry, obs, vars_)

        with self._log_step("Build sampler"):
            s_params = self.create_sampler_params()
            s_params.ckpt_dir_path.mkdir(parents=True, exist_ok=True)
            sampler = Sampler.create_from_context(
                self.ctx, self.io_mng, model, estimators, s_params, self.log
            )

        with self._log_step("Run MCMC"):
            self.paths.get_obs_path().parent.mkdir(parents=True, exist_ok=True)
            sampler.sample()

        return estimators

    def _run_analysis_phase(
        self,
        analysis_hk: AnalysisHook[G, O],
        geometry: G,
        obs: O,
        estimators: list[BaseEstimator],
    ) -> None:
        self._log_phase("ANALYSIS")
        ckpt_dir = self.paths.get_ckpt_dir()
        burnin = self.cfg.analysis.burnin
        analysis_dir = self.paths.get_analysis_dir()
        with self._log_step("Run analysis"):
            results = analysis_hk.run_analysis(
                self.ctx,
                self.io_mng,
                self.cfg,
                geometry,
                obs,
                estimators,
                burnin,
                ckpt_dir,
                self.paths.get_obs_path(),
            )
        with self._log_step("Save analysis results"):
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_hk.save_results(
                self.ctx,
                self.io_mng,
                results,
                analysis_dir,
            )
        with self._log_step("Visualize analysis results"):
            analysis_hk.visualize_results(
                self.ctx, self.io_mng, results, self.paths.get_analysis_dir()
            )

    def create_sampler_params(self) -> SamplerParameters:
        s_params = self.cfg.sampler
        io_params = self.cfg.io
        sampler_params = SamplerParameters(
            s_params.ckpt_size,
            s_params.n_ckpts,
            self.paths.get_ckpt_dir(),
            io_params.ckpt_prefix,
            s_params.seed,
            s_params.start_ckpt_idx,
            io_params.start_ckpt_dir_path,
        )
        return sampler_params

    @contextmanager
    def _log_step(self, step_name: str) -> Generator[None, None, None]:
        self.log.info("  ├── %s...", step_name)
        start = time.perf_counter()
        try:
            yield
        except Exception:
            delta = time.perf_counter() - start
            self.log.exception("  │    └── FAILED after %.1fs", delta)
            raise
        else:
            green = "\033[32m"
            reset = "\033[0m"
            self.log.info(
                f"  │    └── {green}COMPLETED{reset} in %.1fs",
                time.perf_counter() - start,
            )

    def _log_phase(self, phase_name: str) -> None:
        """Helper to create distinct visual separators for major pipeline phases."""
        bold_cyan = "\033[1;36m"
        reset = "\033[0m"
        self.log.info(f"{bold_cyan}[%s]{reset}", phase_name.upper())
