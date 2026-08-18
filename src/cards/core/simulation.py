import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Generic, Literal

from cards.core.execution_context import ExecutionContext
from cards.core.geometry_hook import G, GeometryHook
from cards.core.mcmc_hook import McmcHook
from cards.core.observations_hook import Obs, ObservationsHook
from cards.core.utils import parse_args
from cards.io.io_manager import IOManager
from cards.io.path_builder import PathBuilder
from cards.io.utils import read_json
from cards.logger import build_logger
from cards.models import BaseDistributedModel, BaseModel
from cards.samplers import SamplerParameters
from cards.samplers.sampler import Sampler


class Simulation(Generic[G, Obs]):
    def __init__(
        self,
        geometry_hk: GeometryHook[G],
        obs_hk: ObservationsHook[G, Obs],
        mcmc_hk: McmcHook[G, Obs],
        # analysis_hk: AnalysisHook,
        mode: Literal["serial", "mpi"],
        device: Literal["cpu", "gpu"],
        cfg: dict | Path | str,
        paths: PathBuilder | None = None,
    ) -> None:
        self.geometry_hk = geometry_hk
        self.obs_hk = obs_hk
        self.mcmc_hk = mcmc_hk
        # self.analysis_hk = analysis_hk

        self.ctx = ExecutionContext(mode, device)
        self.cfg = cfg if isinstance(cfg, dict) else read_json(cfg)

        self.paths = paths or PathBuilder(self.cfg, self.ctx)
        self.io_mng = IOManager(self.ctx)
        self.log = build_logger(self.ctx.rank, self.paths.get_log_path())

    @classmethod
    def from_cli(
        cls,
        geom_hk: GeometryHook[G],
        obs_hk: ObservationsHook[G, Obs],
        mcmc_hk: McmcHook[G, Obs],
        # analysis_hk: AnalysisHook,
        paths: PathBuilder | None = None,
    ) -> "Simulation":
        args = parse_args()
        return cls(
            geom_hk,
            obs_hk,
            mcmc_hk,
            # analysis_hk,
            args.mode,
            args.device,
            args.config,
            paths,
        )

    def run(self) -> None:
        """Run all four phases in order. Any exception aborts the whole run —
        matches the original behaviour (SystemExit(1) on failure).
        """
        try:
            geometry = self._run_geometry_phase()
            obs = self._run_observations_phase(geometry)
            model = self._run_mcmc_phase(geometry, obs)
            # self._run_analysis_phase(model, sampler)
            self._log_phase("END")
        except Exception:
            self.log.critical("Simulation pipeline aborted due to an error.")
            raise SystemExit(1)

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

    def _run_observations_phase(self, geometry: G) -> Obs:
        self._log_phase("OBSERVATIONS")
        obs_path = self.paths.get_obs_path()
        if not obs_path.exists():
            with self._log_step(f"Generate observations to `{obs_path}`"):
                obs = self.obs_hk.generate_observations(
                    self.ctx, self.io_mng, self.cfg, geometry
                )
            with self._log_step(f"Save observations to `{obs_path}`"):
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
        self, geometry: G, obs: Obs
    ) -> BaseModel | BaseDistributedModel:
        self._log_phase("MCMC")
        with self._log_step("Build model"):
            model = self.mcmc_hk.build_model(self.ctx, self.cfg, geometry, obs)
        with self._log_step("Build sampler"):
            s_params = self.create_sampler_params()
            sampler = Sampler.create_from_context(
                self.ctx, self.io_mng, model, s_params, self.log
            )
        with self._log_step("Run MCMC"):
            sampler.sample()
        return model

    # def _run_analysis_phase(self, model: M, sampler: S) -> None:
    #     self._log_phase("ANALYSIS")
    #     with self._log_step("Setup analysis"):
    #         self.analysis_hk.setup_analysis(self.cfg, self.ctx, model)
    #     with self._log_step("Run analysis"):
    #         self.analysis_hk.run_analysis(model, sampler)

    def create_sampler_params(self) -> SamplerParameters:
        s_params = self.cfg["sampler"]
        io_params = self.cfg["io"]
        sampler_params = SamplerParameters(
            s_params["ckpt_size"],
            s_params["n_ckpts"],
            self.paths.get_ckpt_dir(),
            io_params["ckpt_prefix"],
            s_params["seed"],
            s_params["start_ckpt_idx"],
            io_params.get("start_ckpt_dir_path", None),
        )
        return sampler_params

    @contextmanager
    def _log_step(self, step_name: str) -> Generator:
        self.log.info("  ├── %s...", step_name)
        start = time.perf_counter()
        try:
            yield
        except Exception as e:
            delta = time.perf_counter() - start
            self.log.error(
                "  │    └── FAILED after %.1fs (%s)", delta, e, exc_info=True
            )
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
