from pathlib import Path

from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field, field_validator, model_validator

from cards.core.execution_context import ContextTag

SAFE_NAME = r"^[\w\-]+$"


class ApplicationConfig(PydanticModel):
    type: str = Field(pattern=SAFE_NAME)
    name: str = Field(pattern=SAFE_NAME)


class IOConfig(PydanticModel):
    root_dir: Path = Field(default_factory=lambda: Path.cwd() / "results")
    log_file_prefix: str = Field(default="sampling", pattern=SAFE_NAME)
    obs_file_stem: str = Field(default="data", pattern=SAFE_NAME)
    ckpt_prefix: str = Field(default="checkpoint_", pattern=SAFE_NAME)
    log_file_path: Path | None = None
    obs_dir_path: Path | None = None
    ckpt_dir_path: Path | None = None
    start_ckpt_dir_path: Path | None = None

    @field_validator(
        "log_file_path",
        "obs_dir_path",
        "ckpt_dir_path",
        "start_ckpt_dir_path",
        mode="before",
    )
    @classmethod
    def convert_empty_string_to_none(cls, s: str | None) -> str | None:
        return None if s == "" else s


class SamplerConfig(PydanticModel):
    ckpt_size: int = Field(ge=1)
    n_ckpts: int = Field(ge=1)
    seed: int = Field(default=42)
    start_ckpt_idx: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_start_index(self) -> "SamplerConfig":
        if self.start_ckpt_idx >= self.n_ckpts:
            raise ValueError(
                f"`start_ckpt_idx` ({self.start_ckpt_idx}) cannot be greater than or "
                f"equal to `n_ckpts` ({self.n_ckpts})."
            )
        return self


class BaseObservationsConfig(PydanticModel):
    model_config = ConfigDict(extra="allow")


class BaseParametersConfig(PydanticModel):
    model_config = ConfigDict(extra="allow")


class AnalysisConfig(PydanticModel):
    burnin: int = Field(default=0, ge=0)
    save_all: bool = False
    source_context: ContextTag | None = None


class SimulationConfig[
    ObsCfg: BaseObservationsConfig,
    ParamCfg: BaseParametersConfig,
](PydanticModel):
    application: ApplicationConfig
    sampler: SamplerConfig
    observations: ObsCfg
    parameters: ParamCfg
    io: IOConfig = Field(default_factory=IOConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="after")
    def validate_burnin(self) -> "SimulationConfig":
        if self.analysis.burnin >= self.sampler.n_ckpts:
            raise ValueError(
                f"`analysis.burnin` ({self.analysis.burnin}) is too high. With "
                f"`n_ckpts={self.sampler.n_ckpts}`, there must remain at "
                f"least one checkpoint to run the analysis."
            )
        return self


DefaultSimulationConfig = SimulationConfig[BaseObservationsConfig, BaseParametersConfig]
