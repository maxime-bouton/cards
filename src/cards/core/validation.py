from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cards.core.execution_context import ContextTag

SAFE_NAME = r"^[\w\-]+$"
_PATH_SUFFIX = "_path"


def _resolve_path_fields(data: Any, anchor_dir: Path) -> Any:
    """Recursively anchor any '*_path' string/Path value to `base_dir`.
    Absolute paths and empty strings are left untouched."""
    if not isinstance(data, dict):
        return data
    resolved = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_path_fields(value, anchor_dir)
        elif (
            key.endswith(_PATH_SUFFIX)
            and isinstance(value, (str, Path))
            and str(value) != ""
        ):
            p = Path(value)
            resolved[key] = p if p.is_absolute() else (anchor_dir / p).resolve()
        else:
            resolved[key] = value
    return resolved


class PydanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _anchor_relative_paths(cls, data: Any, info: ValidationInfo) -> Any:
        anchor_dir = (info.context or {}).get("anchor_dir")
        if anchor_dir is None or not isinstance(data, dict):
            return data
        return _resolve_path_fields(data, Path(anchor_dir))


class ApplicationConfig(PydanticModel):
    type: str = Field(pattern=SAFE_NAME)
    name: str = Field(pattern=SAFE_NAME)


class IOConfig(PydanticModel):
    root_dir_path: Path = Field(default_factory=lambda: Path.cwd() / "results")
    log_file_prefix: str = Field(default="sampling", pattern=SAFE_NAME)
    obs_file_stem: str = Field(default="data", pattern=SAFE_NAME)
    ckpt_prefix: str = Field(default="checkpoint_", pattern=SAFE_NAME)
    log_file_path: Path | None = None
    obs_dir_path: Path | None = None
    ckpt_dir_path: Path | None = None
    start_ckpt_dir_path: Path | None = None

    @field_validator("root_dir_path", mode="before")
    @classmethod
    def handle_empty_root(cls, v: str | None) -> Path | str:
        if v == "" or v is None:
            return Path.cwd() / "results"
        return v

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
