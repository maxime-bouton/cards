from pathlib import Path

import cards.backend as xp
from cards.core.execution_context import ExecutionContext
from cards.estimators.base_estimator import BaseEstimator


def list_checkpoints(ckpt_dir: Path, prefix: str, n_ckpts: int) -> list[Path]:
    all_matches = ckpt_dir.glob(f"{prefix}*.h5")
    ckpt_files = sorted(
        (p for p in all_matches if p.stem.removeprefix(prefix).isdigit()),
        key=lambda p: int(p.stem.removeprefix(prefix)),
    )

    if len(ckpt_files) < n_ckpts:
        raise ValueError(
            f"Not enough checkpoints found. Expected {n_ckpts}, got {len(ckpt_files)}."
        )

    return ckpt_files


def summarize_time(computation_time: xp.ndarray, ndigits: int = 6) -> dict:
    return {
        "mean": computation_time.mean(axis=1).round(ndigits).tolist(),
        "std": computation_time.std(axis=1).round(ndigits).tolist(),
        "min": computation_time.min(axis=1).round(ndigits).tolist(),
        "max": computation_time.max(axis=1).round(ndigits).tolist(),
    }


def reduce_all(
    estimators: list[BaseEstimator],
    per_ckpt_local: list[dict[str, xp.ndarray]],
    burnin: int,
    ctx: ExecutionContext,
):
    reduced_local, full_shapes, slices, uncertainty_keys = {}, {}, {}, set()
    for estimator in estimators:
        left = [{k: d[k] for k in estimator.declared_keys} for d in per_ckpt_local]
        reduced_local.update(estimator.reduce_checkpoints(left, burnin, ctx))
        full_shapes.update(estimator.global_shapes)
        slices.update(estimator.slices)
        uncertainty_keys.update(estimator.uncertainty_keys)
    return reduced_local, full_shapes, slices, uncertainty_keys


def add_error_maps(
    reduced_local: dict[str, xp.ndarray],
    full_shapes: dict[str, tuple[int, ...]],
    slices: dict[str, tuple[slice, ...]],
    ground_truth: xp.ndarray,
):
    for key in list(reduced_local.keys()):
        if key.endswith("_mmse"):
            prefix = key.split("_")[0]
            if prefix in ground_truth and ground_truth[prefix] is not None:
                err_key = f"{prefix}_err"
                reduced_local[err_key] = xp.abs(
                    reduced_local[key] - ground_truth[prefix]
                )
                full_shapes[err_key] = full_shapes[key]
                slices[err_key] = slices[key]


def compute_metrics(
    targets: dict[str, xp.ndarray],
    references: dict[str, xp.ndarray],
    metric_fns: dict,
    ctx: ExecutionContext,
    ndigits: int = 2,
) -> dict:
    metrics = {}
    for k, v in targets.items():
        if v is not None and k in references and references[k] is not None:
            metrics[k] = {
                m_name: round(m_fn(references[k], v, ctx), ndigits)
                for m_name, m_fn in metric_fns.items()
            }
    return metrics


def compose_slices(
    layout_slices: tuple[slice, ...],
    crop_slices: tuple[slice, ...],
) -> tuple[slice, ...]:
    list_s = []
    for ls, cs in zip(layout_slices, crop_slices):
        left = (ls.start or 0) + (cs.start or 0)
        r = (ls.stop or 0) + (cs.start or 0)
        list_s.append(slice(left or None, r or None))
    return tuple(list_s)
