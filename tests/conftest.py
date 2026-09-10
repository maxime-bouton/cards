import pytest
import torch

from cards.core.execution_context import ExecutionContext


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="cpu or gpu",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="serial",
        choices=["serial", "mpi"],
        help="serial or mpi",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mpi: Restrict test to MPI environment")
    config.addinivalue_line("markers", "serial: Restrict test to Serial environment")
    config.addinivalue_line("markers", "gpu: Restrict test to GPU backend")
    config.addinivalue_line("markers", "cpu: Restrict test to CPU backend")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    target_device = config.getoption("--device")
    target_mode = config.getoption("--mode")

    skip_wrong_device = pytest.mark.skip(
        reason=f"Test not supported on {target_device} backend"
    )
    skip_wrong_mode = pytest.mark.skip(
        reason=f"Test not supported in current {target_mode} mode"
    )

    for item in items:
        explicit_markers = {m.name for m in item.iter_markers()}

        has_device_restriction = "cpu" in explicit_markers or "gpu" in explicit_markers
        if not has_device_restriction:
            item.add_marker(pytest.mark.cpu)
            item.add_marker(pytest.mark.gpu)

        has_mode_restriction = "serial" in explicit_markers or "mpi" in explicit_markers
        if not has_mode_restriction:
            item.add_marker(pytest.mark.serial)
            item.add_marker(pytest.mark.mpi)

        current_tags = {m.name for m in item.iter_markers()}

        if (
            target_device == "cpu"
            and "cpu" not in current_tags
            or target_device == "gpu"
            and "gpu" not in current_tags
        ):
            item.add_marker(skip_wrong_device)

        if (
            target_mode == "mpi"
            and "mpi" not in current_tags
            or target_mode == "serial"
            and "serial" not in current_tags
        ):
            item.add_marker(skip_wrong_mode)


@pytest.fixture(scope="session", autouse=True)
def ctx(request: pytest.FixtureRequest) -> ExecutionContext:
    mode = request.config.getoption("--mode")
    device = request.config.getoption("--device")
    return ExecutionContext(mode=mode, device=device)


@pytest.fixture(scope="session")
def comm(ctx: ExecutionContext):
    return ctx.comm


@pytest.fixture(scope="session")
def rank(ctx: ExecutionContext) -> int:
    return ctx.rank


@pytest.fixture(scope="session")
def comm_size(ctx: ExecutionContext) -> int:
    return ctx.comm_size


@pytest.fixture(scope="session")
def mode(ctx: ExecutionContext) -> str:
    return ctx.mode


@pytest.fixture(scope="session")
def device(ctx: ExecutionContext) -> str:
    return ctx.device


@pytest.fixture(scope="session")
def torch_device(ctx: ExecutionContext) -> torch.device:
    if ctx.is_gpu:
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def seed() -> int:
    return 1234


@pytest.fixture(scope="session")
def seed2():
    return 42


@pytest.fixture(scope="session", params=[(3, 64, 64), (1, 31, 31)])
def input_shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    return request.param
