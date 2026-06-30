import pytest
import torch
from mpi4py import MPI


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

        if target_device == "cpu" and "cpu" not in current_tags:
            item.add_marker(skip_wrong_device)
        elif target_device == "gpu" and "gpu" not in current_tags:
            item.add_marker(skip_wrong_device)

        if target_mode == "mpi" and "mpi" not in current_tags:
            item.add_marker(skip_wrong_mode)
        elif target_mode == "serial" and "serial" not in current_tags:
            item.add_marker(skip_wrong_mode)


@pytest.fixture(scope="session")
def mode(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--mode")


@pytest.fixture(scope="session")
def comm(mode: str) -> MPI.Comm | None:
    if mode == "mpi":
        return MPI.COMM_WORLD
    else:
        return None


@pytest.fixture(scope="session")
def rank(comm: MPI.Comm | None) -> int:
    if comm is not None:
        return comm.Get_rank()
    else:
        return 0


@pytest.fixture(scope="session")
def comm_size(comm: MPI.Comm | None) -> int:
    if comm is not None:
        return comm.Get_size()
    else:
        return 1


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--device")


@pytest.fixture(scope="session", autouse=True)
def backend_setup(device: str, rank: int) -> None:
    if device == "gpu":
        import cards.backend as xp

        xp.set_backend("cupy")

        nb_gpu = xp.cuda.runtime.getDeviceCount()
        gpu_id = rank % nb_gpu

        xp.cuda.runtime.setDevice(gpu_id)
        torch.set_default_device(f"cuda:{gpu_id}")

        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


@pytest.fixture(scope="session", autouse=True)
def torch_device(device: str, rank: int) -> torch.device:
    if device == "gpu":
        nb_gpu = torch.cuda.device_count()
        gpu_id = rank % nb_gpu

        return torch.device(f"cuda:{gpu_id}")
    else:
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
