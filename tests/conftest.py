import pytest


def pytest_addoption(parser):
    parser.addoption(
        "-C",
        action="store",
        metavar="NAME",
        default="serial-cpu",
        help="only run tests matching with the context NAME.",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "env(name): mark test to run only on named context"
    )


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("-C")


def pytest_runtest_setup(item):
    envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
    if envnames:
        if item.config.getoption("-C") not in envnames:
            pytest.skip(f"test requires context in {envnames!r}")
