r"""Helper function to select computing backend (``numpy`` or ``cupy``)."""

import importlib

xp = importlib.import_module("numpy")


class gpu_context:
    def __init__(self, id: int):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def set_backend(backend_name="numpy"):
    global xp
    xp = importlib.import_module(backend_name)


def enable_multi_gpu():
    global gpu_context
    from cupy.cuda import Device as gpu_context
