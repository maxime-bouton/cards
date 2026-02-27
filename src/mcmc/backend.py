import importlib


class BackendManager:
    def __init__(self, backend):
        self.set_backend(backend)

    @property
    def xp(self):
        return self._xp

    def set_backend(self, new_backend):
        if new_backend in ["cupy", "numpy"]:
            self._xp = importlib.import_module(new_backend)
        else:
            raise ValueError(
                f"Unsupported backend: {new_backend}. Choose 'cupy' or 'numpy'."
            )


bm = BackendManager("numpy")


class Proxy:
    def __getattr__(self, name):
        return getattr(bm.xp, name)


xp = Proxy()
