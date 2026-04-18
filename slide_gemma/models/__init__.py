"""Model loading and inference.

Submodules are imported lazily to avoid pulling in torch at import time.
"""


def __getattr__(name):
    if name in ("load_model", "load_model_for_training", "generate"):
        from . import loader
        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
