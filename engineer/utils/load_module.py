import importlib
from typing import Callable, Any


def load_module(object: str) -> Callable[..., Any]:
    module, object = object.rsplit(".", 1)
    module = importlib.import_module(module)
    fn = getattr(module, object)
    return fn