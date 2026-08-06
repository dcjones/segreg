def hello() -> str:
    return "Hello from segreg!"

from .regression import RegressionModel

__all__ = [
    "RegressionModel"
]
