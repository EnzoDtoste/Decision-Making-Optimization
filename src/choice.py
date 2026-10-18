from abc import ABC, abstractmethod

class Choice(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def count_params(self) -> int:
        ...

    @abstractmethod
    def __call__(self, n, params) -> int:
        ...

    @abstractmethod
    def apply_constraints(self, params) -> list[float]:
        ...