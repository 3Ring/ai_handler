from __future__ import annotations

from abc import ABC


class Answer(ABC):
    def __init__(self, raw: str):
        self.raw = raw

    @property
    def raw(self) -> str:
        try:
            return self._raw
        except AttributeError as e:
            raise AttributeError(
                "raw property has not been set. "
                "This is likely because the __init__ was overridden without calling super().__init__(raw)."
            ) from e

    @raw.setter
    def raw(self, value: str):
        self._raw = value



class SimpleAnswer(Answer):
    """
    Simple answer that just wraps the raw string.
    """

    def __init__(self, raw: str):
        super().__init__(raw.strip())
