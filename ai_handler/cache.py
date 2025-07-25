from abc import ABC, abstractmethod
import typing as t
from ai_handler.question import Question


class Cache(ABC):
    """
    Abstract base class for a cache used by AiHandler.
    """

    @staticmethod
    def question_key(question: Question) -> int:
        """
        Generate a unique key for the question.
        This is used to store and retrieve cached answers.
        """
        return hash(question)

    @abstractmethod
    def set(self, question: Question, raw_answer: str) -> None:
        """
        Store the raw answer for a given question.
        """
        ...

    @abstractmethod
    def get(self, question: Question) -> t.Optional[str]:
        """
        Retrieve the cached raw answer for a given question.
        Returns None if not present.
        """
        ...

class InMemoryCache(Cache):
    """
    Simple in-memory cache using Python's dict.
    Not persistent across runs.
    """

    def __init__(self):
        self._store: dict[int, str] = {}

    def set(self, question: Question, raw_answer: str) -> None:
        self._store[self.question_key(question)] = raw_answer

    def get(self, question: Question) -> t.Optional[str]:
        return self._store.get(self.question_key(question))

class NullCache(Cache):
    """
    No-op cache. Always misses. Useful as a default/null object.
    """

    def set(self, question: Question, raw_answer: str) -> None:
        pass

    def get(self, question: Question) -> None:
        return None
