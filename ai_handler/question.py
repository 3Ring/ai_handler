from __future__ import annotations
from abc import ABC
import typing as t
import traceback
import ai_handler.errors as ex


class Question(ABC):

    @property
    def question(self) -> str:
        try:
            return self._question
        except AttributeError:
            return ""

    @question.setter
    def question(self, value: str) -> None:
        self._question = value

    @property
    def context(self) -> str:
        try:
            return self._context
        except AttributeError:
            return ""

    @context.setter
    def context(self, value: str) -> None:
        self._context = value

    @property
    def response_format(self) -> str:
        try:
            return self._response_format
        except AttributeError:
            return ""

    @response_format.setter
    def response_format(self, value: str) -> None:
        self._response_format = value

    @property
    def prompt(self) -> str:
        """
        Override this method to provide a custom prompt for the question.
        Default implementation returns the question text.
        """
        if hasattr(self, "_prompt"):
            return self._prompt
        context = f"Context:\n{self.context}" if self.context else ""
        response_format = (
            f"Response Format:\n{self.response_format}" if self.response_format else ""
        )
        return "\n\n\n\n".join([context, f"Question:\n{self.question}", response_format])

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    @prompt.deleter
    def prompt(self):
        if hasattr(self, "_prompt"):
            del self._prompt

    @property
    def on_retry(
        self,
    ) -> t.Optional[t.Callable[[Exception, int], t.Optional[Question]]]:
        """
        Override this method to provide custom behavior when a question is retried.
        Default implementation returns None.
        """
        return None

    @property
    def max_retries(self) -> int:
        """
        Override this property to specify the maximum number of retries for this question.
        Default implementation returns 0, meaning no retries.
        """
        return 0

    @property
    def factory_retry_exceptions(self) -> t.Optional[t.Tuple[type[Exception], ...]]:
        """
        Override this property to specify which exceptions should trigger a retry in the factory.
        Default implementation returns None which will cause no retries to be attempted.
        """
        return None

    def __hash__(self) -> int:
        return hash(self.prompt)


class SimpleQuestion(Question):
    """
    Simple implementation for easy defaults
    """

    def __init__(self, question: str, context: str = "", response_format: str = ""):
        self.question = question
        self.context = context
        self.response_format = response_format

    @property
    def on_retry(self) -> t.Callable[[Exception, int], t.Optional[SimpleQuestion]]:
        return simple_retry(self)

    @property
    def max_retries(self) -> int:
        return 3

    @property
    def factory_retry_exceptions(self) -> t.Optional[t.Tuple[type[Exception], ...]]:
        return (KeyError, ValueError, AttributeError, AssertionError, TypeError)


def simple_retry(
    question: Question,
) -> t.Callable[[Exception, int], t.Optional[Question]]:
    """
    Returns a retry function that modifies the question prompt based on the exception.
    This function is used to create a retry mechanism for the question.
    It captures the exception and the number of retries, and modifies the question prompt accordingly."""
    def simple_retry(e: Exception, retries: int) -> t.Optional[Question]:
        if not isinstance(e, ex.InvalidModelResponseException):
            return None
        del question.prompt  # clear any previously saved retry prompt
        original_prompt = question.prompt
        prompt_parts = [
            f"A {type(e.origin).__name__} was raised during transformation of your response:",
            f"\n{traceback.format_exception(e.origin)}",
        ]
        if retries:
            prompt_parts.append(
                f"this transformation has previously failed {retries} times."
            )
        prompt_parts.append(f"Review this conversation and try again.")
        if original_prompt:
            prompt_parts.append(f"Original question: {original_prompt}")
        question.prompt = "\n".join(prompt_parts)
        return question

    return simple_retry
