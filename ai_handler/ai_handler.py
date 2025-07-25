import typing as t
import ai_handler.errors as ex
from ai_handler.providers.ai_provider_client import AiProviderClient
from ai_handler.question import Question, SimpleQuestion
from ai_handler.answer import Answer, SimpleAnswer
from ai_handler.cache import Cache, InMemoryCache
from ai_handler.errors import InvalidModelResponseException

import logging

logger = logging.getLogger("ai_handler")

T = t.TypeVar("T", bound=Answer)


class AiHandler(t.Generic[T]):
    def __init__(
        self,
        client: AiProviderClient,
        cache: t.Optional[Cache] = None,
    ):
        self.client = client
        self.cache = cache or InMemoryCache()

    def ask(
        self,
        question: Question | str,
        *,
        answer_factory: t.Optional[t.Callable[[str], T]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> T:
        if isinstance(question, str):
            question = SimpleQuestion(question)
        if answer_factory is None:
            answer_factory = SimpleAnswer
        raw = None
        if self.cache and use_cache:
            raw = self.cache.get(question)
        if raw is not None:
            return answer_factory(raw)
        answer = self._ask(question, answer_factory, **kwargs)
        if self.cache and use_cache:
            self.cache.set(question, answer.raw)
        return answer

    def _ask(
        self, question: Question, answer_factory: t.Callable[[str], T], **kwargs
    ) -> T:
        retries = 0
        while True:
            try:
                client_response = self.client.ask(question.prompt, **kwargs)
                return transform(
                    lambda: answer_factory(client_response),
                    to_catch=question.factory_retry_exceptions,
                )
            except InvalidModelResponseException as e:
                logger.warning(f"Attempt {retries}: {e}")
                logger.debug(
                    f"Failed to parse response from AI model for question: {question.prompt}. "
                    f"Response: {client_response if client_response else 'No response'}"
                )
                retry_transformer = question.on_retry
                if not retry_transformer or retries >= question.max_retries:
                    raise e
                if question := retry_transformer(e, retries):
                    retries += 1
                    continue
                raise e


def transform(
    factory: t.Callable[[], T],
    to_catch: t.Optional[tuple[type[Exception], ...]] = None,
) -> T:
    if not to_catch:
        return factory()
    try:
        return factory()
    except to_catch as e:
        raise ex.InvalidModelResponseException(origin=e) from e
