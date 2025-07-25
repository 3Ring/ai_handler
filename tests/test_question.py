from ai_handler.question import SimpleQuestion


def test_simple_question_basic_properties():
    q = SimpleQuestion("test prompt?")
    assert q.question == "test prompt?"
    assert q.prompt  # prompt should not be empty
    assert isinstance(hash(q), int)


def test_simple_question_on_retry_increases_prompt(monkeypatch):
    q = SimpleQuestion("test?")
    retry_fn = q.on_retry

    # Simulate an exception
    class DummyExc(Exception):
        pass

    from ai_handler.errors import InvalidModelResponseException

    exc = InvalidModelResponseException(DummyExc())
    new_q = retry_fn(exc, 1)
    assert isinstance(new_q, SimpleQuestion)
    assert "Review this conversation" in new_q.prompt


def test_simple_question_max_retries():
    q = SimpleQuestion("will it retry?")
    assert q.max_retries == 3
