from ai_handler.answer import SimpleAnswer

def test_simple_answer_parse_and_raw():
    raw = " Hello! "
    answer = SimpleAnswer.parse(raw)
    assert isinstance(answer, SimpleAnswer)
    assert answer.raw == "Hello!"

def test_simple_answer_direct_init():
    answer = SimpleAnswer("hi")
    assert answer.raw == "hi"
