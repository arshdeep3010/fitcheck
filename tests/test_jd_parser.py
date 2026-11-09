# fitcheck: no-index (sample requirements below are test data, not evidence)
from app.jd_parser import JdParser, heuristic_requirements

JD = """About us: we are a fast-growing startup.

Requirements:
- 2+ years building REST APIs with Spring Boot
- Experience with RAG and vector databases
- Kafka is a plus
- Team player
"""


class FakeChat:
    name = "fake-chat"

    def __init__(self, reply: str):
        self.reply = reply

    async def complete(self, system: str, user: str) -> str:
        return self.reply


def test_heuristic_picks_bullets_and_marks_optional_items():
    items = heuristic_requirements(JD, 10)
    texts = [t for t, _, _ in items]
    assert "2+ years building REST APIs with Spring Boot" in texts
    assert ("Kafka is a plus", False, True) in items


async def test_llm_output_is_used_and_ids_assigned():
    chat = FakeChat('<think>hmm</think>{"requirements": [{"text": "Spring Boot", "must_have": true},'
                    ' {"text": "spring boot", "must_have": true}, {"text": "Kafka", "must_have": false}]}')
    reqs, warnings = await JdParser(chat, max_requirements=10).parse(JD)
    assert [(r.id, r.text, r.must_have) for r in reqs] == [("r1", "Spring Boot", True), ("r2", "Kafka", False)]
    assert warnings == []


async def test_garbage_from_llm_falls_back_to_heuristics_with_warning():
    reqs, warnings = await JdParser(FakeChat("sorry, I can't"), max_requirements=10).parse(JD)
    assert reqs and reqs[0].text.startswith("2+ years")
    assert warnings


async def test_max_requirements_is_enforced():
    chat = FakeChat('{"requirements": [' + ",".join(f'{{"text": "skill {i}"}}' for i in range(20)) + "]}")
    reqs, _ = await JdParser(chat, max_requirements=5).parse(JD)
    assert len(reqs) == 5


def test_background_items_are_not_checkable():
    from app.jd_parser import is_background
    assert is_background("1-3 years of software engineering experience")
    assert is_background("5+ years of professional experience")
    assert is_background("B.Tech in Computer Science or equivalent degree")
    assert not is_background("2+ years building REST APIs with Spring Boot")
    assert not is_background("Experience with Docker and CI/CD")


async def test_code_overrides_model_when_it_marks_years_checkable():
    chat = FakeChat('{"requirements": [{"text": "3+ years of software engineering experience", "checkable": true},'
                    ' {"text": "Spring Boot", "checkable": true}]}')
    reqs, _ = await JdParser(chat, max_requirements=10).parse(JD)
    assert [r.checkable for r in reqs] == [False, True]


async def test_background_lines_the_model_dropped_are_re_added():
    jd = "- Spring Boot\n- 3+ years of professional software experience\n- B.Tech in Computer Science\n"
    chat = FakeChat('{"requirements": [{"text": "Spring Boot"}, {"text": "B.Tech in Computer Science", "checkable": false}]}')
    reqs, _ = await JdParser(chat, max_requirements=10).parse(jd)
    texts = [(r.text, r.checkable) for r in reqs]
    assert ("3+ years of professional software experience", False) in texts
    assert sum(1 for t, _ in texts if "B.Tech" in t) == 1  # not duplicated
