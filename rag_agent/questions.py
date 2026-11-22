from pathlib import Path

from pydantic import TypeAdapter

from rag_agent.schemas import Article, EvaluationQuestion

QUESTION_LIST_ADAPTER = TypeAdapter(list[EvaluationQuestion])


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """Load and validate evaluation questions."""
    content = path.read_text(encoding="utf-8-sig")
    questions = QUESTION_LIST_ADAPTER.validate_json(content)

    if not questions:
        raise ValueError("Question dataset must not be empty")

    for index, question in enumerate(questions):
        if question.question_type == "null_query":
            if question.evidence_list:
                raise ValueError(
                    f"Question at index {index}: "
                    "null_query must have no evidence"
                )

            if question.answer != "Insufficient information.":
                raise ValueError(
                    f"Question at index {index}: "
                    "unexpected reference answer for null_query"
                )

        elif not question.evidence_list:
            raise ValueError(
                f"Question at index {index}: "
                "answerable question must have evidence"
            )

    return questions


def validate_evidence_links(
    questions: list[EvaluationQuestion], articles: list[Article]
) -> None:
    """Check that every evidence URL exists in the corpus."""
    article_urls = {article.url for article in articles}

    for question_index, question in enumerate(questions):
        for evidence_index, evidence in enumerate(question.evidence_list):
            if evidence.url not in article_urls:
                raise ValueError(
                    f"Question at index {question_index}, "
                    f"evidence at index {evidence_index}: "
                    f"URL not found in corpus: {evidence.url}"
                )
