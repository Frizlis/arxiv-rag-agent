from typing import Literal

from pydantic import (
    AnyHttpUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class Article(BaseModel):
    """Represent an article from the MultiHopRAG corpus."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    author: str | None
    source: str = Field(min_length=1)
    published_at: AwareDatetime
    category: str = Field(min_length=1)
    url: str = Field(min_length=1)
    body: str = Field(min_length=1)

    @field_validator("title", "source", "category", "body")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        AnyHttpUrl(value)
        return value


class Evidence(BaseModel):
    """Represent reference evidence used for evaluation."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    author: str | None
    source: str = Field(min_length=1)
    published_at: AwareDatetime
    category: str = Field(min_length=1)
    url: str = Field(min_length=1)
    fact: str = Field(min_length=1)

    @field_validator("title", "source", "category", "fact")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        AnyHttpUrl(value)
        return value


class EvaluationQuestion(BaseModel):
    """Represent a question and its reference answer and evidence."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    question_type: Literal[
        "inference_query", "comparison_query", "temporal_query", "null_query"
    ]
    evidence_list: list[Evidence]

    @field_validator("query", "answer")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value
