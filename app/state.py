import operator

from langchain.messages import AnyMessage
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from typing import Literal


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class QualityCheckResult(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall quality score")
    verdict: Literal["pass", "revise", "fail"]
    strengths: list[str]
    issues: list[str]
    rewrite_suggestions: list[str]

class CoverLetterState(MessagesState):
    cv_text: str
    job_text: str
    analysis: str
    draft_letter: str
    quality_check: QualityCheckResult


