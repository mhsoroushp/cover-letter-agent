from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END
from app.prompt import quality_check_prompt, analysis_prompt, draft_prompt
from app.services import model

from app.services.blob_storage import BlobStorageService
from app.state import CoverLetterState, MessagesState, QualityCheckResult


class GraphNodes:
    """Collection of graph nodes and arithmetic tool functions."""
    MAX_HISTORY_MESSAGES = 15

    def __init__(self, blob_service: BlobStorageService):
        self.blob_service = blob_service

    @classmethod
    def _last_messages(cls, state: MessagesState):
        return state.get("messages", [])[-cls.MAX_HISTORY_MESSAGES :]


    def load_documents(self, state: CoverLetterState) -> str:

        cv_text = state.get("cv_text", "").strip()
        job_text = state.get("job_text", "").strip()

        if not cv_text and not job_text:
            print("already loaded")
            cv_text = self.blob_service.read_cv()
            job_text = self.blob_service.read_job_description()

        return {
            "cv_text": cv_text,
            "job_text": job_text
        }
    

    def analysis_attachments(self, state: CoverLetterState):
        cv_text = state.get("cv_text", "").strip()
        job_text = state.get("job_text", "").strip()
        if not cv_text:
            raise ValueError("cv_text is required before analyze_attachments runs.")
        if not job_text:
            raise ValueError(
                "job_text is required before analysis_attachments runs."
            )

        prompt = analysis_prompt(cv_text, job_text)

        response = model.invoke(
            [SystemMessage(content=prompt)] + self._last_messages(state)
        )
        return {"analysis": response.content}
    

    def draft_letter(self, state: CoverLetterState):
        analysis = state.get("analysis", "").strip()
        if not analysis:
            raise ValueError("analysis is required before draft_letter runs.")

        prompt = draft_prompt(analysis)
        response = model.invoke(
            [SystemMessage(content=prompt)]
        )
        return {"draft_letter": response.content}
    

    def check_quality(self, state: CoverLetterState):
        draft_letter = state.get("draft_letter", "").strip()
        if not draft_letter:
            raise ValueError("draft_letter is required before check_quality runs.")

        prompt = quality_check_prompt(draft_letter)

        quality_model = model.with_structured_output(QualityCheckResult)
        result = quality_model.invoke([SystemMessage(content=prompt)])

        return {
            "quality_check": result.model_dump(),
            "draft_letter": draft_letter
        }
