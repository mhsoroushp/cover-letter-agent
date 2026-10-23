from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END

from app.config import SYSTEM_PROMPT
from app.services import model
from app.services.blob_storage import (
    DEFAULT_CV_BLOB_NAME,
    DEFAULT_CV_CONTAINER,
    read_blob_pdf_text,
    read_blob_text,
)

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


    @tool
    def read_cv_from_blob(blob_name: str = "", container_name: str = "") -> str:
        """Read CV content from Azure Blob Storage as UTF-8 text.

        Args:
            blob_name: Blob name/path (or full blob URL) for the CV file.
                If omitted, AZURE_CV_BLOB_NAME is used.
            container_name: Blob container name. If omitted, AZURE_CV_CONTAINER is used.
        """
        selected_container = container_name or DEFAULT_CV_CONTAINER
        selected_blob_name = blob_name or DEFAULT_CV_BLOB_NAME
        if not selected_container:
            raise ValueError(
                "container_name is required when AZURE_CV_CONTAINER is not set."
            )
        if not selected_blob_name:
            raise ValueError(
                "blob_name is required when AZURE_CV_BLOB_NAME is not set."
            )
        return read_blob_text(selected_container, selected_blob_name)

    @tool
    def read_cv_pdf_from_blob(blob_name: str = "", container_name: str = "") -> str:
        """Read CV content from a PDF blob in Azure Blob Storage.

        Args:
            blob_name: Blob name/path (or full blob URL) for the PDF CV file.
                If omitted, AZURE_CV_BLOB_NAME is used.
            container_name: Blob container name. If omitted, AZURE_CV_CONTAINER is used.
        """
        selected_container = container_name or DEFAULT_CV_CONTAINER
        selected_blob_name = blob_name or DEFAULT_CV_BLOB_NAME
        if not selected_container:
            raise ValueError(
                "container_name is required when AZURE_CV_CONTAINER is not set."
            )
        if not selected_blob_name:
            raise ValueError(
                "blob_name is required when AZURE_CV_BLOB_NAME is not set."
            )
        return read_blob_pdf_text(selected_container, selected_blob_name)

    def load_documents(self, state: CoverLetterState) -> str:
        """Load documents from Azure Blob Storage."""
        cv_text = self.blob_service.read_cv()
        job_desc = self.blob_service.read_job_description()

        return {
            "cv_text": cv_text,
            "job_desciption": job_desc
        }

    def analysis_attachments(self, state: CoverLetterState):
        cv_text = state.get("cv_text", "").strip()
        job_description = state.get("job_desciption", "").strip()
        if not cv_text:
            raise ValueError("cv_text is required before analyze_attachments runs.")
        if not job_description:
            raise ValueError(
                "job_desciption is required before analyze_attachments runs."
            )

        prompt = f"""
        Analyze the candidate's CV against the job description.

        CV:
        {cv_text}

        Job description:
        {job_description}

        Identify the most relevant matching skills and experience.
        Do not invent information.
        """

        response = model.invoke(
            [SystemMessage(content=prompt.strip())] + self._last_messages(state)
        )
        return {"analysis": response.content}

    def draft_letter(self, state: CoverLetterState):
        analysis = state.get("analysis", "").strip()
        if not analysis:
            raise ValueError("analysis is required before draft_letter runs.")

        prompt = f"""
        Draft a cover letter based on the analysis of the candidate's CV and the job description.

        Analysis:
        {analysis}

        Do not invent information.
        """
        response = model.invoke(
            [SystemMessage(content=prompt.strip())]
        )
        return {"draft_letter": response.content}

    def check_quality(self, state: CoverLetterState):
        draft_letter = state.get("draft_letter", "").strip()
        if not draft_letter:
            raise ValueError("draft_letter is required before check_quality runs.")

        prompt = f"""
        Evaluate this cover letter and return structured result.

        Draft Letter:
        {draft_letter}
        """

        quality_model = model.with_structured_output(QualityCheckResult)
        result = quality_model.invoke([SystemMessage(content=prompt.strip())])

        return {
            "quality_check": result.model_dump(),
            "draft_letter": draft_letter
        }
