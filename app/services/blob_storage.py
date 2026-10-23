import os
from functools import lru_cache
from io import BytesIO
from urllib.parse import urlparse

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ResourceNotFoundError,
)
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
DEFAULT_CV_CONTAINER = os.getenv("AZURE_CV_CONTAINER")
DEFAULT_CV_BLOB_NAME = os.getenv("AZURE_CV_BLOB_NAME")
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")


CV_TEXT = """
Name: Hadi Soroush
Role: AI Engineer

Skills: 
 - Python
 - .Net, C#, AZ Entra ID, EF Core
 - SQL/NoSQL/Vector DBs
 - Javascript, Typescript, HTML, CSS
 - Angular 2+
 - CI/CD
 - Docker
 - Kubernetes
 - Azure
 - Azure Cognitive Search
 - LLM for downstream tasks
 - AI Agentic (LangChain, LangGraph)
 - Data Analysis
 - ML/AI, Deep Learning
 - Excel
 - Linux
"""

JOB_DESCRIPTION = """
Role: AI Engineer

Responsibilities:
 - Develop and maintain AI solutions
 - Collaborate with cross-functional teams
 - Implement and optimize machine learning models
 - Ensure scalability and performance of AI systems
 - Stay updated with the latest AI research and technologies
"""

@lru_cache(maxsize=1)
def _blob_service_client() -> BlobServiceClient:
    if not AZURE_STORAGE_ACCOUNT_URL:
        raise RuntimeError(
            "Missing required environment variable: AZURE_STORAGE_ACCOUNT_URL"
        )

    credential = DefaultAzureCredential()
    return BlobServiceClient(
        account_url=AZURE_STORAGE_ACCOUNT_URL,
        credential=credential,
    )

class BlobStorageService:

    def read_cv(self) -> str:
        return CV_TEXT.strip()

    def read_job_description(self) -> str:
        return JOB_DESCRIPTION.strip()

def read_blob_text(
    container_name: str,
    blob_name: str,
    *,
    encoding: str = "utf-8",
    max_chars: int = 12000,
) -> str:
    content = _read_blob_bytes(container_name, blob_name)

    try:
        text_content = content.decode(encoding)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Blob content is not valid UTF-8 text. "
            "Store the CV as a text-based file, for example .txt or .md."
        ) from exc

    return _truncate_text(text_content, max_chars=max_chars)


def read_blob_pdf_text(
    container_name: str,
    blob_name: str,
    *,
    max_chars: int = 12000,
) -> str:
    content = _read_blob_bytes(container_name, blob_name)

    try:
        pdf_reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise RuntimeError(
            "Failed to parse PDF content from blob. Ensure the file is a valid PDF."
        ) from exc

    extracted_pages = []
    for page in pdf_reader.pages:
        extracted_pages.append(page.extract_text() or "")

    text_content = "\n".join(extracted_pages).strip()
    if not text_content:
        text_content = _extract_text_with_document_intelligence(content)
        if not text_content:
            raise RuntimeError(
                "No readable text was extracted from the PDF. "
                "If this is a scanned CV, configure AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "for OCR fallback or upload a text-based PDF."
            )

    return _truncate_text(text_content, max_chars=max_chars)


def _read_blob_bytes(container_name: str, blob_name: str) -> bytes:
    if not container_name:
        raise ValueError("container_name is required")
    if not blob_name:
        raise ValueError("blob_name is required")
    normalized_blob_name = _normalize_blob_name(container_name, blob_name)

    try:
        blob_client = _blob_service_client().get_blob_client(
            container=container_name,
            blob=normalized_blob_name,
        )
        content = blob_client.download_blob(max_concurrency=1).readall()
    except ResourceNotFoundError as exc:
        raise RuntimeError(
            f"Blob not found: container='{container_name}', blob='{normalized_blob_name}'"
        ) from exc
    except ClientAuthenticationError as exc:
        raise RuntimeError(
            "Authentication to Azure Blob Storage failed. "
            "Ensure your identity has 'Storage Blob Data Reader' access."
        ) from exc
    except AzureError as exc:
        raise RuntimeError(
            f"Failed to read blob '{normalized_blob_name}' from container '{container_name}'."
        ) from exc

    return content


def _normalize_blob_name(container_name: str, blob_name: str) -> str:
    if blob_name.startswith(("http://", "https://")):
        parsed = urlparse(blob_name)
        path_parts = parsed.path.strip("/").split("/")
        if not path_parts or path_parts == [""]:
            raise ValueError(
                "blob_name URL does not include a blob path. "
                "Pass the blob path, for example: 'cv/soroush_cv.pdf'."
            )
        if path_parts[0] == container_name:
            resolved_blob_name = "/".join(path_parts[1:])
            if not resolved_blob_name:
                raise ValueError(
                    "blob_name URL points only to the container. "
                    "Include the blob path after the container."
                )
            return resolved_blob_name
        raise ValueError(
            "blob_name URL container does not match container_name. "
            f"Expected container '{container_name}'."
        )
    return blob_name


def _truncate_text(text_content: str, *, max_chars: int) -> str:
    if len(text_content) > max_chars:
        return (
            text_content[:max_chars]
            + "\n\n[Truncated content. Provide a smaller file or increase max_chars.]"
        )
    return text_content


@lru_cache(maxsize=1)
def _document_intelligence_client() -> DocumentIntelligenceClient | None:
    if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:
        return None

    credential = DefaultAzureCredential()
    return DocumentIntelligenceClient(
        endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=credential,
    )


def _extract_text_with_document_intelligence(pdf_bytes: bytes) -> str:
    client = _document_intelligence_client()
    if client is None:
        return ""

    try:
        poller = client.begin_analyze_document("prebuilt-read", body=BytesIO(pdf_bytes))
        result = poller.result()
    except ClientAuthenticationError as exc:
        raise RuntimeError(
            "Authentication to Azure Document Intelligence failed. "
            "Ensure your identity has access to the Document Intelligence resource."
        ) from exc
    except AzureError as exc:
        raise RuntimeError("OCR analysis with Azure Document Intelligence failed.") from exc

    lines = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)

    return "\n".join(lines).strip()
