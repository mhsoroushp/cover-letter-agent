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

class BlobStorageService:

    def __init__(self):
        load_dotenv()

        self.AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        self.DEFAULT_CV_CONTAINER = os.getenv("AZURE_CV_CONTAINER")
        self.DEFAULT_CV_BLOB_NAME = os.getenv("AZURE_CV_BLOB_NAME")
        self.DEFAULT_JOB_BLOB_NAME = os.getenv("AZURE_JOB_BLOB_NAME")

        if not self.AZURE_STORAGE_ACCOUNT_URL:
            raise RuntimeError(
                "Missing required environment variable: AZURE_STORAGE_ACCOUNT_URL"
            )
        
        if not self.DEFAULT_CV_CONTAINER:
            raise RuntimeError(
                "Missing required environment variable: AZURE_CV_CONTAINER"
            )
        
        if not self.DEFAULT_CV_BLOB_NAME:
            raise RuntimeError(
                "Missing required environment variable: AZURE_CV_BLOB_NAME"
            )
        if not self.DEFAULT_JOB_BLOB_NAME:
            raise RuntimeError(
                "Missing required environment variable: AZURE_JOB_BLOB_NAME"
            )

        self.blob_service_client = self._blob_service_client()

    @lru_cache(maxsize=1)
    def _blob_service_client(self) -> BlobServiceClient:
        credential = DefaultAzureCredential()
        return BlobServiceClient(
            account_url=self.AZURE_STORAGE_ACCOUNT_URL,
            credential=credential,
        )


    def read_cv(self) -> str:
        return self._read_blob_text(
            container_name=self.DEFAULT_CV_CONTAINER,
            blob_name=self.DEFAULT_CV_BLOB_NAME,
        ).strip()

    def read_job_description(self) -> str:
        return self._read_blob_text(
            container_name=self.DEFAULT_CV_CONTAINER,
            blob_name=self.DEFAULT_JOB_BLOB_NAME,
        ).strip()

    def _read_blob_text(
        self,
        container_name: str,
        blob_name: str,
        *,
        encoding: str = "utf-8",
        max_chars: int = 12000,
    ) -> str:
        content = self._read_blob_bytes(container_name, blob_name)

        try:
            text_content = content.decode(encoding)
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Blob content is not valid UTF-8 text. "
                "Store the CV as a text-based file, for example .txt or .md."
            ) from exc

        return self._truncate_text(text_content, max_chars=max_chars)


    def _read_blob_bytes(self, container_name: str, blob_name: str) -> bytes:

        if not container_name:
            raise ValueError("container_name is required")
        if not blob_name:
            raise ValueError("blob_name is required")
        normalized_blob_name = self._normalize_blob_name(container_name, blob_name)

        try:
            blob_client = self._blob_service_client().get_blob_client(
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


    def _normalize_blob_name(self,container_name: str, blob_name: str) -> str:
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
    

    def _truncate_text(self,text_content: str, max_chars: int) -> str:
        if len(text_content) > max_chars:
            return (
                text_content[:max_chars]
                + "\n\n[Truncated content. Provide a smaller file or increase max_chars.]"
            )
        return text_content
