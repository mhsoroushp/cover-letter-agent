import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

missing_azure_env_vars = [
    name
    for name, value in {
        "AZURE_OPENAI_DEPLOYMENT": azure_openai_deployment,
        "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
        "AZURE_OPENAI_API_KEY": azure_openai_api_key,
    }.items()
    if not value
]

if missing_azure_env_vars:
    missing_vars_csv = ", ".join(missing_azure_env_vars)
    raise RuntimeError(
        f"Missing required Azure OpenAI environment variables: {missing_vars_csv}"
    )

model = init_chat_model(
    model=azure_openai_deployment,
    model_provider="azure_openai",
    temperature=0,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_api_key,
    api_version=azure_openai_api_version,
)
