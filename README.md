# cover-letter-agent

This project is a LangGraph-based cover letter assistant that reads a CV and a job description, a cover letter, and evaluates its quality. It is designed to help generate tailored, structured cover letter outputs using Azure services and an LLM workflow.

For the moment, the cv and job description are created as txt files in Azure blob storage. In the next version, it will added by the user from the Chat GUI. 

## Local setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally with LangGraph:

```bash
langgraph dev
```

## Alternative: Run with PostgreSQL and Redis (Docker)

Create the Dockerfile with the following command if not exist: 

```bash
uv run langgraph dockerfile Dockerfile
```

Build the app image:

```bash
langgraph build -t simple-agent:latest
```

Start services with Docker Compose:

```bash
docker compose up
```
