FROM langchain/langgraph-api:latest-py3.14

# Install Python dependencies using LangGraph base image constraints
ADD requirements.txt /deps/outer-cover-letter-agent/src/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -r /deps/outer-cover-letter-agent/src/requirements.txt

# Copy only graph runtime code (no main.py/FastAPI entrypoint)
ADD app /deps/outer-cover-letter-agent/src/app

# Register graph for LangGraph API server
ENV LANGSERVE_GRAPHS='{"simple-agent":"/deps/outer-cover-letter-agent/src/app/graph.py:agent"}'

# Keep langgraph-api package wiring intact
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

WORKDIR /deps/outer-cover-letter-agent/src