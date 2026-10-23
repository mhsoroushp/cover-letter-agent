import os
import atexit
import sqlite3
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

load_dotenv()
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "simple-agent")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

from langchain.messages import HumanMessage

from app.graph import build_agent

THREAD_ID = "cli-default123"
checkpoint_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
atexit.register(checkpoint_conn.close)
agent = build_agent(checkpointer=SqliteSaver(checkpoint_conn))
app = FastAPI(title="Simple Agent API", version="1.0.0")


class GenerateCoverLetterRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt for the agent")
    thread_id: str = Field(
        default=THREAD_ID,
        min_length=1,
        description="Conversation thread identifier for checkpointing",
    )


class GenerateCoverLetterResponse(BaseModel):
    thread_id: str
    analysis: str | None = None
    draft_letter: str | None = None
    quality_check: dict[str, Any] | None = None
    messages: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cover-letter", response_model=GenerateCoverLetterResponse)
def generate_cover_letter(
    payload: GenerateCoverLetterRequest,
) -> GenerateCoverLetterResponse:
    try:
        state = agent.invoke(
            {"messages": [HumanMessage(content=payload.message)]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    serialized_messages = []
    for message in state.get("messages", []):
        serialized_messages.append(
            {
                "type": message.__class__.__name__,
                "content": message.content,
            }
        )

    return GenerateCoverLetterResponse(
        thread_id=payload.thread_id,
        analysis=state.get("analysis"),
        draft_letter=state.get("draft_letter"),
        quality_check=state.get("quality_check"),
        messages=serialized_messages,
    )


def run_cli() -> None:
    while True:
        user_input = input("Enter a command: ")
        if user_input.lower() in ["exit", ":q"]:
            break

        messages = [HumanMessage(content=user_input)]
        state = agent.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        for message in state["messages"]:
            message.pretty_print()
        for key in ("cv_text", "job_desciption", "analysis", "draft_letter"):
            if key in state:
                print(f"\n{key}:\n{state[key]}")


if __name__ == "__main__":
    run_cli()
