import os
import atexit
import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

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

if __name__ == "__main__":
        while True:
            user_input = input("Enter a command: ")
            if user_input.lower() in ["exit", ":q"]:
                break

            messages = [HumanMessage(content=user_input)]
            state = agent.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": THREAD_ID}},
            )
            for m in state["messages"]:
                m.pretty_print()
            for key in ("cv_text", "job_desciption", "analysis", "draft_letter"):
                if key in state:
                    print(f"\n{key}:\n{state[key]}")