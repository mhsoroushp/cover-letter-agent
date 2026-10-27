from langgraph.graph import END, START, StateGraph
from app.nodes import GraphNodes
from app.state import CoverLetterState
from app.services.blob_storage import BlobStorageService

blob_service = BlobStorageService()

nodes = GraphNodes(blob_service=blob_service)

def build_agent(checkpointer=None):
    agent_builder = StateGraph(CoverLetterState)
    agent_builder.add_node("load_documents", nodes.load_documents)
    agent_builder.add_node("analysis", nodes.analysis_attachments)
    agent_builder.add_node("draft_letter", nodes.draft_letter)
    agent_builder.add_node("check_quality", nodes.check_quality)
    agent_builder.add_edge(START, "load_documents")
    agent_builder.add_edge("load_documents", "analysis")
    agent_builder.add_edge("analysis", "draft_letter")
    agent_builder.add_edge("draft_letter", "check_quality")
    agent_builder.add_edge("check_quality", END)

    if checkpointer is None:
        return agent_builder.compile()
    return agent_builder.compile(checkpointer=checkpointer)


agent = build_agent()