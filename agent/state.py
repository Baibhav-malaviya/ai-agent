# agent/state.py
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Represents the evolving state across the LangGraph multi-agent system.
    Supports multi-intent task management and context preservation.
    """

    # Conversation history (user ↔️ agents ↔️ tools)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Intents identified by router or multi_intent_manager
    detected_intents: Optional[List[str]]  # e.g. ["math", "signup", "utility"]

    # Tasks still to be executed (managed by multi_intent_manager)
    pending_tasks: Optional[List[str]]  # dynamic queue (pop from here as processed)

    # Tracks already finished tasks
    completed_tasks: Optional[List[str]]  # useful for aggregator

    # Aggregated outputs from different agents
    results: Optional[Dict[str, Any]]  # e.g. {"math": 77, "signup": "success", "utility": {...}}

    # Optional helper fields for meta info
    current_intent: Optional[str]  # which agent is currently active
    last_output: Optional[Any]     # last message or data from the previous node
