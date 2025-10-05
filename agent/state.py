# agent/state.py
from typing_extensions import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Represents the state of the conversation flowing through the graph.
    `messages` keeps track of the conversation context, tool calls, and model outputs.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
