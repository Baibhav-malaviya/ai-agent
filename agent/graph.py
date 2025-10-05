# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from .model import model_call
from .state import AgentState
from tools import all_tools


def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue to tool execution or end the graph.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


def create_graph():
    """
    Creates and compiles the LangGraph StateGraph with nodes, edges, and tools.
    """
    graph = StateGraph(AgentState)

    # Add agent node
    graph.add_node("our_agent", model_call)

    # Add tool node
    tool_node = ToolNode(tools=all_tools)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("our_agent")

    # Conditional edges: agent → tools (if needed) OR end
    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {"end": END, "continue": "tools"},
    )

    # After tools execute, loop back to agent for reasoning
    graph.add_edge("tools", "our_agent")
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

