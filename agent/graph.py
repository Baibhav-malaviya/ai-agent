# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage

from .model import (
    greeting_call, signup_call, math_call, farewell_call, utility_call,
    get_route_path, RouterOptions
)
from .state import AgentState
from tools import all_tools, sign_up, utility_tools


def route_from_query(state: AgentState) -> str:
    """
    Routes user query to the right agent using LLM-based classifier.
    Now includes conversation context for better multi-turn routing.
    """
    # Find the last human message (user query)
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]

    if not human_messages:
        # Fallback if no human message found
        return RouterOptions.GREETING.value

    last_user_query = human_messages[-1].content

    # Build conversation context (last 4 messages for context)
    recent_messages = state["messages"][-4:] if len(state["messages"]) > 4 else state["messages"]
    conversation_context = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content[:100]}"
        for msg in recent_messages
        if hasattr(msg, 'content') and msg.content
    ])

    route, reasoning = get_route_path(last_user_query, conversation_context, RouterOptions)
    print(f"[Router] Query: '{last_user_query[:50]}...' → Route: {route}")
    print(f"[Router] Reasoning: {reasoning}")
    return route


def router_node(state: AgentState) -> dict:
    """
    Router node that doesn't modify state, just used for routing logic.
    """
    return state


def should_continue(state: AgentState) -> str:
    """
    Determines if the agent should continue to tools or end.
    Checks if the last message has tool calls.
    """
    last_message = state["messages"][-1]

    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"[Continue] Tool calls detected: {len(last_message.tool_calls)} tools")
        return "tools"

    print("[Continue] No tool calls, ending conversation")
    return "end"


def create_graph():
    graph = StateGraph(AgentState)

    # --- Add router node (for entry point) ---
    graph.add_node("router", router_node)

    # --- Add agent nodes ---
    graph.add_node("greeting", greeting_call)
    graph.add_node("signup", signup_call)
    graph.add_node("math", math_call)
    graph.add_node("utility", utility_call)
    graph.add_node("farewell", farewell_call)

    # --- Add tool execution node ---
    # Combine all tools for the tool node
    all_available_tools = all_tools + [sign_up] + utility_tools
    graph.add_node("tools", ToolNode(all_available_tools))

    # --- Router decides which agent to call ---
    graph.add_conditional_edges(
        "router",
        route_from_query,
        {
            RouterOptions.GREETING.value: "greeting",
            RouterOptions.SIGNUP.value: "signup",
            RouterOptions.MATH.value: "math",
            RouterOptions.UTILITY.value: "utility",
            RouterOptions.END.value: "farewell",  # Route to farewell agent instead of END
        },
    )

    # --- After greeting, check if tools needed or end ---
    graph.add_conditional_edges(
        "greeting",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # --- After signup, check if tools needed or end ---
    graph.add_conditional_edges(
        "signup",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # --- After math, check if tools needed or end ---
    graph.add_conditional_edges(
        "math",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    # --- After utility, check if tools needed or end ---
    graph.add_conditional_edges(
        "utility",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    # --- After farewell, conversation ends ---
    graph.add_edge("farewell", END)

    # --- After tools execute, go back to the appropriate agent ---
    # We route again after tools to determine which agent should handle the result
    graph.add_conditional_edges(
        "tools",
        route_from_query,
        {
            RouterOptions.GREETING.value: "greeting",
            RouterOptions.SIGNUP.value: "signup",
            RouterOptions.MATH.value: "math",
            RouterOptions.UTILITY.value: "utility",
            RouterOptions.END.value: END,
        },
    )

    # --- Set entry point to ROUTER ---
    graph.set_entry_point("router")

    memory = MemorySaver()
    app =  graph.compile(checkpointer=memory)
    print(app.get_graph().draw_mermaid())
    return app