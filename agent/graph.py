# agent/graph.py
from typing import Union

from langgraph.graph import StateGraph, END
from langgraph.graph._branch import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .model import (
    greeting_call, signup_call, math_call, farewell_call, utility_call,
    get_multi_intents, RouterOptions
)
from .state import AgentState
from tools import all_tools, sign_up, utility_tools


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

def route_from_query(state: AgentState) -> Send:
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        return Send(RouterOptions.GREETING.value, state)

    last_user_query = human_messages[-1].content
    recent_messages = state["messages"][-4:] if len(state["messages"]) > 4 else state["messages"]
    conversation_context = "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content[:120]}"
        for msg in recent_messages if hasattr(msg, 'content') and msg.content
    )

    detected_intents, reasoning = get_multi_intents(last_user_query, conversation_context, RouterOptions)
    print(f"\n[Router] Query: '{last_user_query[:80]}...' → \nIntents: {detected_intents}")
    print(f"Reasoning: {reasoning}\n")

    if len(detected_intents) == 0:
        return Send(RouterOptions.GREETING.value, state)
    elif len(detected_intents) == 1:
        return Send(detected_intents[0], state)
    else:
        state["detected_intents"] = detected_intents
        return Send(RouterOptions.MANAGER.value, state)

def router_node(state: AgentState) -> AgentState:
    """Router node: keeps state intact."""
    return state


# ---------------------------------------------------------------------------
# CONTROL FLOW HELPERS
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # If a tool call exists, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[Continue] Tool calls found → executing tool.")
        return "tools"

    # If last message is from Human (new query), route
    if last_message.type == "human":
        print("[Continue] Human message → route to router.")
        return "router"

    # If last message is from System or AI without tool call → end
    print("[Continue] No new tool calls or human input → ending conversation.")
    return "end"



# ---------------------------------------------------------------------------
# MULTI-INTENT MANAGER NODE
# ---------------------------------------------------------------------------

def multi_intent_manager(state: AgentState) -> Union[AgentState, dict]:
    """
    Handles multiple detected intents sequentially.
    Executes each agent node (e.g., math, signup, utility) in order.
    """
    intents = state.get("detected_intents", [])
    print("detected_intents: ", intents)
    if not intents:
        print("[Manager] No intents found, routing to greeting.")
        return {"messages": [AIMessage(content="I'm here to help with anything you need.")]}

    print(f"[Manager] Handling multiple intents: {intents}")

    for intent in intents:
        print(f"[Manager] → Executing agent: {intent}")

        if intent == RouterOptions.MATH.value:
            state.update(math_call(state))
        elif intent == RouterOptions.SIGNUP.value:
            state.update(signup_call(state))
        elif intent == RouterOptions.UTILITY.value:
            state.update(utility_call(state))
        elif intent == RouterOptions.GREETING.value:
            state.update(greeting_call(state))
        elif intent == RouterOptions.END.value:
            state.update(farewell_call(state))

    print("[Manager] Completed all intents.")
    return state


# ---------------------------------------------------------------------------
# GRAPH CREATION
# ---------------------------------------------------------------------------

def create_graph():
    graph = StateGraph(AgentState)

    # --- Add router ---
    graph.add_node("router", router_node)

    # --- Add agent nodes ---
    graph.add_node("greeting", greeting_call)
    graph.add_node("signup", signup_call)
    graph.add_node("math", math_call)
    graph.add_node("utility", utility_call)
    graph.add_node("farewell", farewell_call)
    graph.add_node("multi_intent_manager", multi_intent_manager)

    # --- Add tool node ---
    all_available_tools = all_tools + [sign_up] + utility_tools
    graph.add_node("tools", ToolNode(all_available_tools))

    # -----------------------------------------------------------------------
    # CONDITIONAL ROUTING
    # -----------------------------------------------------------------------

    # Router → agent nodes
    graph.add_conditional_edges(
        "router",
        route_from_query,
        {
            RouterOptions.GREETING.value: "greeting",
            RouterOptions.SIGNUP.value: "signup",
            RouterOptions.MATH.value: "math",
            RouterOptions.UTILITY.value: "utility",
            RouterOptions.END.value: "farewell",
            RouterOptions.MANAGER.value: "multi_intent_manager",
        },
    )

    # After any agent, decide to use tools or re-route
    for node in ["greeting", "signup", "math", "utility"]:
        graph.add_conditional_edges(node, should_continue, {"tools": "tools", "router": "router", "end": END})

    # Farewell → end
    graph.add_edge("farewell", END)

    # Tools → router (decide next step again)
    graph.add_edge("tools", "router")

    # Multi-intent manager → router
    graph.add_edge("multi_intent_manager", "router")

    # -----------------------------------------------------------------------
    # Compile
    # -----------------------------------------------------------------------
    graph.set_entry_point("router")
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    # print("\n--- MERMAID GRAPH ---")
    # print(app.get_graph().draw_mermaid())
    # print("---------------------\n")

    return app
