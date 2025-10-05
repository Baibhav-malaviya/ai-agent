# agent/model.py
from typing import Type
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import create_model, Field

from tools import all_tools, sign_up

load_dotenv()

# Create the model instance
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
).bind_tools(all_tools)

class RouterOptions(str, Enum):
    GREETING = "greeting"
    SIGNUP = "signup"
    MATH = "math"
    END = "end"

def model_call(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def get_route_path(
    query: str,
    conversation_context: str,
    route_options: Type[Enum],
    model_name: str = "gpt-4o-mini"
) -> tuple[str, str]:
    RouteDecision = create_model(
        "RouteDecision",
        route=(route_options, Field(description="Chosen route based on query intent")),
        reasoning=(str, Field(description="Explanation of why this route was chosen")),
    )

    options_description = "\n".join(
        f"- {opt.value}: {opt.name.replace('_', ' ').title()}"
        for opt in route_options
    )

    prompt = f"""
You are a routing assistant. Analyze the query and conversation context to pick the best route.

Available routes:
{options_description}

Recent Conversation Context:
{conversation_context}

Current User Query: {query}

Instructions:
- Continue the same route if conversation is ongoing
- Switch only for a clear new topic
- Use 'end' for goodbyes like 'bye', 'thanks'

Return concise route choice and reasoning.
"""

    model = ChatOpenAI(model=model_name, temperature=0).with_structured_output(RouteDecision)
    decision = model.invoke(prompt)

    return decision.route.value, decision.reasoning


# ============================================================================

GREETING_SYSTEM_PROMPT = """You are a friendly greeting assistant. 
- Welcome users warmly and professionally and ask for assist
- Answer general questions or have casual chat
- Guide users to specialized help if needed

Respond concisely. Do NOT perform math or handle signup; route appropriately."""

MATH_SYSTEM_PROMPT = """You are a math assistant with access to tools.
- Always use tools for calculations
- Provide clear, concise results
- Handle multi-step calculations using tools
- Further ask for related assist

Respond concisely. System executes tools for you."""

SIGNUP_SYSTEM_PROMPT = """You are a signup assistant.
- Collect all the required field as per the tool requirement
- Ask politely for missing info
- Confirm with given data, before proceeding with tool calling
- Also confirm if there is a possibility of typo

Respond concisely. Do NOT assume missing info."""

FAREWELL_SYSTEM_PROMPT = """You are a farewell assistant.
- Thank users warmly
- Offer future assistance
- End conversation positively

Respond concisely and naturally."""

# ============================================================================

greeting_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def greeting_call(state):
    messages = [SystemMessage(content=GREETING_SYSTEM_PROMPT)] + state["messages"]
    response = greeting_model.invoke(messages)
    return {"messages": [response]}

signup_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools([sign_up])

def signup_call(state):
    messages = [SystemMessage(content=SIGNUP_SYSTEM_PROMPT)] + state["messages"]
    response = signup_model.invoke(messages)
    return {"messages": [response]}

math_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(all_tools)

def math_call(state):
    messages = [SystemMessage(content=MATH_SYSTEM_PROMPT)] + state["messages"]
    response = math_model.invoke(messages)
    return {"messages": [response]}

farewell_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def farewell_call(state):
    messages = [SystemMessage(content=FAREWELL_SYSTEM_PROMPT)] + state["messages"]
    response = farewell_model.invoke(messages)
    return {"messages": [response]}
