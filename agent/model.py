# agent/model.py
from typing import Type, List
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import create_model, Field

from tools import all_tools, sign_up, utility_tools

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
    UTILITY = "utility"
    END = "end"
    AGGREGATOR = "aggregator"
    MANAGER = "multi_intent_manager"


def model_call(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def get_multi_intents(
    query: str,
    conversation_context: str,
    route_options: Type[Enum],
    model_name: str = "gpt-4o-mini"
) -> tuple[list[str], str]:
    """
    Detect multiple intents in a user query (e.g., 'math', 'signup', 'utility')
    using a structured model call.
    """
    MultiIntentDecision = create_model(
        "MultiIntentDecision",
        intents=(List[route_options], Field(description="All intents detected in the query")),
        reasoning=(str, Field(description="Explain how these intents were identified")),
    )

    options_description = "\n".join(
        f"- {opt.value}: {opt.name.replace('_', ' ').title()}"
        for opt in route_options
        if opt.value != "end"
    )

    prompt = f"""
You are an intent extraction assistant.
Identify *all* applicable intents from the user's query.

Available intents:
{options_description}

Conversation Context:
{conversation_context}

User Query:
{query}

Instructions:
- Output multiple intents if the query clearly includes more than one.
- If the query is a greeting or closing, return only that.
- Example: "add 5 and 3 and sign me up" → ["math", "signup"]
"""

    model = ChatOpenAI(model=model_name, temperature=0).with_structured_output(MultiIntentDecision)
    decision = model.invoke(prompt)
    print(f"decision: {decision}")

    detected_intents = [intent.value for intent in decision.intents]
    return detected_intents, decision.reasoning


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

UTILITY_SYSTEM_PROMPT = """
You are a utility assistant. 
Use the following tools to handle user requests:
- search_web: search for current or general information.
- word_count: count words in a given text.
- weather_info: get weather for a specified location.
- get_current_datetime: return the current date and time.

Always pick the most suitable tool and use it. 
Be brief, factual, and avoid small talk or guesses.
If a request doesn’t fit any tool, say you can only handle utility tasks.
"""


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

math_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(all_tools) #todo actually all_tools contains mostly math tools

def math_call(state):
    messages = [SystemMessage(content=MATH_SYSTEM_PROMPT)] + state["messages"]
    response = math_model.invoke(messages)
    return {"messages": [response]}

farewell_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def farewell_call(state):
    messages = [SystemMessage(content=FAREWELL_SYSTEM_PROMPT)] + state["messages"]
    response = farewell_model.invoke(messages)
    return {"messages": [response]}

utility_model = ChatOpenAI(model='gpt-4o-mini', temperature=0.6).bind_tools(utility_tools)
def utility_call(state):
    messages =[SystemMessage(content=UTILITY_SYSTEM_PROMPT)] + state['messages']
    response = utility_model.invoke(messages)
    return {"messages": [response]}