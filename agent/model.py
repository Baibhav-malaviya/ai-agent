# agent/model.py
from typing import Type
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import create_model, Field

from agent.state import AgentState
from tools import all_tools  # Import your tools (we'll assume you collect them in tools/__init__.py)


load_dotenv()
# Create the model instance
model = ChatOpenAI(
    model="gpt-4o-mini",   # Use gpt-4o for reasoning, gpt-4o-mini for cheaper calls
    temperature=0.3        # Low temperature for more deterministic responses
).bind_tools(all_tools)

def model_call(state):
    """
    LLM call function that uses a system prompt + conversation history to generate the next step.
    """

    response = model.invoke(state["messages"])
    return {"messages": [response]}

def get_route_path(
            query: str,
            route_options: Type[Enum],
            model_name: str = "gpt-4o-mini"
    ) -> tuple[str, str]:
        """
        LLM-based router that classifies a query into one of the given routes.

        Args:
            query: The user query to route.
            route_options: Enum class defining available routes.
            model_name: OpenAI model to use (default: gpt-4o-mini).

        Returns:
            (route_value, reasoning): The chosen route and explanation.
        """

        # Dynamically build decision schema
        RouteDecision = create_model(
            "RouteDecision",
            route=(route_options, Field(description="Chosen route based on query intent")),
            reasoning=(str, Field(description="Explanation of why this route was chosen")),
        )

        # Describe available routes
        options_description = "\n".join(
            f"- {opt.value}: {opt.name.replace('_', ' ').title()}"
            for opt in route_options
        )

        prompt = f"""
    You are a routing assistant. Analyze the query and pick the best route.

    Available routes:
    {options_description}

    User Query: {query}

    Return the most appropriate route and a brief reasoning.
    """

        # Ask LLM with structured output
        model = ChatOpenAI(model=model_name, temperature=0).with_structured_output(RouteDecision)
        decision = model.invoke(prompt)

        return decision.route.value, decision.reasoning



