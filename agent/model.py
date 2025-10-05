# agent/model.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools import all_tools  # Import your tools (we'll assume you collect them in tools/__init__.py)
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp_client import get_tools
load_dotenv()
# Create the model instance

def print_tools(tools):
    for tool in tools:
        print(f"\nname: {tool.name}, desc: {tool.description}, args: {tool.args}")

async def build_model():
    tools = await get_tools()
    print(f"Loaded tools: ", [tool.name for tool in tools])
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    return model

async def model_call(state):
    """
    LLM call function that uses a system prompt + conversation history to generate the next step.
    """
    model = await build_model()
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}
