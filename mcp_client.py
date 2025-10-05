import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
#
# async def get_tools():
#     server_path = Path(__file__).parent / "mcp_server.py"
#     server_params = StdioServerParameters(
#         command='uv',
#         args = ['run', str(server_path)]
#     )
#
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#
#             tools = await load_mcp_tools(session)
#             return tools

async def main():
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Load MCP tools as LangChain tools
            tools = await load_mcp_tools(session)

            print("Loaded tools:\n", {tool.name: tool.description for tool in tools})

            # Create LangGraph agent with Claude
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            agent = create_react_agent(llm, tools)
            # agent = llm.bind_tools(tools)

            response = await agent.ainvoke({"messages": [('user', "what is the sum of 20 and 7")]})
            print(f"Answer: {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())