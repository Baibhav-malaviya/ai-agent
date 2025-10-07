import os
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from agent.graph import create_graph

load_dotenv()


def main():
    # This config will be reused for every call to ensure memory continuity
    config = {"configurable": {"thread_id": "conversation_main"}}
    app = create_graph()

    # Inject SystemMessage ONCE at the start
    initial_state = {"messages": [SystemMessage(content="You are a helpful assistant named 'Villager Chat' who answers step by step.")]}

    # 🔑 Send initial state into the graph so it gets saved into memory
    app.invoke(initial_state, config=config)

    # Now enter conversation loop
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break

        user_message = {"messages": [HumanMessage(content=query)]}
        result = app.invoke(user_message, config=config)

        print("🤖 :", result["messages"][-1].content)


if __name__ == "__main__":
    main()
    # create_graph()