import os
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from agent.graph import create_graph

load_dotenv()


def main():
    # Persistent conversation thread
    config = {"configurable": {"thread_id": "conversation_main"}}
    app = create_graph()

    # ✅ Initialize full AgentState with new keys
    initial_state = {
        "messages": [
            SystemMessage(
                content=(
                    "You are Villager Chat, a multi-agent assistant capable of handling "
                    "multiple tasks in one query (like math, signup, and utility). "
                    "Work step-by-step and route each intent correctly."
                )
            )
        ],
        "pending_tasks": [],       # 🧠 New: store multiple detected intents
        "completed_tasks": [],     # 🧾 Track what has already been processed
        "current_task": None       # 🔁 Helps manager know which task is active
    }

    # Load initial context into memory
    app.invoke(initial_state, config=config)

    # Conversation loop
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        user_message = {"messages": [HumanMessage(content=query)]}
        result = app.invoke(user_message, config=config)

        # Retrieve assistant message safely
        last_msg = result["messages"][-1].content if result["messages"] else "<no response>"

        # Debug (optional): Show state info for dev inspection
        pending = result.get("pending_tasks", [])
        completed = result.get("completed_tasks", [])
        current = result.get("current_task", None)

        print(f"\n🤖 {last_msg}")
        print(f"🧭 Pending Tasks: {pending}")
        print(f"✅ Completed Tasks: {completed}")
        print(f"🎯 Current Task: {current}")


if __name__ == "__main__":
    main()
