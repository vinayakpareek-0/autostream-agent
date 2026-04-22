"""CLI entry point — interactive conversation loop with the AutoStream agent."""

from langchain_core.messages import HumanMessage
from src.graph import graph


def main() -> None:
    """Run the agent in an interactive terminal loop."""
    config = {"configurable": {"thread_id": "cli-session"}}

    print("\n🎬 AutoStream AI Assistant")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        ai_message = response["messages"][-1].content
        print(f"Agent: {ai_message}\n")


if __name__ == "__main__":
    main()
