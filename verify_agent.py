from src.agent.graph import agent_app
from langchain_core.messages import HumanMessage
import os

def test_workflow():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Skipping test: GOOGLE_API_KEY not set.")
        return

    # Turn 1: Greeting
    print("\n--- Turn 1: Greeting ---")
    state = {"messages": [HumanMessage(content="Hi there!")]}
    res = agent_app.invoke(state)
    print(f"Intent: {res['intent']}")
    print(f"AI: {res['messages'][-1].content}")

    # Turn 2: Pricing Inquiry
    print("\n--- Turn 2: Pricing ---")
    state = res
    state["messages"].append(HumanMessage(content="What are your pricing plans?"))
    res = agent_app.invoke(state)
    print(f"Intent: {res['intent']}")
    print(f"AI: {res['messages'][-1].content}")

    # Turn 3: High Intent
    print("\n--- Turn 3: High Intent ---")
    state = res
    state["messages"].append(HumanMessage(content="I want to sign up for the Pro plan for my YouTube channel."))
    res = agent_app.invoke(state)
    print(f"Intent: {res['intent']}")
    print(f"AI: {res['messages'][-1].content}")
    print(f"Lead Name: {res.get('lead_name')}, Email: {res.get('lead_email')}, Platform: {res.get('lead_platform')}")

    # Turn 4: Providing missing info
    print("\n--- Turn 4: Providing Info ---")
    state = res
    state["messages"].append(HumanMessage(content="My name is Adarsh and my email is adarsh@example.com"))
    res = agent_app.invoke(state)
    print(f"Intent: {res['intent']}")
    print(f"AI: {res['messages'][-1].content}")
    print(f"Lead Name: {res.get('lead_name')}, Email: {res.get('lead_email')}, Platform: {res.get('lead_platform')}")
    print(f"Tool Called: {res.get('is_tool_called')}")

if __name__ == "__main__":
    test_workflow()
