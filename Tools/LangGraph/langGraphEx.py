from langgraph.graph import StateGraph, MessagesState, START, END
import json

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello geek!"}]}

graph = (MessagesState)
graph.add_node("mock_llm", mock_llm)
graph.add_eStateGraphdge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
result["messages"] = [{"role": msg.type, "content": msg.content} for msg in result["messages"]]
print(json.dumps(result, indent=2))
