from langchain.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
import operator

model = ChatOllama(
    model="gemma3:4b",
    base_url="http://localhost:11434",
    temperature=0
)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b

# Setup tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

# State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# LLM node
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    # call model directly 
    response = model.invoke(
        [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
        + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# Tool node
def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

# Routing logic
def should_continue(state: MessagesState) -> str:
    """Decide if we should continue the loop or stop"""
    # No tools binding available 
    return END

# Build graph
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Invoke
messages = [HumanMessage(content="Multiply 13 and 5.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()

# messages = [HumanMessage(content="What is the capital of France?")]
# result = agent.invoke({"messages": messages})
# for m in result["messages"]:
#     m.pretty_print()
