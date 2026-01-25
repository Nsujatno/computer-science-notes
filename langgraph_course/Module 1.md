# What is an agent?
- In non agentic workflows, most LLMs will follow a defined control flow
	- but in agentic workflows, the LLM **determines** the flow
- Certain systems can be more or less agentic
	- less agentic system: LLM determines only one step
	- more agentic system: LLM determines the entire flow
- More autonomous agents lose reliability
- Langchain has some features that langgraph can also use, for example a vector store

# Simple graph
![[Pasted image 20260121151015.png]]

## State:
- Serves as the input schema for all nodes and edges in the graph
``` Python
class State(TypedDict):
	graph_state: str
```

## Nodes:
- Nodes are just python functions
- Take in state, each node operates on the state
- By default, each node will also override the prior state value
``` Python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

## Edges:
- How we connect nodes
- Normal edges are used if you always use them
- Conditional edges are used want to optionally route between nodes
- A conditional edge is implemented as a function that returns the next node to visit based upon some logic
``` Python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```

## Graph construction
- State graph class to build graph
- Add nodes and edges
- Start node to send user input to the graph
- End node is a special node that represents a terminal node
``` Python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

## Graph invocation
- Compiled graph implements the runnable protocol
- `invoke` is one of the standard methods
- input is a dict: `{"graph_state": "Hi, this is lance."}`. which sets the initial value for our graph state dict
- invoke runs the entire graph synchronously
	- This waits for each step to complete before moving to the next
	- Returns the final state of the graph after all nodes have executed

# Chain
![[Pasted image 20260121153254.png]]

## Messages
- chat models can use messages, which capture different roles within a conversation
- different types such as:
	- HumanMessage
	- AIMessage
	- SystemMessage
	- ToolMessage
- each message can be supplied with a few things
	- content - content of the message
	- name - optionally, who is creating the message
	- response_metadata - optionally a dict of metadata that is specific to each model
``` Python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()
```

## Tools
- tools are needed when you want to connect LLM to external API
![[Pasted image 20260121153516.png]]
``` Python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])
```

Using messages as state
- Define state as messages state
- So messages is a list of messages
``` Python
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: list[AnyMessage]
```
- since our state gets overridden every time, the messages don't add up

## Reducer functions
- allow us to specify how state updates are performed
- since we want to append messages we can use a pre built `add_messages` reducer
``` Python
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
	messages: Annotated[list[AnyMessage], add_messages]
```

since having a list of messages in your state is so common, LangGraph has a pre built `MessagesState`
- pre build single messages key
- a list of AnyMessage objects and uses the add_messages reducer

``` Python
from langgraph.graph import MessagesState

class State(MessagesState):
	# add any keys needed beyond messages, which is pre-built
	pass
```

# Router
- we've built a graph that uses messages as a state and a chat model with bound tools
- the LLM determines whether to use a tool call or respond in natural language
- this is called a router

extend to 
- add a node that will call our tool
- add a conditional edge that will look at the chat model output, and route to our tool calling node or simply end if no tool call is performed

``` Python
# built in tool node and pass a list of tools
# built in tools_condition as our conditional edge
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# in build graph
builder.add_node("tools", ToolNode([multiply]))
builder.add_conditional_edge(
	"tool_calling_llm",
	# if latest message is a tool call -> route to tools
	# if latest message isn't a tool call -> route to end
	tools_condition	
)
```

## Agent
- previously we built the router that made it so if it needs a tool, it routes to tool
- if not, it routes to end
- what if we take that tool message and route it back to the model

this is the intuition behind ReAct.
- act - let the model call specific tools
- observe - pass the tool output back to the model
- reason - let the model reason about the tool output to decide what to do next
``` Python
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
	"assistant",
	tool_condition
)
# new edge added
builder.add_edge("tools", "assistant")
```

## Agent with memory
- previously we built the ReAct architecture
- new we want to introduce memory

``` Python
messages = [HumanMessage(content="Add 3 and 4")]
messages = react_graph.invoke({"messages": messages})
```
model gets 7

``` Python
messages = [HumanMessage(content="Multiply that by 2")]
messages = react_graph.invoke({"messages": messages})
```
the llm doesn't have memory of the value 7.

we need persistence to address this.
- langgraph can use a checkpointer to save the graph state after each step
- one of the easiest checkpointers to use is `MemorySaver`, an in memory key value store for Graph state
``` Python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

when we use memory, we need to specify a `thread_id`
- will store our collection of graph states

``` Python
# specify a thread
config = {"configurable": {"thread_id": "1"}}

# specify an input
messages = [HumanMessage(content="Add 3 and 4")]

messages = react_graph_memory.invoke({"messages": messages}, config)
```

if we pass the same thread_id, then we can proceed from the previously logged state checkpoint.
``` Python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
```
