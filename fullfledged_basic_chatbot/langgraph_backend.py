from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(model = 'llama3.2')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message = state['messages']

    response = llm.invoke(message)

    return {'messages' : [response]}


graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpoint = InMemorySaver()

chatbot = graph.compile(checkpointer = checkpoint)


