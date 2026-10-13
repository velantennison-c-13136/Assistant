#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph, add_messages


SYSTEM_PROMPT = """You are a helpful local AI assistant running with Qwen through Ollama and orchestrated by LangGraph.

Priorities:
- Give accurate, grounded answers.
- For coding, debugging, and logs, be concrete and step-by-step.
- Use attached file contents and earlier messages as evidence.
- If information is missing, say what is missing instead of guessing.
- Keep answers concise unless the user asks for more detail.
"""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def to_message(item: dict[str, object]) -> BaseMessage:
    role = item.get("role")
    content = item.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        raise ValueError("Each message must contain a supported role and string content.")
    if role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)


def text_from_response(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def build_graph(model: str, endpoint: str):
    llm = ChatOllama(
        model=model,
        base_url=endpoint,
        temperature=0.2,
        num_ctx=8192,
        num_predict=1024,
        keep_alive="15m"
    )

    def call_model(state: ChatState) -> dict[str, list[BaseMessage]]:
        total_chars = sum(len(message.content) for message in state["messages"] if isinstance(message.content, str))
        emit({"message": {"thinking": "LangGraph is analyzing your request...\n"}})
        if total_chars > 20000:
            emit({"message": {"thinking": "Large attachments or long chat history detected. This response may take longer.\n"}})
        parts: list[str] = []
        for chunk in llm.stream([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]], think=False):
            text = text_from_response(chunk.content)
            if text:
                parts.append(text)
                emit({"message": {"content": text}})
        content = "".join(parts).strip()
        if not content:
            raise ValueError("The assistant returned no answer.")
        emit({"done": True})
        return {"messages": [AIMessage(content=content)]}

    graph = StateGraph(ChatState)
    graph.add_node("assistant", call_model)
    graph.add_edge(START, "assistant")
    graph.add_edge("assistant", END)
    return graph.compile()


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        model = payload["model"]
        endpoint = payload["endpoint"]
        messages = payload["messages"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model is required.")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("Ollama endpoint is required.")
        if not isinstance(messages, list) or not messages:
            raise ValueError("At least one message is required.")
        graph = build_graph(model.strip(), endpoint.strip())
        graph.invoke({"messages": [to_message(item) for item in messages]})
        return 0
    except Exception as error:  # surfaced to the caller and shown in the UI
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
