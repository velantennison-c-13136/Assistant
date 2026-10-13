#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


WORKSPACE_DIR = Path(os.environ.get("ASSISTANT_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))).resolve()

SYSTEM_PROMPT = """You are an autonomous engineering assistant running with Qwen through Ollama and orchestrated by LangGraph.

You have access to execution tools to interact with the project workspace:
- read_file: read file contents from the workspace.
- write_file: create or update files in the workspace.
- run_command: run terminal commands such as test runners (e.g. pytest, npm test), linters, git, and python scripts.
- list_files: list files and directories in the workspace.

Guidelines:
- When asked to run tests, execute code, or verify changes, USE YOUR TOOLS immediately instead of saying you cannot run code.
- Check files, verify bug fixes, and execute test commands directly to confirm changes work.
- Provide clear, concrete, step-by-step explanations alongside tool results.
- Keep final answers grounded in evidence from files and tool execution outputs.
"""


def safe_resolve_path(target_path: str) -> Path:
    target = (WORKSPACE_DIR / target_path).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR)):
        raise PermissionError(f"Access denied: {target_path} is outside the allowed workspace ({WORKSPACE_DIR}).")
    return target


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a text file inside the workspace. Specify path relative to workspace or filename."""
    try:
        resolved = safe_resolve_path(file_path)
        if not resolved.exists():
            return f"Error: File '{file_path}' does not exist."
        if resolved.is_dir():
            return f"Error: '{file_path}' is a directory, not a file."
        return resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file in the workspace with the specified content."""
    try:
        resolved = safe_resolve_path(file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    except Exception as e:
        return f"Error writing file '{file_path}': {e}"


@tool
def list_files(directory_path: str = ".") -> str:
    """List files and folders in a workspace directory (relative path, defaults to workspace root)."""
    try:
        resolved = safe_resolve_path(directory_path)
        if not resolved.exists():
            return f"Error: Directory '{directory_path}' does not exist."
        if not resolved.is_dir():
            return f"Error: '{directory_path}' is a file, not a directory."
        items = sorted(os.listdir(resolved))
        entries = []
        for item in items:
            if item.startswith(".") and item != ".gitignore":
                continue
            item_path = resolved / item
            kind = "dir" if item_path.is_dir() else "file"
            entries.append(f"[{kind}] {item}")
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing files in '{directory_path}': {e}"


@tool
def run_command(command: str) -> str:
    """Execute a shell command inside the workspace directory (e.g. running pytest, tests, git, linters, python scripts)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=60
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = (stdout + ("\n" + stderr if stderr else "")).strip()
        if not combined:
            return f"Command finished with exit code {result.returncode} (no output)."
        return combined
    except subprocess.TimeoutExpired:
        return f"Command timed out after 60 seconds: {command}"
    except Exception as e:
        return f"Error running command '{command}': {e}"


ALL_TOOLS = [read_file, write_file, list_files, run_command]


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


def build_agent(model: str, endpoint: str):
    llm = ChatOllama(
        model=model,
        base_url=endpoint,
        temperature=0.2,
        num_ctx=8192,
        num_predict=1024,
        keep_alive="15m"
    )
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT)
    )


def run_agent(agent, messages: list[BaseMessage]) -> None:
    emit({"message": {"thinking": "LangGraph agent is analyzing your request...\n"}})

    final_content_parts: list[str] = []

    for mode, payload in agent.stream(
        {"messages": messages},
        stream_mode=["updates", "messages"]
    ):
        if mode == "messages":
            chunk, metadata = payload
            node = metadata.get("langgraph_node")
            if node == "agent":
                if hasattr(chunk, "content") and chunk.content:
                    text = text_from_response(chunk.content)
                    if text:
                        final_content_parts.append(text)
                        emit({"message": {"content": text}})
        elif mode == "updates":
            if "agent" in payload:
                agent_msgs = payload["agent"].get("messages", [])
                for msg in agent_msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "tool")
                            args_str = json.dumps(tool_call.get("args", {}))
                            emit({"message": {"thinking": f"⚙️ Calling tool: {tool_name}({args_str})\n"}})
            elif "tools" in payload:
                tool_msgs = payload["tools"].get("messages", [])
                for tmsg in tool_msgs:
                    content_preview = str(tmsg.content)
                    if len(content_preview) > 300:
                        content_preview = content_preview[:300] + "... (truncated)"
                    emit({"message": {"thinking": f"✔️ Tool result ({getattr(tmsg, 'name', 'tool')}):\n{content_preview}\n"}})

    final_text = "".join(final_content_parts).strip()
    if not final_text:
        # If model generated answers without streaming tokens, check agent final state
        emit({"message": {"content": "Task completed."}})
    emit({"done": True})


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

        agent = build_agent(model.strip(), endpoint.strip())
        run_agent(agent, [to_message(item) for item in messages])
        return 0
    except Exception as error:  # surfaced to the caller and shown in the UI
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
