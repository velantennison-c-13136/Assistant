#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover - optional warning class
    NotOpenSSLWarning = Warning

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="The default value of `allowed_objects` will change in a future version.")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph, add_messages


# Default working directory if paths are relative
DEFAULT_WORKING_DIR = Path(os.environ.get("ASSISTANT_WORKSPACE", Path(__file__).resolve().parent.parent)).resolve()

SYSTEM_PROMPT = """You are an autonomous engineering assistant running with Qwen through Ollama and orchestrated by LangGraph.

You have full, unrestricted access to the user's filesystem and tools to execute actions on their system:
- read_file: read file contents from any path (absolute path or relative path).
- write_file: create or update files at any path (absolute path or relative path).
- run_command: execute any terminal commands such as test runners (e.g. pytest, npm test), linters, git, and python scripts across any directory.
- list_files: list files and directories at any path.

Guidelines:
- You have FULL and UNRESTRICTED access to all filesystem paths on the system (e.g., /Users/vela-13136/LLM, /Users/vela-13136/LLM/Assistant, or any other directory).
- NEVER refuse to read, write, list files, or run commands by claiming workspace or sandbox limitations. You have full access and tools to do so.
- When asked to inspect files, edit code, run tests, or access any path, USE YOUR TOOLS immediately.
- Check files, verify bug fixes, and execute test commands directly to confirm changes work.
- Provide clear, concrete, step-by-step explanations alongside tool results.
- Keep final answers grounded in evidence from files and tool execution outputs.
"""


def resolve_path(target_path: str) -> Path:
    raw_path = Path(target_path.strip()).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (DEFAULT_WORKING_DIR / raw_path).resolve()


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a text file from any path on the filesystem (absolute or relative)."""
    try:
        resolved = resolve_path(file_path)
        if not resolved.exists():
            return f"Error: File '{file_path}' does not exist (resolved to '{resolved}')."
        if resolved.is_dir():
            return f"Error: '{file_path}' is a directory, not a file."
        return resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


@tool
def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file at any path on the filesystem (absolute or relative) with the specified content."""
    try:
        resolved = resolve_path(file_path)
        existed = resolved.exists()
        old_content = ""
        if existed:
            try:
                old_content = resolved.read_text(encoding="utf-8", errors="replace")
            except Exception:
                old_content = ""

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        # Compute relative or display path for Copilot-style display in chat
        try:
            rel_path = str(resolved.relative_to(DEFAULT_WORKING_DIR))
        except ValueError:
            rel_path = str(resolved)

        old_lines = old_content.splitlines(keepends=True) if existed else []
        new_lines = content.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm=""
        ))
        diff_text = "\n".join(diff_lines)

        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        status_type = "created" if not existed else "modified"
        emit({
            "file_edit": {
                "path": rel_path,
                "status": status_type,
                "diff": diff_text,
                "content": content,
                "additions": additions,
                "deletions": deletions
            }
        })

        return f"Successfully wrote {len(content)} characters to '{file_path}' (status: {status_type}, +{additions}/-{deletions})."
    except Exception as e:
        return f"Error writing file '{file_path}': {e}"


@tool
def list_files(directory_path: str = ".") -> str:
    """List files and folders at any path on the filesystem (absolute or relative, defaults to current directory)."""
    try:
        resolved = resolve_path(directory_path)
        if not resolved.exists():
            return f"Error: Directory '{directory_path}' does not exist (resolved to '{resolved}')."
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
def run_command(command: str, working_directory: str = "") -> str:
    """Execute a shell command anywhere on the system (e.g. running pytest, tests, git, linters, python scripts, file system commands). Optionally specify working_directory."""
    try:
        cwd = resolve_path(working_directory) if working_directory.strip() else DEFAULT_WORKING_DIR
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
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
TOOL_REQUEST_MARKERS = (
    "run ", "test", "pytest", "npm ", "pnpm ", "yarn ", "lint", "build", "execute",
    "command", "shell", "terminal", "edit ", "modify ", "update ", "change ",
    "write file", "create file", "save ", "list files", "read file", "open file",
    "workspace", "directory", "folder", "git ", "cat ", "ls ", "touch ", "mkdir ",
    "rm ", "check file", "inspect file", "show file", "view file"
)


def should_use_tools(messages: list[BaseMessage]) -> bool:
    if not messages:
        return False
    last_user = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
    if last_user is None or not isinstance(last_user.content, str):
        return False
    content = last_user.content.lower().strip()
    
    # Check keyword markers
    if any(marker in content for marker in TOOL_REQUEST_MARKERS):
        return True
        
    # Check if prompt contains path-like structures (e.g. /Users/..., ./..., or file extensions)
    if "/" in content or "\\" in content or "~/" in content:
        return True
    file_extensions = (".py", ".js", ".ts", ".json", ".html", ".css", ".md", ".txt", ".sh", ".yaml", ".yml", ".sql")
    if any(ext in content for ext in file_extensions):
        return True

    return False


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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


def build_llm(model: str, endpoint: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=endpoint,
        temperature=0.2,
        num_ctx=8192,
        num_predict=1024,
        keep_alive="15m"
    )


def build_chat_graph(model: str, endpoint: str):
    llm = build_llm(model, endpoint)

    def call_model(state: ChatState) -> dict[str, list[BaseMessage]]:
        emit({"message": {"thinking": "LangGraph is analyzing your request...\n"}})
        total_chars = sum(len(message.content) for message in state["messages"] if isinstance(message.content, str))
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


def build_agent(model: str, endpoint: str):
    from langgraph.prebuilt import create_react_agent

    llm = build_llm(model, endpoint)
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=False
    )


def run_tool_agent(agent, messages: list[BaseMessage]) -> None:
    emit({"message": {"thinking": "LangGraph agent is analyzing your request...\n"}})

    final_content_parts: list[str] = []

    try:
        for mode, payload in agent.stream(
            {"messages": messages},
            stream_mode=["updates", "messages"],
            config={"recursion_limit": 8}
        ):
            if mode == "messages":
                chunk, metadata = payload
                node = metadata.get("langgraph_node")
                if node == "agent" and hasattr(chunk, "content") and chunk.content:
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
                                emit({"message": {"thinking": f"Calling tool: {tool_name}({args_str})\n"}})
                elif "tools" in payload:
                    tool_msgs = payload["tools"].get("messages", [])
                    for tmsg in tool_msgs:
                        content_preview = str(tmsg.content)
                        if len(content_preview) > 300:
                            content_preview = content_preview[:300] + "... (truncated)"
                        emit({"message": {"thinking": f"Tool result ({getattr(tmsg, 'name', 'tool')}):\n{content_preview}\n"}})
    except GraphRecursionError as error:
        raise ValueError(
            "The tool-using LangGraph agent exceeded its step limit. Try a narrower request or ask for one action at a time."
        ) from error

    final_text = "".join(final_content_parts).strip()
    if not final_text:
        emit({"message": {"content": "Task completed."}})
    emit({"done": True})


def run_chat_graph(graph, messages: list[BaseMessage]) -> None:
    try:
        graph.invoke({"messages": messages}, config={"recursion_limit": 4})
    except GraphRecursionError as error:
        raise ValueError("The LangGraph chat flow exceeded its step limit unexpectedly.") from error


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

        chat_messages = [to_message(item) for item in messages]
        if should_use_tools(chat_messages):
            agent = build_agent(model.strip(), endpoint.strip())
            run_tool_agent(agent, chat_messages)
        else:
            graph = build_chat_graph(model.strip(), endpoint.strip())
            run_chat_graph(graph, chat_messages)
        return 0
    except Exception as error:  # surfaced to the caller and shown in the UI
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
