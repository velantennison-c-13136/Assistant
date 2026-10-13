#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import requests
except ImportError:
    requests = None

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

You have full access to tools to execute actions on the user's system and online:
- web_search: search the live web using DuckDuckGo to find real-time documentation, releases, error fixes, or current information.
- fetch_web_page: fetch and read the content of a specific webpage by URL.
- read_file: read file contents from any path (absolute path or relative path).
- write_file: create or update files at any path (absolute path or relative path).
- run_command: execute any terminal commands such as test runners (e.g. pytest, npm test), linters, git, and python scripts across any directory.
- list_files: list files and directories at any path.

CRITICAL INSTRUCTIONS:
- You CAN perform web searches and fetch web pages using `web_search` and `fetch_web_page`. NEVER claim "I cannot perform web searches" or "I do not have live internet browsing capabilities". When the user asks for current information, current leaders/officeholders, news, latest documentation, or explicitly to search online, CALL `web_search` immediately!
- When invoking a tool, ALWAYS use the official function/tool call mechanism. NEVER write the tool name as raw plain text (such as outputting 'web_search') in your response without making the actual tool call.
- You have FULL and UNRESTRICTED access to all filesystem paths on the user's machine (including `/Users/vela-13136/LLM`, `/Users/vela-13136/LLM/Assistant`, or any path).
- There is NO SANDBOX and NO WORKSPACE BOUNDARY. The tools (`write_file`, `read_file`, `list_files`, `run_command`, `web_search`, `fetch_web_page`) run directly on the host machine.
- NEVER say "I cannot access or modify your local file system paths", "I cannot write directly to /Users/vela-13136/LLM", or "The tools operate strictly within a workspace/sandbox". That is false!
- If the user asks to create, write, edit, read, list files, search online, or run commands, CALL THE TOOL IMMEDIATELY. Do not apologize or refuse.
- Check files, verify bug fixes, and execute test commands directly to confirm changes work.
- Provide clear, concrete, step-by-step explanations alongside tool results.
"""

CHAT_ONLY_PROMPT = """You are a helpful engineering assistant running with Qwen through Ollama."""


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

        # Check if the path is inside DEFAULT_WORKING_DIR
        try:
            rel_to_cwd = resolved.relative_to(DEFAULT_WORKING_DIR)
            rel_path = str(rel_to_cwd)
        except ValueError:
            # Absolute path outside Assistant directory (e.g. /Users/vela-13136/LLM/test.txt)
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


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for up-to-date information, documentation, news, or solutions using DuckDuckGo."""
    query = query.strip()
    if not query:
        return "Error: Empty search query."

    # Strategy 1: DuckDuckGo HTML interface
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://html.duckduckgo.com/"
    }
    results: list[dict[str, str]] = []

    try:
        url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")

        if BeautifulSoup:
            soup = BeautifulSoup(html_text, "html.parser")
            for r in soup.select(".result"):
                snippet_el = r.select_one(".result__snippet")
                title_el = r.select_one(".result__title a")
                if not snippet_el or not title_el:
                    continue
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True)
                href = title_el.get("href", "")
                if "uddg=" in href:
                    m = re.search(r"uddg=([^&]+)", href)
                    if m:
                        href = urllib.parse.unquote(m.group(1))
                results.append({"title": title, "url": href, "snippet": snippet})
                if len(results) >= max_results:
                    break
    except Exception:
        pass

    # Strategy 2 fallback: Lite DuckDuckGo
    if not results:
        try:
            lite_url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({"q": query}).encode("utf-8")
            lite_headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://lite.duckduckgo.com/"
            }
            req = urllib.request.Request(lite_url, data=data, headers=lite_headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                html_text = resp.read().decode("utf-8", errors="ignore")

            if BeautifulSoup:
                soup = BeautifulSoup(html_text, "html.parser")
                for a in soup.select("a.result-link")[:max_results]:
                    title = a.get_text(strip=True)
                    href = a.get("href", "")
                    if "uddg=" in href:
                        m = re.search(r"uddg=([^&]+)", href)
                        if m:
                            href = urllib.parse.unquote(m.group(1))
                    tr = a.find_parent("tr")
                    snippet = ""
                    if tr:
                        next_tr = tr.find_next_sibling("tr")
                        if next_tr:
                            snip_td = next_tr.select_one(".result-snippet")
                            if snip_td:
                                snippet = snip_td.get_text(strip=True)
                    results.append({"title": title, "url": href, "snippet": snippet})
        except Exception:
            pass

    # Strategy 3 fallback: DuckDuckGo Instant Answer API
    if not results:
        try:
            api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText") or data.get("Abstract")
            if abstract:
                results.append({
                    "title": data.get("Heading") or query,
                    "url": data.get("AbstractURL") or "",
                    "snippet": abstract
                })
        except Exception:
            pass

    if not results:
        return f"No web search results found for '{query}'."

    formatted: list[str] = [f"Web search results for '{query}':\n"]
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. [{r['title']}]({r['url']})\n   {r['snippet']}\n")
    return "\n".join(formatted)


@tool
def fetch_web_page(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract readable text from a web page URL."""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: URL must start with http:// or https://"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if requests:
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            raw_html = resp.text
        else:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_html = resp.read().decode("utf-8", errors="ignore")

        if BeautifulSoup:
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "svg", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... (truncated to {max_chars} chars)"
            return text if text else "Page loaded, but contains no readable text."
        else:
            clean = re.sub(r"<[^>]+>", " ", raw_html)
            clean = re.sub(r"\s+", " ", clean).strip()
            return clean[:max_chars]
    except Exception as e:
        return f"Error fetching web page '{url}': {e}"


ALL_TOOLS = [read_file, write_file, list_files, run_command, web_search, fetch_web_page]
TOOL_REQUEST_MARKERS = (
    "run ", "test", "pytest", "npm ", "pnpm ", "yarn ", "lint", "build", "execute",
    "command", "shell", "terminal", "edit ", "modify ", "update ", "change ",
    "write file", "create file", "save ", "list files", "read file", "open file",
    "workspace", "directory", "folder", "git ", "cat ", "ls ", "touch ", "mkdir ",
    "rm ", "check file", "inspect file", "show file", "view file",
    "websearch", "web search", "search web", "google", "duckduckgo", "search online",
    "search the web", "search for", "look up", "lookup", "online", "latest",
    "what is", "browse", "fetch url", "fetch webpage", "http://", "https://"
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
        for chunk in llm.stream([SystemMessage(content=CHAT_ONLY_PROMPT), *state["messages"]], think=False):
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
            config={"recursion_limit": 16}
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


RAW_TOOL_NAMES = {
    "web_search", "fetch_web_page", "read_file", "write_file",
    "list_files", "run_command"
}


def clean_history_for_tools(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Clean up misleading assistant claims about workspace/sandbox restrictions, lack of web search, or incomplete raw tool names from earlier turns."""
    cleaned: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            lower = msg.content.lower().strip()
            # Check if this assistant message falsely claims sandbox/workspace refusal
            if ("cannot write directly" in lower or "strictly within the" in lower or "cannot directly access or modify your local" in lower) and "workspace" in lower:
                continue
            # Check if this assistant message falsely claims it cannot perform web searches
            if ("cannot perform web searches" in lower or "cannot perform web search" in lower or "do not have live internet" in lower or "do not have access to tools like google_search" in lower or "cannot search the web" in lower):
                continue
            # Check if this assistant message got stuck emitting a raw tool name at the end
            lines = [l.strip() for l in msg.content.split("\n") if l.strip()]
            if lines and lines[-1].lower() in RAW_TOOL_NAMES:
                # If it was just an announcement plus the tool name (like "I will search... web_search"), drop it completely
                if len(lines) <= 2:
                    continue
                # Otherwise, remove the trailing raw tool name line
                msg.content = "\n".join(lines[:-1])
        cleaned.append(msg)
    # Ensure there is at least the last user message
    if not cleaned:
        return messages[-1:]
    return cleaned


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
        chat_messages = clean_history_for_tools(chat_messages)
        try:
            agent = build_agent(model.strip(), endpoint.strip())
            run_tool_agent(agent, chat_messages)
        except Exception:
            # Fallback to chat graph if tool agent encounters an unrecoverable failure
            graph = build_chat_graph(model.strip(), endpoint.strip())
            run_chat_graph(graph, chat_messages)
        return 0
    except Exception as error:  # surfaced to the caller and shown in the UI
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
