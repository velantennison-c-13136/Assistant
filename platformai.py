"""LangChain chat model for the PlatformAI InternalAPI used by InQuest."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


MODEL_NAMES = (
    "zlabs-qwen35-122B-v1",
    "zlabs-qwen35-122B-v1-thinking-MOE",
)
MODEL_PREFIX = "platformai:"
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


@dataclass(frozen=True)
class PlatformAiHost:
    key: str
    base_url: str
    portal_id: str
    dc: str


def is_platformai_model(model: str) -> bool:
    return model.startswith(MODEL_PREFIX)


def _environment_key(key: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", key.upper())


def _hosts() -> list[PlatformAiHost]:
    known = {
        "in": "https://platformai.zoho.in",
        "csez": "https://platformai.csez.zohocorpin.com",
    }
    listed = [key.strip().lower() for key in os.getenv("PLATFORMAI_HOSTS", "").split(",") if key.strip()]
    if listed:
        keys = listed
    else:
        configured = {
            match.group(1).lower()
            for name in os.environ
            if (match := re.fullmatch(r"PLATFORMAI_([A-Z0-9_]+)_(?:BASE_URL|PORTAL_ID)", name))
        }
        keys = ["in", *sorted(configured - {"in"})]

    hosts: list[PlatformAiHost] = []
    for index, key in enumerate(keys):
        upper = _environment_key(key)
        first = index == 0
        base_url = (
            os.getenv(f"PLATFORMAI_{upper}_BASE_URL", "").strip()
            or (os.getenv("PLATFORMAI_BASE_URL", "").strip() if first else "")
            or known.get(key, "")
        )
        portal_id = (
            os.getenv(f"PLATFORMAI_{upper}_PORTAL_ID", "").strip()
            or (os.getenv("PLATFORMAI_PORTAL_ID", "").strip() if first else "")
        )
        if not base_url or not portal_id:
            continue
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"https", "http"}
            or not parsed_url.hostname
            or parsed_url.username
            or parsed_url.password
            or parsed_url.query
            or parsed_url.fragment
            or parsed_url.path not in {"", "/"}
        ):
            raise ValueError(f"PlatformAI gateway '{key}' has an invalid base URL.")
        dc = (
            os.getenv(f"PLATFORMAI_{upper}_DC", "").strip()
            or (os.getenv("PLATFORMAI_DC", "").strip() if first else "")
            or _dc_from_url(base_url)
        )
        hosts.append(PlatformAiHost(key, base_url.rstrip("/"), portal_id, dc))
    return hosts


def _dc_from_url(base_url: str) -> str:
    internal = re.fullmatch(r"https?://platformai\.([a-z0-9-]+)\.zohocorpin\.com/?", base_url, re.I)
    if internal:
        return internal.group(1).lower()
    public = re.fullmatch(r"https?://platformai(?:-ro)?\.zoho\.([a-z.]+?)/?", base_url, re.I)
    return public.group(1).lower() if public else ""


def platformai_model_list() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for host in _hosts():
        extra_key = f"PLATFORMAI_{_environment_key(host.key)}_EXTRA_MODELS"
        models = [*MODEL_NAMES, *(part.strip() for part in os.getenv(extra_key, "").split(","))]
        for model in dict.fromkeys(name for name in models if name):
            result.append({
                "id": f"{MODEL_PREFIX}{model}@{host.key}",
                "name": f"PlatformAI · {model}@{host.key}",
            })
    return result


def _host_for_model(model: str) -> tuple[str, PlatformAiHost]:
    if not is_platformai_model(model):
        raise ValueError("PlatformAI model IDs must start with 'platformai:'.")
    requested = model[len(MODEL_PREFIX):]
    if "@" in requested:
        wire_model, host_key = requested.rsplit("@", 1)
    else:
        wire_model, host_key = requested, ""
    hosts = _hosts()
    host = next((candidate for candidate in hosts if candidate.key == host_key), None) if host_key else (hosts[0] if hosts else None)
    if host is None:
        raise ValueError(
            "No configured PlatformAI gateway is available. Check PLATFORMAI_HOSTS and each gateway's portal ID."
        )
    if not wire_model:
        raise ValueError("A PlatformAI model name is required.")
    return wire_model, host


def _token(host: PlatformAiHost) -> str:
    dc = host.dc.lower()
    if not dc:
        raise ValueError(f"Cannot determine the Zoho data center for PlatformAI gateway '{host.key}'.")
    env_dc = _environment_key(dc)
    cache_key = f"ZAPI:{dc}"
    cached = _TOKEN_CACHE.get(cache_key)
    if cached and cached[1] > time.time():
        return cached[0]

    client_id = os.getenv(f"ZAPI_{env_dc}_CLIENT_ID", "").strip()
    client_secret = os.getenv(f"ZAPI_{env_dc}_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv(f"ZAPI_{env_dc}_REFRESH_TOKEN", "").strip()
    if client_id and client_secret and refresh_token:
        accounts_host = os.getenv(f"ZOHO_ACCOUNTS_HOST_{env_dc}", "").strip()
        if not accounts_host:
            accounts_host = "accounts.csez.zohocorpin.com" if dc == "csez" else f"accounts.zoho.{dc}"
        payload = urlencode({
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }).encode()
        request = Request(
            f"https://{accounts_host}/oauth/v2/token",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=15) as response:
                token_data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            status = getattr(error, "code", None)
            suffix = f" (HTTP {status})" if status else ""
            raise RuntimeError(f"Could not refresh the PlatformAI OAuth token for '{dc}'{suffix}.") from error
        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"PlatformAI OAuth token refresh returned no access token for '{dc}'.")
        expires_in = token_data.get("expires_in", 3600)
        try:
            expires_at = time.time() + max(0, float(expires_in) - 300)
        except (TypeError, ValueError):
            expires_at = time.time() + 3300
        _TOKEN_CACHE[cache_key] = (access_token, expires_at)
        return access_token

    access_token = os.getenv(f"ZAPI_{env_dc}_ACCESS_TOKEN", "").strip()
    if access_token:
        return access_token
    raise ValueError(
        f"PlatformAI credentials are missing for '{dc}'. Configure ZAPI_{env_dc}_CLIENT_ID, "
        f"ZAPI_{env_dc}_CLIENT_SECRET, and ZAPI_{env_dc}_REFRESH_TOKEN."
    )


def _wire_messages(messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for message in messages:
        content = message.content if isinstance(message.content, str) else json.dumps(message.content)
        if message.type == "system":
            if content:
                system_parts.append(content)
        elif isinstance(message, ToolMessage):
            wire.append({
                "role": "tool",
                "tool_results": [{
                    "id": message.tool_call_id,
                    "content": content,
                    "type": "function",
                }],
            })
        elif isinstance(message, AIMessage) and message.tool_calls:
            wire_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [{
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                    },
                } for call in message.tool_calls],
            }
            if content.strip():
                wire_message["content"] = content
            wire.append(wire_message)
        else:
            wire.append({"role": "assistant" if message.type == "ai" else "user", "content": content})
    return "\n\n".join(system_parts), wire


def _tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for index, call in enumerate(turn.get("tool_calls") or []):
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as error:
            raise ValueError("PlatformAI returned invalid JSON in a tool call.") from error
        if not isinstance(args, dict):
            raise ValueError("PlatformAI tool-call arguments must be a JSON object.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("PlatformAI returned a tool call without a function name.")
        calls.append({
            "name": name,
            "args": args,
            "id": call.get("id") or f"call-{index}",
            "type": "tool_call",
        })
    return calls


class ChatPlatformAI(BaseChatModel):
    """Chat model for PlatformAI's OAuth-protected InternalAPI endpoint."""

    model: str
    temperature: float = 0.1
    num_ctx: int = 12288
    num_predict: int = 1536

    @property
    def _llm_type(self) -> str:
        return "platformai"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        del tool_choice
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _request_body(self, messages: list[BaseMessage], kwargs: dict[str, Any], stream: bool) -> tuple[PlatformAiHost, dict[str, Any]]:
        wire_model, host = _host_for_model(self.model)
        context, wire_messages = _wire_messages(messages)
        body: dict[str, Any] = {
            "model": wire_model,
            "messages": wire_messages,
            "ai_vendor": os.getenv("PLATFORMAI_AI_VENDOR", "zia").strip() or "zia",
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream,
        }
        if context:
            body["context"] = context
        max_tokens = int(os.getenv("PLATFORMAI_MAX_TOKENS", "0") or "0")
        if max_tokens <= 0 and self.num_predict > 0:
            max_tokens = self.num_predict
        if max_tokens > 0:
            body["max_tokens"] = max_tokens
        tools = kwargs.get("tools")
        if tools:
            body["tools"] = tools
        return host, body

    def _post(self, host: PlatformAiHost, body: dict[str, Any], stream: bool) -> Any:
        url = f"{host.base_url}/internalapi/v2/ai/chat"
        for attempt in range(2):
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "chat-response-format": "msg-format",
                "portal_id": host.portal_id,
                "Authorization": f"Zoho-oauthtoken {_token(host)}",
            }
            request = Request(
                url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                timeout_ms = int(os.getenv("PLATFORMAI_TIMEOUT_MS", "600000") or "600000")
                timeout = timeout_ms / 1000 if timeout_ms > 0 else None
                return urlopen(request, timeout=timeout)
            except HTTPError as error:
                if error.code == 401 and attempt == 0:
                    _TOKEN_CACHE.pop(f"ZAPI:{host.dc.lower()}", None)
                    continue
                try:
                    error_data = json.loads(error.read().decode("utf-8"))
                    detail = error_data.get("error", {}).get("message")
                except (ValueError, AttributeError):
                    detail = None
                suffix = f": {detail}" if isinstance(detail, str) and detail else ""
                raise RuntimeError(f"PlatformAI HTTP {error.code}{suffix}") from error
            except (URLError, TimeoutError) as error:
                raise RuntimeError(f"PlatformAI request failed: {error.reason if isinstance(error, URLError) else error}") from error
        raise RuntimeError("PlatformAI authentication retry did not complete.")

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        del stop, run_manager
        host, body = self._request_body(messages, kwargs, stream=False)
        with self._post(host, body, stream=False) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        turn = (parsed.get("data") or {}).get("messages") or []
        if not turn or not isinstance(turn[0], dict):
            raise RuntimeError("PlatformAI returned an empty or malformed assistant response.")
        result = turn[0]
        content = result.get("content") or ""
        if not isinstance(content, str):
            raise RuntimeError("PlatformAI returned a non-text assistant response.")
        reason = result.get("finish_reason")
        message = AIMessage(
            content=content,
            tool_calls=_tool_calls(result),
            response_metadata={"done_reason": reason} if reason else {},
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"done_reason": reason},
        )

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        del stop
        if kwargs.get("tools"):
            raise ValueError("PlatformAI does not support streaming tool-call responses.")
        host, body = self._request_body(messages, kwargs, stream=True)
        accumulated = ""
        final_turn: dict[str, Any] | None = None
        with self._post(host, body, stream=True) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if not payload_text or payload_text == "[DONE]":
                    continue
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError as error:
                    raise RuntimeError("PlatformAI returned a malformed streaming event.") from error
                terminal = payload.get("data") or {}
                if terminal.get("is_final_chunk"):
                    turns = terminal.get("messages") or []
                    final_turn = turns[0] if turns else {}
                    final_content = final_turn.get("content") or ""
                    if isinstance(final_content, str) and final_content.startswith(accumulated):
                        remainder = final_content[len(accumulated):]
                        if remainder:
                            accumulated += remainder
                            yield ChatGenerationChunk(message=AIMessageChunk(content=remainder))
                    break
                turns = payload.get("messages") or []
                if turns and isinstance(turns[0], dict):
                    delta = turns[0].get("content") or ""
                    if isinstance(delta, str) and delta:
                        accumulated += delta
                        yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
        if final_turn is None:
            raise RuntimeError("PlatformAI stream ended before its final response event.")
        metadata = {}
        if final_turn.get("finish_reason"):
            metadata["done_reason"] = final_turn["finish_reason"]
        yield ChatGenerationChunk(message=AIMessageChunk(content="", response_metadata=metadata))
