# Personal Assistant

Local chat UI powered by your selected Ollama model, with a deterministic code assistant for planning, inspection, and approval-based file changes.

## Prerequisites

- **Node.js** 22.13 or newer (uses the built-in `node:sqlite` module)
- **Python** 3.9 or newer
- **Ollama** installed and available on your terminal's `PATH`
- Ollama model files downloaded — for example:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:7b
  ollama pull gemma3:4b
  ollama pull mistral
  ```

No npm packages need to be installed.

## Install

Clone the repository and install the Python dependencies once before starting:

```bash
python3 -m pip install -r requirements.txt
```

If you install into a virtual environment, either activate it before running the server or set `PYTHON_BIN` to that environment's Python executable:

```bash
PYTHON_BIN=/path/to/venv/bin/python node server.js
```

## Start

```bash
cd Assistant
node server.js
```

Keep the terminal open, then visit:

**http://Assistant.localhost:8765/**

## Prompt improvement and file access

Before answering, the app uses `qwen3.5:9b` to improve only the current user's
prompt text. Attachment contents and conversation history are not sent to this
rewrite step. The selected answering model then receives the improved prompt
along with the unchanged attachments and conversation context.

Attached files are supplied as text, so the assistant does not need to find them
on disk. Attachment-bearing messages are retained for follow-up questions even
when older chat messages are compacted. Large attachments can still exceed the
selected model's context capacity.

In Code assistant mode, file and command tools can access paths outside the
default working directory, including absolute paths and `~/` paths, subject to
the permissions of the account running the server. The default working directory
is the repository's parent directory; set `ASSISTANT_WORKSPACE` to change it.
This does not grant additional operating-system permissions. File write and
delete proposals still require approval.
