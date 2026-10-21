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

Before answering, the app uses `qwen3.5:9b` to improve the current user's
prompt text. When the current request explicitly refers to the previous prompt,
that immediately preceding user prompt is also supplied as context; assistant
replies and attachment contents are not sent to the rewrite step. Attached file
names are provided so requests to print or format an attached file do not ask
for its contents again. The selected answering model then receives the improved
prompt along with the unchanged attachments and conversation context.

Attached files are supplied as text, so the assistant does not need to find them
on disk. Attachment-bearing messages are retained for follow-up questions even
when older chat messages are compacted. Large attachments can still exceed the
selected model's context capacity.

Attachment answers have no fixed output-token cap. If Ollama reports that an
output or context limit stopped generation, the UI marks the reply incomplete
instead of saving it as a completed exchange. A model can still omit information;
use the uploaded-file preview to compare its response with the original data.
Collapsed code blocks limit only the visible height: scroll or expand to view
the remaining output, and Copy copies the full generated block.

In Code assistant mode, file and command tools can access paths outside the
default working directory, including absolute paths and `~/` paths, subject to
the permissions of the account running the server. The default working directory
is the repository's parent directory; set `ASSISTANT_WORKSPACE` to change it.
This does not grant additional operating-system permissions. File write and
delete proposals still require approval.
