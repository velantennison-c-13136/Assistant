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
