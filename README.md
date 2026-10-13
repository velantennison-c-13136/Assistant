# Personal Assistant: Start and Stop

Local chat UI powered by Qwen through Ollama, with LangGraph handling the
assistant workflow.

## Requirements

- Node.js 22.13 or newer (uses the built-in `node:sqlite` module).
- Python 3.9 or newer.
- Ollama installed, with `ollama` available on your terminal's PATH.
- Model files stored in `/Users/vela-13136/LLM/Qwen`.

No npm packages need to be installed. Chat history is stored in
`chats.sqlite` in this directory. Install the Python LangGraph dependencies
once before starting:

```bash
cd /Users/vela-13136/LLM/Assistant
python3 -m pip install -r requirements.txt
```

If you install them into a virtual environment, either activate that
environment before running the server or set `PYTHON_BIN` to that
environment's Python executable. Use `node server.js`, not `npm server.js`.

## Start the assistant

Open a terminal and run:

```bash
cd /Users/vela-13136/LLM/Assistant
node server.js
```

Keep the terminal open, then visit:

**http://127.0.0.1:8080/**

The app automatically:

1. Checks for an Ollama server at `http://127.0.0.1:11434`.
2. Reuses it if it is already running.
3. Otherwise starts Ollama using `~/LLM/Qwen` as the model directory.
4. Waits up to 30 seconds for Ollama to become ready.
5. Uses LangGraph to send your conversation to the selected Qwen model.

You do not need to run `ollama run qwen3.5:9b` separately. Sending a
message from the UI loads the selected model, then routes it through the
LangGraph assistant.

### Use a different model directory

```bash
cd /Users/vela-13136/LLM/Assistant
OLLAMA_MODELS=/absolute/path/to/models node server.js
```

This setting applies only when the app starts Ollama. An existing Ollama
server keeps its own model directory.

### Use a specific Python interpreter

```bash
cd /Users/vela-13136/LLM/Assistant
PYTHON_BIN=/absolute/path/to/python node server.js
```

Use this when LangGraph is installed in a virtual environment or a
non-default Python location.

## Stop the assistant

Press **Ctrl+C** in the terminal running `node server.js`.

- The chat UI server stops.
- If the app started Ollama, it stops that Ollama process too.
- If Ollama was already running independently, it remains running.
- Downloaded models and SQLite chat history are not deleted.

If you launched the app through a VS Code task:

1. Open the Command Palette with **Cmd+Shift+P**.
2. Select **Tasks: Terminate Task**.
3. Select the task that started the assistant.

### Stop from another terminal

Find the process listening on the UI port:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

Check that the entry belongs to this app's `node` process, then send it a
termination signal using its actual PID:

```bash
kill <NODE_PID>
```

Replace `<NODE_PID>` with the numeric process ID shown by `lsof`.
Do not terminate unrelated processes that happen to use the same port.

## Start or stop Ollama separately

To run Ollama independently:

```bash
OLLAMA_MODELS=/Users/vela-13136/LLM/Qwen \
OLLAMA_HOST=127.0.0.1:11434 \
ollama serve
```

Press **Ctrl+C** in that terminal to stop it. If you started Ollama through
the macOS app or a service manager, stop it through that app or service manager.

## Check the servers

```bash
curl --noproxy '*' http://127.0.0.1:11434/api/version
curl --noproxy '*' -I http://127.0.0.1:8080/
```

Ollama should return version JSON; the UI should return HTTP `200`.

## Troubleshooting

| Problem                              | Action                                                                                                                                                                     |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EADDRINUSE` on port 8080          | Check`lsof` as shown above. Reuse the running UI or stop only the conflicting app you own.                                                                               |
| `spawn ollama ENOENT`              | Ensure Ollama is installed and`command -v ollama` finds its executable.                                                                                                  |
| `Cannot start LangGraph assistant` | Check that`python3` (or `PYTHON_BIN`) exists and has `langgraph`, `langchain-ollama`, and `langchain-core` installed from [requirements.txt](./requirements.txt). |
| No models appear                     | Check the running Ollama server's model directory. With the server running, use`ollama pull qwen3.5:9b` to download the model there.                                     |
| Replies feel stuck on large file reviews | The assistant now streams progress, but large attachments can still be slow on local models. Try one file at a time, smaller excerpts, or narrower questions. Requests that run longer than 120 seconds are stopped automatically. |
| Ollama fails to become ready | Read the terminal error output; try the standalone Ollama command above to diagnose startup. |
| UI changes do not appear             | Restart`node server.js` and refresh the page. The server loads the HTML at startup.                                                                                      |
| `/v1` returns 404                  | `/v1` is an API base URL, not a chat page. Open port 8080 for the UI; use `/v1/models` to list models through the compatible API.                                      |
| Saved chats are missing              | Check that`chats.sqlite` exists in the app directory and that the app can write to that directory. History is shared by browsers using this local app.                   |
