# Personal Assistant: Start and Stop

Local chat UI powered by Qwen through Ollama, with a deterministic code
assistant for planning, inspection, and approval-based file changes.

## Tools & Capabilities

The assistant can execute code, run tests, inspect files, and prepare
approval-gated edits or deletions across any system path:

- `read_file`: Read source code and test files from any system path (absolute or relative).
- `write_file`: Create and update files at any system path with diff generation.
- `run_command`: Run commands such as `pytest`, `npm test`, linters, git, and python scripts across any directory.
- `list_files`: List files and directories at any system path.

The assistant has full filesystem access to inspect files, edit code, and execute commands across any path (e.g. `/Users/vela-13136/LLM`, `/Users/vela-13136/LLM/Assistant`, or any local path).

### Current-event answers

Explicit web-search requests, current/future-year questions, and corrections to
those questions require a successful search and readable source pages before
the model can answer. Earlier assistant claims are excluded from this research
context; saved chat history is not deleted. The runtime date is supplied next to
the model's generation context, and checked source URLs accompany the answer.
Only the final model response is displayed as the answer. Intermediate tool
steps remain in the reasoning panel. Failed research is reported as an error
instead of falling back to model memory. Local code tasks and file attachments
are not automatically sent as web-search queries. Attachment processing uses
the mode described below; other requests keep the existing tool flow.
Code-assistant chats now use a deterministic controller instead of the generic
LangGraph ReAct tool loop. For local code tasks, the controller reuses project
roots mentioned earlier in the chat, blocks unnecessary web searches, avoids
broad whole-home filesystem scans, and stops once it has enough evidence.
Planned file writes and deletions are shown in the UI first and only apply
after explicit user approval. The assistant also refuses unrequested
README/documentation creation and requires a file to be read before it can be
modified within the same request.

### Faster attachment reading

Chats containing attachments default to **Fast reading (LangGraph, no tools)**
under **Settings > Attachment processing**. This uses a single-node LangGraph
flow, disables model thinking, and streams answer text as it is generated.
It avoids the tool-agent planning loop and tool-schema prompt overhead while
keeping LangGraph orchestration. Multiple attachments are also read locally
in parallel, preserving their selection order and existing text validation.

Choose **Code assistant (approval required)** before sending requests that need
planning, file edits, commands, or web research. The setting also applies to
follow-up messages and reopened chats containing attachments. Chats without
attachments use the same deterministic controller automatically. All
attachment text and conversation history are still supplied; large inputs can
still be slow or exceed the model's context window.

Run the offline attachment regressions:

```bash
python3 -m unittest discover -s tests -p 'test_attachments.py'
node --test tests/attachments.test.js
```

Search connection failures appear in the chat with retry and connection guidance.
Raw provider errors remain available under **Technical error details**. Failed
exchanges are not saved to model history, and the prompt is kept for retrying.
Run the error-message tests with `node --test tests/chat-error.test.js`.

### GPU memory errors

On connection, if no installed model is already selected, the UI selects the
smallest installed model by download size to reduce memory pressure. Explicit
selections and saved-chat model choices are preserved. Download size is only a
heuristic: model weights, context, and runtime buffers must all fit in memory.

MLX/Metal out-of-memory errors now show guidance to choose a smaller installed
model, close memory-heavy apps, or reduce attachments. The original error remains
under **Technical error details**. Requests are not automatically retried or
switched to another model, so completed tool actions cannot be replayed silently.
The failed exchange stays out of model history and the prompt remains available.

Run the memory-handling regressions with `node --test tests/model-memory.test.js`.

Run the offline research regressions with the Python interpreter used by the
server (or its virtual environment):

```bash
python3 -m unittest discover -s tests -p 'test_research.py'
```

### Changed files

The compact panel above the message input shows one card per file in the current
chat. Repeated edits update that card's cumulative diff, comparing the content
before its first edit with the latest content. Expand a card to see removed and
added lines together, with original/current line numbers and nearby context.
Counts reflect net changes, not the sum of intermediate edits. Reopening a saved
chat reconstructs the same view; starting or switching chats resets the panel.
Each card also has a maximize button for a large, scrollable diff viewer. Close
the viewer with its Close button or Escape to return to the compact panel.

Run the diff regression tests with `node --test tests/file-diff.test.js`.

## Requirements

- Node.js 22.13 or newer (uses the built-in `node:sqlite` module).
- Python 3.9 or newer.
- Ollama installed, with `ollama` available on your terminal's PATH.
- Model files stored in `/Users/vela-13136/LLM/Qwen`.

No npm packages need to be installed. Chat history is stored in
`chats.sqlite` in this directory. Saved conversations can be reopened even if
their original model was deleted. They use the currently selected installed
model instead, with a notice; if no model is available, the conversation is
still readable, but connecting to Ollama with an installed model is required
to continue chatting. The next saved reply records the selected model.

Run the saved-chat regressions with `node --test tests/saved-chats.test.js`.

Install the Python assistant dependencies
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

**http://Assistant.localhost:8765/**

The app automatically:

1. Checks for an Ollama server at `http://127.0.0.1:11434`.
2. Reuses it if it is already running.
3. Otherwise starts Ollama using `~/LLM/Qwen` as the model directory.
4. Waits up to 30 seconds for Ollama to become ready.
5. Uses LangGraph to send your conversation to the selected Qwen model,
   using streaming attachment reading or the tool agent as described above.

You do not need to run `ollama run qwen3.5:9b` separately. Sending a
message from the UI loads the selected model, then routes it through the
selected LangGraph flow.

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
lsof -nP -iTCP:8765 -sTCP:LISTEN
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
curl --noproxy '*' -I http://Assistant.localhost:8765/
```

Ollama should return version JSON; the UI should return HTTP `200`.

## Troubleshooting

| Problem                                  | Action                                                                                                                                                                                                                             |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EADDRINUSE` on port 8765              | Check`lsof` as shown above. Reuse the running UI or stop only the conflicting app you own.                                                                                                                                       |
| `spawn ollama ENOENT`                  | Ensure Ollama is installed and`command -v ollama` finds its executable.                                                                                                                                                          |
| `Cannot start LangGraph assistant`     | Check that`python3` (or `PYTHON_BIN`) exists and has `beautifulsoup4`, `langgraph`, `langchain-ollama`, and `langchain-core` installed from [requirements.txt](./requirements.txt).                                     |
| No models appear                         | Check the running Ollama server's model directory. With the server running, use`ollama pull qwen3.5:9b` to download the model there.                                                                                             |
| Replies feel stuck on large file reviews | The assistant now streams progress, keeps recent context focused, and waits up to 240 seconds before timing out. Large attachments can still be slow on local models, so try one file at a time, smaller excerpts, or narrower questions when possible. |
| The assistant searches the wrong project or the whole disk | Include the project path once in the chat. The assistant now reuses that root for later turns, avoids web search for local code tasks, and blocks broad `find /Users/...` scans when it already knows the active project root. |
| Ollama fails to become ready             | Read the terminal error output; try the standalone Ollama command above to diagnose startup.                                                                                                                                       |
| UI changes do not appear                 | Restart`node server.js` and refresh the page. The server loads the HTML at startup.                                                                                                                                              |
| `/v1` returns 404                      | `/v1` is an API base URL, not a chat page. Open port 8765 for the UI; use `/v1/models` to list models through the compatible API.                                                                                              |
| Saved chats are missing                  | Check that`chats.sqlite` exists in the app directory and that the app can write to that directory. History is shared by browsers using this local app.                                                                           |
