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
for its contents again. Short approval or confirmation follow-ups skip rewriting
and remain linked to the preceding user request. If the rewrite model echoes
its own instructions or answers a question instead of rewriting it, the app
keeps the original request and reports that fallback. The selected answering
model then receives the improved prompt along with the unchanged attachments
and conversation context.

In Agent tools mode, current-information questions and explicit web-search
requests automatically search the web and fetch source pages before answering.
The answering model is instructed to cite checked source URLs; the app does not
append a separate "Sources checked" footer. If searching or
reading sources fails, the request reports an error rather than answering from
model memory. Prompt rewriting and answering receive the current local date so
the model's training cutoff is not treated as today's date.
Local file and directory requests use filesystem tools, even when they mention
the "current" environment or file dates. Web search and page fetching are disabled
for these tasks unless the user explicitly requests online research.

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

Uploaded contents and local paths can be used separately or together. Explicit
absolute, home-relative (`~/`), and relative (`./` or `../`) paths in the request
are inspected before the code assistant answers; quote paths containing spaces.
Requests containing these paths use Code assistant processing even when fast
attachment reading is selected. Uploaded contents remain available alongside
the filesystem observations and do not need to exist on disk.

Large local files are passed to the model as line-numbered excerpts selected
using the request, within a shared evidence budget. Small files remain complete.
This keeps accumulated project reads from overwhelming the model context or
the request deadline. Repeated inspections of the same path use the evidence
already collected; fix requests move to preparing a concrete edit instead of
announcing future work. Uploaded contents are unchanged.

Fix requests automatically inspect the supplied project, search quoted error
messages in a bounded set of source files, and prepare exact replacement edits.
Requests that include verification steps are not treated as plan-only requests.
Replacement text must match the original source uniquely, and proposed changes
receive a model review before being offered for approval. This review is not a
substitute for running the project's tests. Rejected changes are sent back for
revision; unfinished fixes are reported as incomplete, not successful.
Applying source changes still requires approval in the chat.

Empty or malformed JSON from the controller, edit proposal, or review model is
retried once with the original request and source context. If that retry fails,
the error identifies the failed stage; invalid responses never approve edits.
The notice that an exchange was not added to model history means the request
failed, not that the model generated that notice.

For calls with a finite output cap, a token-limit cutoff triggers one retry
with twice the output allowance, up to 8,192 tokens. When Ollama reports prompt
token counts, the retry reserves context room for the larger response, up to
32,768 context tokens. Original input and attachment contents are preserved.
Repeated cutoffs remain incomplete and identify the failing structured stage;
truncated replies never become source edits or approvals. Unlimited attachment
responses still fail explicitly if the model exhausts its context.

## Request timeouts

Code assistant tasks have a 15-minute total limit. Attachment-only requests
keep a four-minute total limit. Both have a four-minute inactivity limit that
resets whenever the Python worker sends output; progress cannot extend the
total limit indefinitely. Stop still cancels the running worker.

Override these limits in milliseconds when starting the server:

```bash
ASSISTANT_CHAT_TIMEOUT_MS=1200000 ASSISTANT_CHAT_IDLE_TIMEOUT_MS=300000 node server.js
```

Restart the Node server after changing these settings or the server code.
