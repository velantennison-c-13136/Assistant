'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const { setTimeout: delay } = require('node:timers/promises');
const { DatabaseSync } = require('node:sqlite');

const ollamaUrl = 'http://127.0.0.1:11434';
const langGraphScript = path.join(__dirname, 'langgraph_agent.py');
const pythonBin = process.env.PYTHON_BIN || 'python3';
const database = new DatabaseSync(path.join(__dirname, 'chats.sqlite'));
database.exec(`
  PRAGMA journal_mode = WAL;
  CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages TEXT NOT NULL
  );
`);
const saveChat = database.prepare(`
  INSERT INTO chats (id, title, model, updated_at, messages)
  VALUES (?, ?, ?, ?, ?)
  ON CONFLICT(id) DO UPDATE SET
    title = excluded.title,
    model = excluded.model,
    updated_at = excluded.updated_at,
    messages = excluded.messages
`);
const deleteChat = database.prepare('DELETE FROM chats WHERE id = ?');
const listChats = database.prepare(`
  SELECT id, title, model, updated_at AS updatedAt, messages
  FROM chats
  ORDER BY updated_at DESC
`);
const maxRequestBytes = 20 * 1024 * 1024;
const langGraphTimeoutMs = 240000;
let ollamaProcess;
let shuttingDown = false;
let shutdownPromise;

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function validChat(chat) {
  return chat && typeof chat.id === 'string' && chat.id.length > 0 &&
    typeof chat.title === 'string' && typeof chat.model === 'string' &&
    typeof chat.updatedAt === 'string' && !Number.isNaN(Date.parse(chat.updatedAt)) &&
    Array.isArray(chat.messages) && chat.messages.length > 0 &&
    chat.messages.every((item) => item && ['user', 'assistant'].includes(item.role) &&
      typeof item.content === 'string' &&
      (item.display === undefined || typeof item.display === 'string') &&
      (item.files === undefined || (Array.isArray(item.files) && item.files.every((name) => typeof name === 'string'))));
}

function validPromptMessage(message) {
  return message && ['user', 'assistant'].includes(message.role) && typeof message.content === 'string';
}

function normalizeOllamaOrigin(input) {
  let url;
  try {
    url = new URL(input);
  } catch {
    throw new HttpError(400, 'Use a valid local Ollama URL.');
  }
  if (!['http:', 'https:'].includes(url.protocol) ||
      !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) ||
      url.username || url.password || url.search || url.hash ||
      !['/', '/v1', '/v1/'].includes(url.pathname)) {
    throw new HttpError(400, 'Use a local Ollama URL such as http://127.0.0.1:11434.');
  }
  return url.origin;
}

function sendJson(response, status, data) {
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff'
  });
  response.end(JSON.stringify(data));
}

async function readJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxRequestBytes) throw new HttpError(413, 'Request body is too large.');
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw new HttpError(400, 'Request body must be valid JSON.');
  }
}

function saveChatsAtomically(chats) {
  const save = database.prepare(`
    INSERT OR IGNORE INTO chats (id, title, model, updated_at, messages)
    VALUES (?, ?, ?, ?, ?)
  `);
  database.exec('BEGIN IMMEDIATE');
  try {
    for (const chat of chats) {
      save.run(chat.id, chat.title, chat.model, chat.updatedAt, JSON.stringify(chat.messages));
    }
    database.exec('COMMIT');
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function sanitizePythonStderr(stderr) {
  return stderr
    .split(/\r?\n/)
    .filter((line) => line.trim() && ![
      'NotOpenSSLWarning',
      'LangChainPendingDeprecationWarning',
      'warnings.warn(',
      'JsonPlusSerializer'
    ].some((token) => line.includes(token)))
    .join('\n')
    .trim();
}

async function streamLangGraphChat({ model, endpoint, messages, mode, signal, response }) {
  return await new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [langGraphScript], {
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONWARNINGS: 'ignore' },
      stdio: ['pipe', 'pipe', 'pipe']
    });
    let stderr = '';
    let settled = false;
    let errorSent = false;
    const timeout = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      emitError('LangGraph timed out after 240 seconds. Try a smaller excerpt, fewer files, or a more specific request.');
      finish(new HttpError(504, 'LangGraph timed out after 240 seconds. Try a smaller excerpt, fewer files, or a more specific request.'));
    }, langGraphTimeoutMs);

    function finish(error, value) {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abortChild);
      if (error) reject(error);
      else resolve(value);
    }

    function abortChild() {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      finish(new HttpError(499, 'Chat generation cancelled.'));
    }

    function emitError(message) {
      if (errorSent || response.destroyed || response.writableEnded) return;
      errorSent = true;
      response.write(`${JSON.stringify({ error: message })}\n`);
    }

    child.once('error', (error) => {
      emitError(`Cannot start LangGraph assistant: ${error.message}`);
      finish(new HttpError(500, `Cannot start LangGraph assistant: ${error.message}`));
    });
    child.stdout.on('data', (chunk) => {
      if (!response.destroyed && !response.writableEnded) response.write(chunk);
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.once('exit', (code, signalName) => {
      if (signal?.aborted) {
        finish(new HttpError(499, 'Chat generation cancelled.'));
        return;
      }
      if (code !== 0) {
        const detail = sanitizePythonStderr(stderr) || `LangGraph exited with ${signalName || code}.`;
        emitError(detail);
        finish(new HttpError(500, detail));
        return;
      }
      finish(null);
    });

    if (signal?.aborted) {
      abortChild();
      return;
    }
    signal?.addEventListener('abort', abortChild, { once: true });
    child.stdin.end(JSON.stringify({ model, endpoint, messages, mode }));
  });
}

async function proxyOllama(request, response, target, method) {
  if (request.method !== method) {
    response.writeHead(405, { Allow: method });
    response.end('Method not allowed');
    return;
  }
  const abortController = new AbortController();
  response.on('close', () => {
    if (!response.writableEnded) abortController.abort();
  });
  const init = { method, signal: abortController.signal };
  if (method === 'POST') {
    const body = await readJson(request);
    if (typeof body?.model !== 'string' || !body.model.trim()) throw new HttpError(400, 'Model is required.');
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify({ model: body.model.trim(), stream: true });
  }
  let upstream;
  try {
    upstream = await fetch(`${ollamaUrl}${target}`, init);
  } catch {
    throw new HttpError(502, 'Ollama is not reachable.');
  }
  response.writeHead(upstream.status, {
    'Content-Type': upstream.headers.get('content-type') || 'application/json',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff'
  });
  try {
    for await (const chunk of upstream.body) response.write(chunk);
  } catch {
    // client disconnected or upstream aborted
  }
  if (!response.destroyed && !response.writableEnded) response.end();
}

async function handleApi(request, response, url) {
  if (url.pathname === '/api/ollama/tags') {
    await proxyOllama(request, response, '/api/tags', 'GET');
    return;
  }
  if (url.pathname === '/api/ollama/pull') {
    await proxyOllama(request, response, '/api/pull', 'POST');
    return;
  }
  if (url.pathname === '/api/assistant/chat') {
    if (request.method !== 'POST') {
      response.writeHead(405, { Allow: 'POST' });
      response.end('Method not allowed');
      return;
    }
    const body = await readJson(request);
    const endpoint = normalizeOllamaOrigin(body?.endpoint);
    if (typeof body?.model !== 'string' || !body.model.trim()) {
      throw new HttpError(400, 'Model is required.');
    }
    if (!Array.isArray(body.messages) || !body.messages.length || !body.messages.every(validPromptMessage)) {
      throw new HttpError(400, 'Chat messages are invalid.');
    }
    const mode = body.mode ?? 'agent';
    if (!['agent', 'attachment'].includes(mode)) {
      throw new HttpError(400, 'Chat mode must be agent or attachment.');
    }
    const abortController = new AbortController();
    response.on('close', () => {
      if (!response.writableEnded) abortController.abort();
    });
    response.writeHead(200, {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff'
    });
    await streamLangGraphChat({
      model: body.model.trim(),
      endpoint,
      messages: body.messages,
      mode,
      signal: abortController.signal,
      response
    });
    if (!response.destroyed && !response.writableEnded) response.end();
    return;
  }

  if (url.pathname === '/api/chats') {
    if (request.method === 'GET') {
      const chats = listChats.all().map((chat) => ({
        ...chat,
        messages: JSON.parse(chat.messages)
      }));
      sendJson(response, 200, chats);
      return;
    }
    if (request.method === 'POST') {
      const body = await readJson(request);
      if (!body || !Array.isArray(body.chats) || !body.chats.every(validChat) ||
          new Set(body.chats.map((chat) => chat.id)).size !== body.chats.length) {
        throw new HttpError(400, 'Chat import data is invalid.');
      }
      saveChatsAtomically(body.chats);
      sendJson(response, 200, { imported: body.chats.length });
      return;
    }
  }

  const chatMatch = /^\/api\/chats\/([^/]+)$/.exec(url.pathname);
  if (chatMatch) {
    let id;
    try {
      id = decodeURIComponent(chatMatch[1]);
    } catch {
      throw new HttpError(400, 'Chat ID is invalid.');
    }
    if (request.method === 'PUT') {
      const chat = await readJson(request);
      if (!validChat(chat) || chat.id !== id) throw new HttpError(400, 'Chat data is invalid.');
      saveChat.run(chat.id, chat.title, chat.model, chat.updatedAt, JSON.stringify(chat.messages));
      sendJson(response, 200, { saved: true });
      return;
    }
    if (request.method === 'DELETE') {
      deleteChat.run(id);
      sendJson(response, 200, { deleted: true });
      return;
    }
  }
  sendJson(response, 404, { error: 'Not found' });
}

const listenHost = process.env.HOST || '127.0.0.1';
const listenPort = Number(process.env.PORT) || 8765;
const accessPassword = process.env.ASSISTANT_PASSWORD || '';
const accessUser = process.env.ASSISTANT_USER || 'velan';
const loopbackOnly = ['127.0.0.1', 'localhost', '::1'].includes(listenHost);

function authorized(request) {
  if (!accessPassword) return true;
  const header = request.headers.authorization || '';
  if (!header.startsWith('Basic ')) return false;
  const supplied = Buffer.from(header.slice(6), 'base64').toString('utf8');
  const separator = supplied.indexOf(':');
  const safeEqual = (a, b) => {
    const left = Buffer.from(a);
    const right = Buffer.from(b);
    return left.length === right.length && crypto.timingSafeEqual(left, right);
  };
  const userOk = safeEqual(supplied.slice(0, Math.max(separator, 0)), accessUser);
  const passOk = safeEqual(supplied.slice(separator + 1), accessPassword);
  return userOk && passOk;
}

const server = http.createServer((request, response) => {
  if (!authorized(request)) {
    response.writeHead(401, { 'WWW-Authenticate': 'Basic realm="Assistant", charset="UTF-8"' });
    response.end('Authentication required');
    return;
  }
  let url;
  try {
    url = new URL(request.url, `http://localhost:${listenPort}`);
  } catch {
    sendJson(response, 400, { error: 'Invalid request URL.' });
    return;
  }
  if (url.pathname.startsWith('/api/')) {
    void handleApi(request, response, url).catch((error) => {
      if (response.headersSent) {
        if (!response.writableEnded && !response.destroyed) response.end();
        return;
      }
      const status = error instanceof HttpError ? error.status : 500;
      if (status === 500) console.error(`Chat database request failed: ${error.message}`);
      sendJson(response, status, { error: status === 500 ? 'Chat database request failed.' : error.message });
    });
    return;
  }
  if (request.url === '/' || request.url === '/index.html') {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' });
      response.end('Method not allowed');
      return;
    }
    fs.readFile(path.join(__dirname, 'index.html'), (error, page) => {
      if (error) {
        console.error(`Could not read the chat UI: ${error.message}`);
        response.writeHead(500, {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'no-store',
          'X-Content-Type-Options': 'nosniff'
        });
        response.end('Could not load the chat UI.');
        return;
      }
      response.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff'
      });
      response.end(request.method === 'HEAD' ? undefined : page);
    });
    return;
  }
  sendJson(response, 404, { error: 'Not found' });
});

async function ollamaReady() {
  try {
    const response = await fetch(`${ollamaUrl}/api/version`, {
      signal: AbortSignal.timeout(1000)
    });
    if (!response.ok) throw new Error(`Ollama health check returned HTTP ${response.status}.`);
    const data = await response.json();
    if (typeof data.version !== 'string') throw new Error('Port 11434 did not return an Ollama version.');
    return true;
  } catch (error) {
    if (error.cause?.code === 'ECONNREFUSED') return false;
    throw error;
  }
}

function shutdown(exitCode = 0) {
  if (exitCode) process.exitCode = exitCode;
  if (shutdownPromise) return shutdownPromise;
  shuttingDown = true;
  shutdownPromise = (async () => {
    const closeServer = new Promise((resolve) => {
      server.close(resolve);
      server.closeAllConnections();
    });
    const child = ollamaProcess;
    if (child?.pid && child.exitCode === null && child.signalCode === null) {
      await new Promise((resolve) => {
        const timer = setTimeout(() => {
          console.error('Ollama did not stop within 5 seconds; terminating the owned process.');
          child.kill('SIGKILL');
        }, 5000);
        child.once('exit', () => {
          clearTimeout(timer);
          resolve();
        });
        child.kill('SIGTERM');
      });
    }
    await closeServer;
    database.close();
  })();
  return shutdownPromise;
}

async function ensureOllama() {
  if (await ollamaReady()) {
    console.log(`Using existing Ollama server at ${ollamaUrl} (it will not be stopped by this app).`);
    return;
  }
  if (shuttingDown) return;
  const models = process.env.OLLAMA_MODELS || path.join(os.homedir(), 'LLM', 'Qwen');
  console.log(`Starting Ollama with model storage: ${models}`);
  ollamaProcess = spawn('ollama', ['serve'], {
    env: {
      ...process.env,
      OLLAMA_HOST: '127.0.0.1:11434',
      OLLAMA_MODELS: models,
      OLLAMA_ORIGINS: [process.env.OLLAMA_ORIGINS, 'http://Assistant.localhost:8765'].filter(Boolean).join(',')
    },
    stdio: 'inherit'
  });
  let childFailure;
  ollamaProcess.once('error', (error) => {
    childFailure = error;
    console.error(`Cannot start Ollama: ${error.message}. Install Ollama and ensure it is on PATH.`);
    void shutdown(1);
  });
  ollamaProcess.once('exit', (code, signal) => {
    if (shuttingDown) return;
    childFailure = new Error(`Ollama exited unexpectedly (${signal || code}).`);
    console.error(childFailure.message);
    void shutdown(1);
  });
  const deadline = Date.now() + 30000;
  while (!shuttingDown && Date.now() < deadline) {
    if (childFailure) throw childFailure;
    if (await ollamaReady()) {
      if (!shuttingDown) console.log(`Ollama is ready at ${ollamaUrl}.`);
      return;
    }
    await delay(250);
  }
  if (!shuttingDown) throw new Error('Ollama did not become ready within 30 seconds.');
}

process.once('SIGINT', () => { void shutdown(); });
process.once('SIGTERM', () => { void shutdown(); });
server.on('error', (error) => {
  console.error(`Cannot start chat UI: ${error.message}`);
  void shutdown(1);
});
if (!loopbackOnly && !accessPassword) {
  console.error('Refusing to listen on a non-local HOST without ASSISTANT_PASSWORD set.');
  process.exit(1);
}
server.listen(listenPort, listenHost, async () => {
  try {
    await ensureOllama();
    if (!shuttingDown) console.log(`Local Ollama Chat: http://${loopbackOnly ? 'Assistant.localhost' : listenHost}:${listenPort}/`);
  } catch (error) {
    console.error(`Chat startup failed: ${error.message}`);
    await shutdown(1);
  }
});
