"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// SIMULTANEOUS MULTI-TAB CHAT
// Each tab has its own isolated DOM pane + state. Tabs can all stream in
// parallel. The shared sidebar shows history for all tabs; the active tab's
// pane is visible, others are kept alive (display:none) so streaming continues.
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  // ── Shared state ────────────────────────────────────────────────────────────
  let connected = false;
  let installedModels = [];
  let platformAiModels = [];
  let platformAiError = '';
  let ollamaError = '';
  let selectedProvider = localStorage.getItem('assistant-provider-v1') === 'platformai' ? 'platformai' : 'ollama';
  let connectionAttempt = 0;
  let savedChats = [];
  let storageLoaded = false;
  const legacyStorageKey = 'ollama-chat-history-v1';
  const codeLanguages = new Set(['js','javascript','ts','typescript','json','py','python','java','sql','bash','sh','shell','html','xml','css']);
  const maxAttachmentBytes = 1024 * 1024;
  const largeAttachmentWarningBytes = 24 * 1024;

  // ── Tab registry ─────────────────────────────────────────────────────────────
  let tabs = [];       // array of { id, pane, tabEl, label }
  let activeTabId = null;
  let nextTabNum = 1;

  // ── Shared helpers ────────────────────────────────────────────────────────────
  function status(text, error = false) {
    $('status').textContent = text;
    $('status').dataset.error = String(error);
  }

  function showHistoryError(text) {
    $('storage-error').textContent = text;
  }

  async function chatRequest(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: { ...(options.headers || {}), ...(options.body ? { 'Content-Type': 'application/json' } : {}) }
    });
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`Chat history service returned invalid JSON (HTTP ${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Chat history service returned HTTP ${response.status}.`);
    return data;
  }

  function validChat(chat) {
    return chat && typeof chat.id === 'string' && typeof chat.title === 'string' &&
      typeof chat.model === 'string' && typeof chat.updatedAt === 'string' &&
      Array.isArray(chat.messages) && chat.messages.length > 0 &&
      chat.messages.every(item => item && ['user', 'assistant'].includes(item.role) &&
        typeof item.content === 'string' &&
        (item.display === undefined || typeof item.display === 'string') &&
        (item.files === undefined || (Array.isArray(item.files) && item.files.every(name => typeof name === 'string'))));
  }

  async function loadChats() {
    try {
      const data = await chatRequest('/api/chats');
      if (!Array.isArray(data) || !data.every(validChat) ||
          new Set(data.map(chat => chat.id)).size !== data.length) {
        throw new Error('SQLite returned invalid chat history.');
      }
      savedChats = data;
      storageLoaded = true;
      try {
        const raw = localStorage.getItem(legacyStorageKey);
        if (raw !== null) {
          const legacyChats = JSON.parse(raw);
          if (!Array.isArray(legacyChats) || !legacyChats.every(validChat) ||
              new Set(legacyChats.map(chat => chat.id)).size !== legacyChats.length) {
            throw new Error('Existing browser chat data is invalid and was left unchanged.');
          }
          await chatRequest('/api/chats', { method: 'POST', body: JSON.stringify({ chats: legacyChats }) });
          savedChats = await chatRequest('/api/chats');
          localStorage.removeItem(legacyStorageKey);
        }
      } catch (error) {
        showHistoryError(`Could not migrate existing browser history: ${error.message} The original data was left unchanged.`);
      }
      if (!$('storage-error').textContent) showHistoryError('');
    } catch (error) {
      showHistoryError(`Cannot load chat history from SQLite: ${error.message}`);
    }
    renderChats();
  }

  // ── Chat list sidebar ────────────────────────────────────────────────────────
  let ctxChat = null;

  function hideContextMenu() {
    $('chat-context-menu').classList.remove('visible');
    ctxChat = null;
  }

  function showContextMenu(x, y, chat) {
    ctxChat = chat;
    const menu = $('chat-context-menu');
    menu.classList.add('visible');
    const vw = window.innerWidth, vh = window.innerHeight;
    let left = x, top = y;
    const rect = menu.getBoundingClientRect();
    if (left + rect.width > vw - 8) left = vw - rect.width - 8;
    if (top + rect.height > vh - 8) top = vh - rect.height - 8;
    menu.style.left = `${Math.max(8, left)}px`;
    menu.style.top = `${Math.max(8, top)}px`;
  }

  document.addEventListener('mousedown', (event) => {
    if (!$('chat-context-menu').contains(event.target)) hideContextMenu();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hideContextMenu();
  });
  document.addEventListener('scroll', hideContextMenu, true);

  $('ctx-open-tab').addEventListener('click', () => {
    const chat = ctxChat;
    hideContextMenu();
    if (!chat) return;
    const params = new URLSearchParams({ chatId: chat.id });
    window.open(`${location.pathname}?${params}`, '_blank', 'noopener');
  });

  $('ctx-delete').addEventListener('click', async () => {
    const chat = ctxChat;
    hideContextMenu();
    if (!chat) return;
    const activePane = getActivePane();
    if ((activePane && (activePane.busy || activePane.approvalBusy || activePane.readingFiles)) ||
        !window.confirm(`Delete saved chat "${chat.title}"?`)) return;
    try {
      await chatRequest(`/api/chats/${encodeURIComponent(chat.id)}`, { method: 'DELETE' });
      savedChats = savedChats.filter(item => item.id !== chat.id);
      showHistoryError('');
      renderChats();
      // If any tab is showing this chat, reset it
      for (const tab of tabs) {
        if (tab.pane.activeChatId === chat.id) tab.pane.newConversation();
      }
      status('Saved chat deleted.');
    } catch (error) {
      showHistoryError(`Chat history was not updated: ${error.message}`);
    }
  });

  function renderChats() {
    $('chat-list').replaceChildren();
    if (!savedChats.length) { $('chat-list').textContent = 'No saved conversations yet.'; return; }
    const anyBusy = tabs.some(t => t.pane.busy || t.pane.approvalBusy || t.pane.readingFiles);
    for (const chat of [...savedChats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))) {
      const row = document.createElement('div');
      row.className = 'saved-chat';
      const open = document.createElement('button');
      open.type = 'button';
      const title = document.createElement('span');
      title.className = 'chat-title';
      title.textContent = chat.title;
      const meta = document.createElement('span');
      meta.className = 'chat-meta';
      meta.textContent = `${modelLabel(chat.model)} | ${new Date(chat.updatedAt).toLocaleString()}`;
      open.append(title, meta);
      // Mark active in the active pane
      const activePane = getActivePane();
      open.setAttribute('aria-current', String(activePane?.activeChatId === chat.id));
      open.disabled = anyBusy;
      open.addEventListener('click', () => { getActivePane()?.reopenChat(chat); });
      row.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        showContextMenu(event.clientX, event.clientY, chat);
      });
      row.append(open);
      $('chat-list').append(row);
    }
  }

  // ── Global controls sync ──────────────────────────────────────────────────────
  function syncGlobalControls() {
    // sync shared settings controls based on active pane state
    const pane = getActivePane();
    const busy = pane?.busy || false;
    const approvalBusy = pane?.approvalBusy || false;
    const readingFiles = pane?.readingFiles || false;
    for (const id of ['provider', 'endpoint', 'model', 'attachment-mode', 'connect', 'files']) {
      const el = $(id);
      if (el) el.disabled = busy || approvalBusy || readingFiles;
    }
    $('update').disabled = busy || readingFiles || !connected || !$('model').value ||
      $('model').value.startsWith('platformai:');
    $('clear').disabled = busy || approvalBusy || readingFiles;
  }

  function modelLabel(model) {
    return Array.from($('model').options).find(option => option.value === model)?.textContent || model;
  }

  function applyProviderModels(preferredModel = $('model').value) {
    const models = selectedProvider === 'platformai'
      ? platformAiModels
      : installedModels.map(item => ({ id: item.name, name: item.name }));
    $('model').replaceChildren();
    for (const item of models) $('model').add(new Option(item.name, item.id));
    $('model').value = models.some(item => item.id === preferredModel)
      ? preferredModel
      : (selectedProvider === 'ollama'
        ? (smallestInstalledModel(installedModels)?.name ?? models[0]?.id ?? '')
        : (models[0]?.id || ''));
    $('ollama-endpoint-setting').hidden = selectedProvider !== 'ollama';
    connected = models.length > 0;
    updateModelPill();
    for (const tab of tabs) tab.pane.refreshModelPill();
    if (connected) {
      status(`Ready with ${modelLabel($('model').value)} via ${selectedProvider === 'platformai' ? 'PlatformAI' : 'Ollama'}.`);
    } else {
      status(
        selectedProvider === 'platformai'
          ? (platformAiError || 'No PlatformAI models are configured. Check the server .env gateway and portal settings.')
          : (ollamaError || 'No Ollama models found. Start Ollama or switch the provider to PlatformAI.'),
        true
      );
    }
    syncGlobalControls();
    for (const tab of tabs) tab.pane.onActivated();
  }

  // ── Model pill (shared) ──────────────────────────────────────────────────────
  // The global #model select is the source of truth. Each tab's pane renders its
  // own per-pane model pill/popover via updatePaneModelPill(). This function just
  // refreshes all live tab panes so their pills stay in sync.
  function updateModelPill() {
    for (const t of tabs) { if (t.pane && t.pane.refreshModelPill) t.pane.refreshModelPill(); }
  }

  $('model').addEventListener('change', updateModelPill);

  // ── Shared rendering helpers ─────────────────────────────────────────────────
  function makeLink(href, labelText) {
    const a = document.createElement('a');
    a.href = href; a.textContent = labelText;
    a.target = '_blank'; a.rel = 'noopener noreferrer';
    return a;
  }

  function appendInlineFormatted(target, text) {
    const pattern = /(`[^`\n]+`|\[([^\]]+)\][ \t]*\((https?:\/\/[^\s)]+)\)|~~[^~\n]+~~|\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_|https?:\/\/[^\s<>"')\]]+)/g;
    let lastIndex = 0;
    for (const match of text.matchAll(pattern)) {
      const [token] = match;
      const index = match.index;
      if (index > lastIndex) target.append(document.createTextNode(text.slice(lastIndex, index)));
      if (token.startsWith('`') && token.endsWith('`')) {
        const code = document.createElement('code');
        code.textContent = token.slice(1, -1);
        target.append(code);
      } else if (token.startsWith('[') && match[2] && match[3]) {
        let linkLabel = match[2];
        if (/^https?:\/\//.test(linkLabel) && linkLabel.includes('%')) {
          try { linkLabel = decodeURIComponent(linkLabel); } catch { /* keep raw */ }
        }
        target.append(makeLink(match[3], linkLabel));
      } else if (token.startsWith('~~') && token.endsWith('~~')) {
        const del = document.createElement('del');
        del.textContent = token.slice(2, -2);
        target.append(del);
      } else if ((token.startsWith('**') && token.endsWith('**')) || (token.startsWith('__') && token.endsWith('__'))) {
        const strong = document.createElement('strong');
        appendInlineFormatted(strong, token.slice(2, -2));
        target.append(strong);
      } else if ((token.startsWith('*') && token.endsWith('*')) || (token.startsWith('_') && token.endsWith('_'))) {
        const em = document.createElement('em');
        appendInlineFormatted(em, token.slice(1, -1));
        target.append(em);
      } else if (/^https?:\/\//.test(token)) {
        const url = token.replace(/[.,;:!?'")\]]+$/, '');
        const trailing = token.slice(url.length);
        let label;
        try { label = decodeURIComponent(url); } catch { label = url; }
        target.append(makeLink(url, label));
        if (trailing) target.append(document.createTextNode(trailing));
      }
      lastIndex = index + token.length;
    }
    if (lastIndex < text.length) target.append(document.createTextNode(text.slice(lastIndex)));
  }

  function appendParagraph(container, lines) {
    const paragraph = document.createElement('p');
    appendInlineFormatted(paragraph, lines.join('\n'));
    container.append(paragraph);
  }

  function appendList(container, lines) {
    const ordered = /^\s*\d+\.\s+/.test(lines[0]);
    const list = document.createElement(ordered ? 'ol' : 'ul');
    for (const line of lines) {
      const item = document.createElement('li');
      appendInlineFormatted(item, line.replace(/^\s*(?:[-*]|\d+\.)\s+/, ''));
      list.append(item);
    }
    container.append(list);
  }

  function escapeHtml(text) {
    return text.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  }

  function highlightCode(text, language) {
    const supported = codeLanguages.has(language);
    if (!supported) return escapeHtml(text);
    const keywords = /^(?:class|const|let|var|function|return|if|else|for|while|switch|case|break|continue|try|catch|finally|throw|new|import|from|export|default|extends|implements|interface|type|public|private|protected|static|final|async|await|def|lambda|yield|with|in|is|not|and|or|null|true|false|void|package)\b/;
    const types = /^(?:String|Integer|Long|Boolean|Double|Float|List|Map|Set|Object|Promise|Array|Number|Date|Response|Request|Exception|Error)\b/;
    const patterns = [
      ['comment', /^(?:\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)/],
      ['string', /^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)/],
      ['number', /^\b\d+(?:\.\d+)?\b/],
      ['keyword', keywords], ['type', types],
      ['function', /^\b[A-Za-z_$][\w$]*(?=\s*\()/],
      ['operator', /^[()[\]{}=+\-/*<>!&|?:.,;]+/]
    ];
    let index = 0, output = '';
    while (index < text.length) {
      const source = text.slice(index);
      let matched = false;
      for (const [className, pattern] of patterns) {
        const match = pattern.exec(source);
        if (!match) continue;
        output += `<span class="token-${className}">${escapeHtml(match[0])}</span>`;
        index += match[0].length; matched = true; break;
      }
      if (!matched) { output += escapeHtml(text[index]); index++; }
    }
    return output;
  }

  function buildCodeBlock(language, source) {
    const pre = document.createElement('pre');
    const isLong = source.split('\n').length > 12;
    if (isLong) pre.classList.add('is-collapsed');
    const header = document.createElement('div');
    header.className = 'code-block-header';
    const label = document.createElement('div');
    label.className = 'code-language';
    label.textContent = language || 'code';
    const actions = document.createElement('div');
    actions.className = 'code-block-actions';
    if (isLong) {
      const toggle = document.createElement('button');
      toggle.type = 'button'; toggle.className = 'toggle-code'; toggle.textContent = 'Expand';
      toggle.setAttribute('aria-label', 'Expand code block');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.addEventListener('click', () => {
        const expanded = pre.classList.toggle('is-collapsed') === false;
        toggle.textContent = expanded ? 'Collapse' : 'Expand';
        toggle.setAttribute('aria-label', `${expanded ? 'Collapse' : 'Expand'} code block`);
        toggle.setAttribute('aria-expanded', String(expanded));
      });
      actions.append(toggle);
    }
    const copy = document.createElement('button');
    copy.type = 'button'; copy.className = 'copy-code'; copy.dataset.copy = source; copy.textContent = 'Copy';
    const code = document.createElement('code');
    code.innerHTML = highlightCode(source, (language || '').toLowerCase());
    actions.append(copy);
    header.append(label, actions);
    pre.append(header, code);
    return pre;
  }

  function renderMarkdownContent(container, text) {
    container.replaceChildren();
    const normalized = text.replace(/\r\n?/g, '\n');
    const blocks = normalized.split(/```/);
    for (let index = 0; index < blocks.length; index++) {
      const block = blocks[index];
      if (!block) continue;
      if (index % 2 === 1) {
        const [firstLine, ...rest] = block.split('\n');
        container.append(buildCodeBlock(firstLine.trim(), rest.join('\n').replace(/^\n/, '')));
        continue;
      }
      const lines = block.split('\n');
      for (let lineIndex = 0; lineIndex < lines.length;) {
        if (!lines[lineIndex].trim()) { lineIndex++; continue; }
        const headingMatch = /^(#{1,6})\s+(.+)$/.exec(lines[lineIndex]);
        if (headingMatch) {
          const heading = document.createElement(`h${headingMatch[1].length}`);
          appendInlineFormatted(heading, headingMatch[2].trim());
          container.append(heading); lineIndex++; continue;
        }
        if (/^\s*---+\s*$/.test(lines[lineIndex])) { container.append(document.createElement('hr')); lineIndex++; continue; }
        if (/^\s*>/.test(lines[lineIndex])) {
          const bqLines = [];
          while (lineIndex < lines.length && /^\s*>/.test(lines[lineIndex])) {
            bqLines.push(lines[lineIndex].replace(/^\s*>\s?/, '')); lineIndex++;
          }
          const blockquote = document.createElement('blockquote');
          const bqP = document.createElement('p');
          appendInlineFormatted(bqP, bqLines.join('\n'));
          blockquote.append(bqP); container.append(blockquote); continue;
        }
        if (/^\s*(?:[-*]|\d+\.)\s+/.test(lines[lineIndex])) {
          const listLines = [];
          while (lineIndex < lines.length && /^\s*(?:[-*]|\d+\.)\s+/.test(lines[lineIndex])) {
            listLines.push(lines[lineIndex]); lineIndex++;
          }
          appendList(container, listLines); continue;
        }
        if (/^\|/.test(lines[lineIndex].trim()) || /\|/.test(lines[lineIndex])) {
          const tableLines = [];
          while (lineIndex < lines.length && lines[lineIndex].trim() && /\|/.test(lines[lineIndex])) {
            tableLines.push(lines[lineIndex]); lineIndex++;
          }
          if (tableLines.length >= 2 && /^\|?[\s\-:|]+(\|[\s\-:|]+)*\|?$/.test(tableLines[1].trim())) {
            const table = document.createElement('table');
            const thead = document.createElement('thead'), tbody = document.createElement('tbody');
            const parseRow = (line) => line.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
            const headerRow = document.createElement('tr');
            for (const cell of parseRow(tableLines[0])) {
              const th = document.createElement('th');
              appendInlineFormatted(th, cell); headerRow.append(th);
            }
            thead.append(headerRow);
            for (let r = 2; r < tableLines.length; r++) {
              const tr = document.createElement('tr');
              for (const cell of parseRow(tableLines[r])) {
                const td = document.createElement('td');
                appendInlineFormatted(td, cell); tr.append(td);
              }
              tbody.append(tr);
            }
            table.append(thead, tbody); container.append(table); continue;
          }
          appendParagraph(container, tableLines); continue;
        }
        const paragraphLines = [];
        while (lineIndex < lines.length && lines[lineIndex].trim() &&
               !/^(#{1,6})\s+/.test(lines[lineIndex]) && !/^\s*---+\s*$/.test(lines[lineIndex]) &&
               !/^\s*(?:[-*]|\d+\.)\s+/.test(lines[lineIndex]) && !/\|/.test(lines[lineIndex])) {
          paragraphLines.push(lines[lineIndex]); lineIndex++;
        }
        appendParagraph(container, paragraphLines);
      }
    }
    if (!container.childNodes.length) container.textContent = text;
  }

  function setMessageContent(node, text, formatted = false) {
    node.classList.toggle('plain', !formatted);
    if (!formatted) { node.textContent = text; return; }
    renderMarkdownContent(node, text);
  }

  // ── Diff helpers ─────────────────────────────────────────────────────────────
  function fileLines(content) { return content.match(/[^\n]*\n|[^\n]+$/g) || []; }

  function originalFileLines(fileEdit) {
    const latest = fileLines(fileEdit.content);
    if (fileEdit.status.toLowerCase() === 'created') return [];
    if (!fileEdit.diff.trim()) return latest;
    const rows = fileEdit.diff.split('\n');
    const retainedNewlines = rows.some((row, index) =>
      /^[ +\-]/.test(row) && !/^(---|\+\+\+)/.test(row) && rows[index + 1] === '');
    const original = [];
    let cursor = 0, index = 0;
    while (index < rows.length) {
      const hunk = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(rows[index++]);
      if (!hunk) continue;
      const oldCount = Number(hunk[2] ?? 1), newCount = Number(hunk[4] ?? 1);
      const start = Number(hunk[3]) - (newCount ? 1 : 0);
      if (start < cursor || start > latest.length) throw new Error('Invalid diff hunk position.');
      original.push(...latest.slice(cursor, start));
      cursor = start;
      let oldSeen = 0, newSeen = 0;
      while (index < rows.length && !rows[index].startsWith('@@')) {
        const row = rows[index++];
        if (!row || row.startsWith('\\')) continue;
        const kind = row[0];
        if (![' ', '+', '-'].includes(kind)) throw new Error('Invalid diff row.');
        let token = row.slice(1);
        if (retainedNewlines ? rows[index] === '' : true) token += '\n';
        if (!retainedNewlines && rows[index]?.startsWith('\\')) token = token.replace(/\n$/, '');
        if (kind !== '-') {
          const actual = latest[cursor++];
          if (actual === undefined || actual.replace(/\r?\n$/, '') !== token.replace(/\r?\n$/, ''))
            throw new Error('Diff does not match the saved file content.');
          if (kind === ' ') original.push(actual);
          newSeen++;
        } else { original.push(token); }
        if (kind !== '+') oldSeen++;
      }
      if (oldSeen !== oldCount || newSeen !== newCount) throw new Error('Incomplete diff hunk.');
    }
    if (!rows.some(row => row.startsWith('@@'))) throw new Error('Diff has no valid hunks.');
    original.push(...latest.slice(cursor));
    return original;
  }

  function inlineFileDiff(original, latest) {
    const edits = [];
    function scores(a, b) {
      const row = new Uint32Array(b.length + 1);
      for (const line of a) {
        let diagonal = 0;
        for (let j = 1; j <= b.length; j++) {
          const above = row[j];
          row[j] = line === b[j-1] ? diagonal + 1 : Math.max(row[j], row[j-1]);
          diagonal = above;
        }
      }
      return row;
    }
    function compare(a, b) {
      let prefix = 0;
      while (prefix < a.length && prefix < b.length && a[prefix] === b[prefix]) edits.push(' ' + a[prefix++]);
      a = a.slice(prefix); b = b.slice(prefix);
      let suffix = 0;
      while (suffix < a.length && suffix < b.length && a[a.length-suffix-1] === b[b.length-suffix-1]) suffix++;
      const tail = suffix ? a.slice(-suffix) : [];
      if (suffix) { a = a.slice(0, -suffix); b = b.slice(0, -suffix); }
      if (!a.length) { edits.push(...b.map(l => '+' + l)); }
      else if (!b.length) { edits.push(...a.map(l => '-' + l)); }
      else if (a.length === 1) {
        const match = b.indexOf(a[0]);
        if (match < 0) { edits.push('-' + a[0], ...b.map(l => '+' + l)); }
        else { edits.push(...b.slice(0, match).map(l => '+' + l), ' ' + a[0], ...b.slice(match+1).map(l => '+' + l)); }
      } else {
        const midpoint = Math.floor(a.length / 2);
        const left = scores(a.slice(0, midpoint), b);
        const right = scores(a.slice(midpoint).reverse(), [...b].reverse());
        let split = 0;
        for (let j = 1; j <= b.length; j++) {
          if (left[j] + right[b.length-j] > left[split] + right[b.length-split]) split = j;
        }
        compare(a.slice(0, midpoint), b.slice(0, split));
        compare(a.slice(midpoint), b.slice(split));
      }
      edits.push(...tail.map(l => ' ' + l));
    }
    compare(original, latest);
    const additions = edits.filter(l => l[0] === '+').length;
    const deletions = edits.filter(l => l[0] === '-').length;
    const ranges = [];
    edits.forEach((line, index) => {
      if (line[0] === ' ') return;
      const start = Math.max(0, index - 3), end = Math.min(edits.length, index + 4);
      const previous = ranges.at(-1);
      if (previous && start <= previous.end) previous.end = end;
      else ranges.push({ start, end });
    });
    const diff = [];
    let oldLine = 1, newLine = 1, cursor = 0;
    for (const { start, end } of ranges) {
      while (cursor < start) {
        if (edits[cursor][0] !== '+') oldLine++;
        if (edits[cursor++][0] !== '-') newLine++;
      }
      const hunk = edits.slice(start, end);
      const oldCount = hunk.filter(l => l[0] !== '+').length;
      const newCount = hunk.filter(l => l[0] !== '-').length;
      diff.push(`@@ -${oldCount ? oldLine : oldLine-1},${oldCount} +${newCount ? newLine : newLine-1},${newCount} @@`);
      for (const line of hunk) {
        diff.push(line.replace(/\n$/, ''));
        if (!line.endsWith('\n')) diff.push('\\ No newline at end of file');
        if (line[0] !== '+') oldLine++;
        if (line[0] !== '-') newLine++;
        cursor++;
      }
    }
    return { diff: diff.join('\n'), additions, deletions };
  }

  function renderDiffLines(container, diffText) {
    const lines = diffText.split('\n');
    let oldLine = 0, newLine = 0;
    for (const line of lines) {
      if (!line) continue;
      const lineElem = document.createElement('div');
      lineElem.className = 'copilot-diff-line';
      const prefixElem = document.createElement('span');
      prefixElem.className = 'copilot-diff-prefix';
      const contentElem = document.createElement('span');
      contentElem.className = 'copilot-diff-content';
      if (line.startsWith('@@')) {
        const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)/.exec(line);
        if (hunk) { oldLine = Number(hunk[1]); newLine = Number(hunk[2]); }
        lineElem.classList.add('hunk');
        prefixElem.textContent = ' '; contentElem.textContent = line;
      } else if (line.startsWith('\\ No newline')) {
        lineElem.classList.add('hunk'); contentElem.textContent = line;
      } else if (/^(--- a\/|\+\+\+ b\/)/.test(line)) {
        lineElem.classList.add('header'); prefixElem.textContent = ' '; contentElem.textContent = line;
      } else if (line.startsWith('+')) {
        lineElem.classList.add('add'); prefixElem.textContent = '+'; contentElem.textContent = line.slice(1);
        if (!line.slice(1).trim()) lineElem.classList.add('blank');
      } else if (line.startsWith('-')) {
        lineElem.classList.add('del'); prefixElem.textContent = '-'; contentElem.textContent = line.slice(1);
        if (!line.slice(1).trim()) lineElem.classList.add('blank');
      } else {
        lineElem.classList.add('ctx'); prefixElem.textContent = ' ';
        const content = line.startsWith(' ') ? line.slice(1) : line;
        contentElem.textContent = content;
        if (!content.trim()) lineElem.classList.add('blank');
      }
      if (!lineElem.classList.contains('hunk') && !lineElem.classList.contains('header')) {
        const oldNumber = document.createElement('span'), newNumber = document.createElement('span');
        oldNumber.className = newNumber.className = 'copilot-diff-number';
        oldNumber.textContent = line[0] === '+' ? '' : String(oldLine++);
        newNumber.textContent = line[0] === '-' ? '' : String(newLine++);
        lineElem.append(oldNumber, newNumber);
      }
      lineElem.append(prefixElem, contentElem);
      container.append(lineElem);
    }
  }

  // ── Per-tab helpers ──────────────────────────────────────────────────────────
  function getActivePane() {
    return tabs.find(t => t.id === activeTabId)?.pane ?? null;
  }

  function baseUrl() {
    const url = new URL($('endpoint').value.trim());
    if (!['http:', 'https:'].includes(url.protocol) ||
        !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) ||
        url.username || url.password || url.search || url.hash ||
        !['/', '/v1', '/v1/'].includes(url.pathname)) {
      throw new Error('Use a local Ollama URL such as http://127.0.0.1:11434.');
    }
    return url.origin;
  }

  function endpointForModel(model) {
    return model.startsWith('platformai:') ? 'http://127.0.0.1:11434' : baseUrl();
  }

  function smallestInstalledModel(models) {
    return models.reduce((smallest, model) => {
      if (!Number.isFinite(model.size) || model.size <= 0) return smallest;
      return !smallest || model.size < smallest.size ? model : smallest;
    }, null);
  }

  function chatErrorMessage(error) {
    const detail = error.message || 'An unexpected error occurred.';
    if (/\b(out of memory|insufficient memory)\b|kIOGPUCommandBufferCallbackErrorOutOfMemory/i.test(detail)) {
      const smaller = smallestInstalledModel(installedModels);
      const suggestion = smaller && smaller.name !== $('model').value
        ? ` Select ${smaller.name}, the smallest installed model, using the model selector.`
        : ' Select a smaller installed model using the model selector if one is available.';
      return `The model ran out of GPU memory.${suggestion} Close other memory-heavy apps and retry.`;
    }
    if (/^Web search could not be completed\b/i.test(detail)) {
      if (/503|service unavailable/i.test(detail)) return 'Web search is temporarily unavailable (503). Please try again later.';
      if (/timed? out|timeout/i.test(detail)) return 'Web search timed out. Check your internet connection and try again.';
      return 'Web search could not connect. Check your internet or VPN connection.';
    }
    return detail;
  }

  async function readStream(response, consume) {
    if (!response.ok) throw new Error(`Ollama HTTP ${response.status}: ${await response.text()}`);
    if (!response.body) throw new Error('The browser did not provide a response stream.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;
    let model = null;
    function line(text) {
      if (!text.trim()) return;
      const chunk = JSON.parse(text);
      if (chunk.error) throw new Error(chunk.error);
      if (chunk.message && typeof chunk.message === 'object' && !Array.isArray(chunk.message)) {
        model = chunk.model || $('model').value;
      }
      consume({ ...chunk, model });
    }
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const text of lines) line(text);
        if (done) { line(buffer); finished = true; break; }
      }
    } finally {
      if (!finished) {
        try { await reader.cancel(); }
        catch (error) { console.error('Could not close response stream:', error); }
      }
      reader.releaseLock();
    }
  }

  // ── createChatPane — one per tab ─────────────────────────────────────────────
  function createChatPane(tabId) {
    // Per-pane state
    let history = [];
    let busy = false;
    let approvalBusy = false;
    let readingFiles = false;
    let controller = null;
    let activeChatId = null;
    let checkpoint = null;
    let attachments = [];
    const chatFileChanges = new Map();
    const historyItemByArticle = new WeakMap();

    // ── Build pane DOM ──────────────────────────────────────────────────────────
    const paneEl = document.createElement('div');
    paneEl.className = 'tab-pane';
    paneEl.dataset.tabId = tabId;

    // Empty state
    const emptyStateTemplate = document.createElement('div');
    emptyStateTemplate.id = `empty-${tabId}`;
    emptyStateTemplate.innerHTML = `
      <h2>How can I help you today?</h2>
      <p>Think through an idea, explore your code, or make sense of your logs.</p>
      <div class="suggestions" aria-label="Suggested prompts">
        <button type="button" data-prompt="Help me investigate an error in my application logs.">Investigate an error <span>Find the cause in your logs</span></button>
        <button type="button" data-prompt="Help me review and explain my code.">Understand code <span>Explore how something works</span></button>
        <button type="button" data-prompt="Help me plan an implementation step by step.">Plan a solution <span>Turn an idea into next steps</span></button>
      </div>`;

    const messagesEl = document.createElement('section');
    messagesEl.className = 'pane-messages';
    messagesEl.setAttribute('aria-label', 'Conversation');
    messagesEl.append(emptyStateTemplate.cloneNode(true));

    const fileChangesEl = document.createElement('section');
    fileChangesEl.className = 'pane-file-changes';
    fileChangesEl.setAttribute('aria-label', 'Changed files');
    fileChangesEl.hidden = true;
    const fileChangesHeading = document.createElement('div');
    fileChangesHeading.className = 'file-changes-heading';
    fileChangesHeading.textContent = 'Changed files';
    const fileChangeListEl = document.createElement('div');
    fileChangeListEl.className = 'pane-file-change-list';
    fileChangesEl.append(fileChangesHeading, fileChangeListEl);

    // Composer
    const composerEl = document.createElement('form');
    composerEl.className = 'pane-composer';
    composerEl.innerHTML = `
      <label for="prompt-${tabId}" class="sr-only">Your message</label>
      <textarea id="prompt-${tabId}" class="pane-prompt" placeholder="Message your Personal assistant..." rows="2"></textarea>
      <div class="pane-attachments" aria-label="Attached files"></div>
      <div class="actions">
        <label for="files-${tabId}" class="attach-label">+ Attach files
          <input id="files-${tabId}" type="file" multiple class="pane-files">
        </label>
        <button type="button" class="pane-model-switcher-btn model-switcher-btn" disabled>
          <span class="model-pill-dot"></span>
          <span class="pane-active-model-name">—</span>
          <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" style="opacity:.6"><path d="M4.427 7.427l3.396 3.396a.25.25 0 0 0 .354 0l3.396-3.396A.25.25 0 0 0 11.396 7H4.604a.25.25 0 0 0-.177.427z"/></svg>
        </button>
        <span class="composer-limit">Text, logs, code &middot; 1 MB</span>
        <button type="submit" class="pane-send" disabled>Send &uarr;</button>
        <button type="button" class="pane-stop" disabled>Stop</button>
      </div>`;

    const promptEl = composerEl.querySelector('.pane-prompt');
    const attachmentsEl = composerEl.querySelector('.pane-attachments');
    const filesEl = composerEl.querySelector('.pane-files');
    const sendBtn = composerEl.querySelector('.pane-send');
    const stopBtn = composerEl.querySelector('.pane-stop');
    const modelSwitcherBtn = composerEl.querySelector('.pane-model-switcher-btn');
    const activeModelNameEl = composerEl.querySelector('.pane-active-model-name');

    // Model popover (per-pane, anchored to pane model button)
    const modelPopoverEl = document.createElement('div');
    modelPopoverEl.className = 'model-popover';
    modelPopoverEl.setAttribute('role', 'listbox');
    modelPopoverEl.setAttribute('aria-label', 'Select model');
    modelPopoverEl.innerHTML = '<div class="model-popover-title">Switch model</div><div class="pane-model-popover-list"></div>';
    const modelPopoverListEl = modelPopoverEl.querySelector('.pane-model-popover-list');

    paneEl.append(messagesEl, fileChangesEl, composerEl, modelPopoverEl);

    // ── Internal helpers ──────────────────────────────────────────────────────
    function historyIndexForArticle(article) {
      const item = historyItemByArticle.get(article);
      return item ? history.indexOf(item) : -1;
    }

    function controls() {
      const isBusy = busy || approvalBusy || readingFiles;
      sendBtn.disabled = isBusy || !connected || (!promptEl.value.trim() && !attachments.length);
      stopBtn.disabled = !busy;
      filesEl.disabled = isBusy;
      modelSwitcherBtn.disabled = isBusy || !connected;
      if (activeTabId === tabId) {
        syncGlobalControls();
        renderChats();
      }
    }

    promptEl.addEventListener('input', controls);

    function updatePaneModelPill() {
      const modelName = $('model').value ? modelLabel($('model').value) : '—';
      activeModelNameEl.textContent = modelName;
    }

    function rebuildPaneModelPopover() {
      modelPopoverListEl.replaceChildren();
      for (const option of $('model').options) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'model-option-btn';
        btn.dataset.model = option.value;
        btn.textContent = option.textContent;
        if (option.value === $('model').value) btn.classList.add('active');
        btn.addEventListener('click', () => {
          if (busy || approvalBusy || readingFiles) return;
          if (option.value === $('model').value) { hidePaneModelPopover(); return; }
          $('model').value = option.value;
          updateModelPill();
          for (const t of tabs) t.pane.refreshModelPill();
          hidePaneModelPopover();
          status(`Model switched to ${option.textContent}.`);
        });
        modelPopoverListEl.append(btn);
      }
    }

    let paneModelPopoverOpen = false;
    function showPaneModelPopover() {
      rebuildPaneModelPopover();
      modelPopoverEl.classList.add('visible');
      paneModelPopoverOpen = true;
      const rect = modelSwitcherBtn.getBoundingClientRect();
      const popW = Math.max(modelPopoverEl.offsetWidth, 220);
      const popH = modelPopoverEl.offsetHeight;
      let left = rect.left, top = rect.top - popH - 6;
      if (left + popW > window.innerWidth - 8) left = window.innerWidth - popW - 8;
      if (top < 8) top = rect.bottom + 6;
      modelPopoverEl.style.left = `${Math.max(8, left)}px`;
      modelPopoverEl.style.top = `${top}px`;
    }
    function hidePaneModelPopover() {
      modelPopoverEl.classList.remove('visible');
      paneModelPopoverOpen = false;
    }

    modelSwitcherBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (paneModelPopoverOpen) hidePaneModelPopover(); else showPaneModelPopover();
    });
    document.addEventListener('mousedown', (e) => {
      if (paneModelPopoverOpen && !modelPopoverEl.contains(e.target) && e.target !== modelSwitcherBtn)
        hidePaneModelPopover();
    });

    function renderAttachments() {
      attachmentsEl.replaceChildren();
      for (const attachment of attachments) {
        const chip = document.createElement('div');
        chip.className = 'attachment-chip';
        chip.innerHTML = `
          <span class="attachment-chip-icon"><svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M4 1.5A1.5 1.5 0 0 0 2.5 3v10A1.5 1.5 0 0 0 4 14.5h8a1.5 1.5 0 0 0 1.5-1.5V6.414a1.5 1.5 0 0 0-.44-1.06L9.647 1.94A1.5 1.5 0 0 0 8.586 1.5H4zm5 1.414L12.086 6H9.5a.5.5 0 0 1-.5-.5V2.914zM3.5 3a.5.5 0 0 1 .5-.5h4v3A1.5 1.5 0 0 0 9.5 7h3v6a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V3z"/></svg></span>
          <span class="attachment-chip-name" title="${attachment.name}">${attachment.name}</span>
          <span class="attachment-chip-size">(${attachment.size.toLocaleString()} bytes)</span>`;
        const previewBtn = document.createElement('button');
        previewBtn.type = 'button'; previewBtn.className = 'attachment-chip-preview';
        previewBtn.innerHTML = `<svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/><path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8zm8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/></svg> Preview`;
        previewBtn.addEventListener('click', () => openAttachmentPreviewGlobal(attachment));
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button'; removeBtn.className = 'attachment-chip-remove'; removeBtn.textContent = 'Remove';
        removeBtn.addEventListener('click', () => {
          attachments.splice(attachments.indexOf(attachment), 1);
          renderAttachments(); controls();
        });
        chip.append(previewBtn, removeBtn);
        attachmentsEl.append(chip);
      }
    }

    function buildAttachedChips(item) {
      if (!item.files?.length) return null;
      const names = document.createElement('div');
      names.className = 'hint';
      if (item.fileContents?.length) {
        names.style.cssText = 'display:flex;flex-wrap:wrap;align-items:center;gap:6px';
        names.append(Object.assign(document.createElement('span'), { textContent: 'Attached:' }));
        for (const fc of item.fileContents) {
          const chip = document.createElement('button');
          chip.type = 'button'; chip.className = 'attachment-chip-preview'; chip.style.fontSize = '11px';
          chip.innerHTML = `<svg viewBox="0 0 16 16" width="11" height="11" fill="currentColor"><path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/><path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8zm8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/></svg> ${fc.name}`;
          chip.addEventListener('click', () => openAttachmentPreviewGlobal(fc));
          names.append(chip);
        }
      } else {
        names.textContent = `Attached: ${item.files.join(', ')}`;
      }
      return names;
    }

    // ── File change cards ──────────────────────────────────────────────────────
    function createCopilotFileCard(fileEdit) {
      const card = document.createElement('div');
      card.className = 'copilot-file-card';
      const header = document.createElement('div');
      header.className = 'copilot-file-header';
      const icon = document.createElement('span');
      icon.className = 'copilot-file-icon';
      icon.innerHTML = `<svg viewBox="0 0 16 16"><path d="M4 1.5A1.5 1.5 0 0 0 2.5 3v10A1.5 1.5 0 0 0 4 14.5h8a1.5 1.5 0 0 0 1.5-1.5V6.414a1.5 1.5 0 0 0-.44-1.06L9.647 1.94A1.5 1.5 0 0 0 8.586 1.5H4zm5 1.414L12.086 6H9.5a.5.5 0 0 1-.5-.5V2.914zM3.5 3a.5.5 0 0 1 .5-.5h4v3A1.5 1.5 0 0 0 9.5 7h3v6a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V3z"/></svg>`;
      const pathElem = document.createElement('span');
      pathElem.className = 'copilot-file-path';
      pathElem.textContent = fileEdit.path || 'file'; pathElem.title = fileEdit.path || 'file';
      const fileStatus = (fileEdit.status || '').toLowerCase();
      const statusBadge = document.createElement('span');
      statusBadge.className = `copilot-file-badge ${fileStatus === 'created' ? 'created' : fileStatus === 'deleted' ? 'deleted' : 'modified'}`;
      statusBadge.textContent = fileStatus === 'created' ? 'Created' : fileStatus === 'deleted' ? 'Deleted' : 'Modified';
      const stats = document.createElement('span');
      stats.className = 'copilot-file-stats';
      if (fileEdit.additions > 0) stats.append(Object.assign(document.createElement('span'), { className: 'stat-add', textContent: `+${fileEdit.additions}` }));
      if (fileEdit.deletions > 0) stats.append(Object.assign(document.createElement('span'), { className: 'stat-del', textContent: `-${fileEdit.deletions}` }));
      const toggleBtn = document.createElement('button');
      toggleBtn.type = 'button'; toggleBtn.className = 'copilot-file-toggle'; toggleBtn.textContent = 'Show diff';
      toggleBtn.setAttribute('aria-expanded', 'false');
      const maximizeBtn = document.createElement('button');
      maximizeBtn.type = 'button'; maximizeBtn.className = 'copilot-file-maximize';
      maximizeBtn.innerHTML = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M6 2H2v4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg>';
      header.append(icon, pathElem, statusBadge, stats, toggleBtn, maximizeBtn);
      const body = document.createElement('div');
      body.className = 'copilot-file-body collapsed';
      const diffPre = document.createElement('pre');
      diffPre.className = 'copilot-file-diff';
      if (fileEdit.diff && fileEdit.diff.trim()) {
        renderDiffLines(diffPre, fileEdit.diff);
      } else {
        const notice = document.createElement('div');
        notice.className = 'copilot-diff-line ctx';
        notice.textContent = fileEdit.additions === 0 && fileEdit.deletions === 0
          ? 'No net changes.' : '(no diff available)';
        diffPre.append(notice);
      }
      body.append(diffPre);
      card.append(header, body);
      maximizeBtn.addEventListener('click', () => {
        $('file-diff-title').textContent = fileEdit.path;
        $('file-diff-dialog').dataset.path = fileEdit.path;
        $('file-diff-view').replaceChildren(diffPre.cloneNode(true));
        $('file-diff-dialog').showModal();
      });
      toggleBtn.addEventListener('click', () => {
        const isCollapsed = body.classList.toggle('collapsed');
        toggleBtn.textContent = isCollapsed ? 'Show diff' : 'Hide diff';
        toggleBtn.setAttribute('aria-expanded', String(!isCollapsed));
      });
      return card;
    }

    function showFileEdit(fileEdit) {
      try {
        if (!fileEdit || typeof fileEdit.path !== 'string' || typeof fileEdit.content !== 'string' ||
            typeof fileEdit.status !== 'string' || typeof fileEdit.diff !== 'string') {
          throw new Error('File edit is missing required fields.');
        }
        const previous = chatFileChanges.get(fileEdit.path);
        const original = previous ? previous.original : originalFileLines(fileEdit);
        const cumulative = { ...fileEdit, status: previous?.status ?? fileEdit.status,
          ...inlineFileDiff(original, fileLines(fileEdit.content)) };
        const card = createCopilotFileCard(cumulative);
        if (previous) {
          if (previous.card.querySelector('.copilot-file-toggle').getAttribute('aria-expanded') === 'true')
            card.querySelector('.copilot-file-toggle').click();
          previous.card.replaceWith(card);
        } else {
          fileChangeListEl.append(card);
        }
        chatFileChanges.set(fileEdit.path, { original, status: cumulative.status, card });
        if ($('file-diff-dialog').open && $('file-diff-dialog').dataset.path === fileEdit.path)
          $('file-diff-view').replaceChildren(card.querySelector('.copilot-file-diff').cloneNode(true));
        fileChangesEl.hidden = false;
        if (!previous) fileChangeListEl.scrollTop = fileChangeListEl.scrollHeight;
      } catch (error) {
        console.error('Cannot render cumulative file diff:', error);
        const notice = document.createElement('div');
        notice.className = 'hint'; notice.setAttribute('role', 'alert');
        notice.textContent = `Cannot show diff for ${fileEdit?.path || 'file'}: ${error.message}`;
        fileChangeListEl.append(notice);
        fileChangesEl.hidden = false;
      }
    }

    // ── Approval rendering ─────────────────────────────────────────────────────
    function findApprovalRecord(approvalId) {
      for (const item of history) {
        if (!Array.isArray(item.approvalRequests)) continue;
        const approval = item.approvalRequests.find(e => e && e.id === approvalId);
        if (approval) return { item, approval };
      }
      return null;
    }

    function setApprovalStatus(approvalId, nextStatus) {
      const record = findApprovalRecord(approvalId);
      if (record) record.approval.status = nextStatus;
    }

    function updateApprovalCardState(approvalId, nextStatus, noteText) {
      for (const card of messagesEl.querySelectorAll(`.approval-card[data-approval-id="${approvalId}"]`)) {
        card.className = `approval-card ${nextStatus}`;
        const badge = card.querySelector('.approval-card-badge');
        if (badge) { badge.className = `approval-card-badge ${nextStatus}`; badge.textContent = nextStatus; }
        const note = card.querySelector('.approval-status-note');
        if (note) note.textContent = noteText;
        for (const btn of card.querySelectorAll('.approval-action')) btn.disabled = true;
      }
    }

    function pendingApprovals() {
      const approvals = [];
      for (const item of history) {
        if (!Array.isArray(item.approvalRequests)) continue;
        for (const approval of item.approvalRequests)
          if (approval?.status === 'pending') approvals.push(approval);
      }
      return approvals;
    }

    function isApprovalIntent(text) {
      return /^(?:yes|yes do it|do it|apply(?: the)? fix|apply(?: it)?|go ahead|proceed|continue|ok|okay|sure|confirm|approve(?: it| this)?|approved)$/i.test(text.trim());
    }
    function isRejectionIntent(text) {
      return /^(?:no|reject|reject it|reject this|cancel|don't apply|do not apply|skip)$/i.test(text.trim());
    }

    function renderApprovalRequest(approval, article) {
      if (!approval || typeof approval.id !== 'string' || typeof approval.path !== 'string') return;
      const card = document.createElement('section');
      card.className = `approval-card ${approval.status || 'pending'}`;
      card.dataset.approvalId = approval.id;
      const header = document.createElement('div');
      header.className = 'approval-card-header';
      const textWrap = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'approval-card-title';
      title.textContent = approval.kind === 'delete' ? 'Delete requires approval' : 'Change requires approval';
      const pathLine = document.createElement('div');
      pathLine.className = 'approval-card-path'; pathLine.textContent = approval.path;
      textWrap.append(title, pathLine);
      const badge = document.createElement('span');
      badge.className = `approval-card-badge ${approval.status || 'pending'}`;
      badge.textContent = approval.status || 'pending';
      header.append(textWrap, badge);
      const body = document.createElement('div');
      body.className = 'approval-card-body';
      const rationale = document.createElement('p');
      rationale.textContent = approval.rationale || approval.message || 'Review this change before applying it.';
      body.append(rationale);
      if (approval.file_edit?.diff) {
        const diff = document.createElement('pre');
        diff.className = 'approval-diff';
        renderDiffLines(diff, approval.file_edit.diff);
        body.append(diff);
      }
      const note = document.createElement('div');
      note.className = 'approval-status-note';
      note.textContent = approval.status === 'approved' ? 'Approved and applied.'
        : approval.status === 'rejected' ? 'Rejected. No file changes were applied.'
        : 'No file changes will happen until you approve.';
      body.append(note);
      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'approval-actions';
      const approveBtn = document.createElement('button');
      approveBtn.type = 'button'; approveBtn.className = 'approval-action approval-approve';
      approveBtn.textContent = approval.kind === 'delete' ? 'Approve delete' : 'Approve change';
      const rejectBtn = document.createElement('button');
      rejectBtn.type = 'button'; rejectBtn.className = 'approval-action approval-reject';
      rejectBtn.textContent = 'Reject';
      actionsDiv.append(approveBtn, rejectBtn);
      body.append(actionsDiv);

      function syncState(nextStatus, text) {
        approval.status = nextStatus;
        updateApprovalCardState(approval.id, nextStatus, text);
      }
      if (approval.status === 'approved') syncState('approved', 'Approved and applied.');
      else if (approval.status === 'rejected') syncState('rejected', 'Rejected. No file changes were applied.');

      approveBtn.addEventListener('click', async () => {
        if (approvalBusy || approval.status !== 'pending') return;
        approvalBusy = true; controls();
        status(approval.kind === 'delete' ? 'Deleting file...' : 'Applying approved change...');
        approveBtn.disabled = true; rejectBtn.disabled = true;
        try {
          const response = await chatRequest('/api/assistant/approval', {
            method: 'POST', body: JSON.stringify({ approval })
          });
          syncState('approved', 'Approved and applied.'); setApprovalStatus(approval.id, 'approved');
          const appliedText = (typeof response.message === 'string' && response.message.trim())
            ? response.message.trim() : (approval.kind === 'delete' ? 'Approved deletion applied.' : 'Approved change applied.');
          const chatModel = $('model').value;
          const assistantMessage = { role: 'assistant', content: appliedText, fileEdits: response.fileEdit ? [response.fileEdit] : [], model: chatModel };
          const assistant = addMessage('assistant', appliedText, assistantMessage.fileEdits, [], chatModel);
          history.push(assistantMessage);
          historyItemByArticle.set(assistant.article, assistantMessage);
          await saveConversationSafely('Approved change applied.');
        } catch (error) {
          approval.status = 'pending'; approveBtn.disabled = false; rejectBtn.disabled = false;
          status(`Could not apply approved change: ${error.message}`, true);
        } finally { approvalBusy = false; controls(); }
      });

      rejectBtn.addEventListener('click', async () => {
        if (approvalBusy || approval.status !== 'pending') return;
        syncState('rejected', 'Rejected. No file changes were applied.'); setApprovalStatus(approval.id, 'rejected');
        const rejectedText = approval.kind === 'delete' ? 'Deletion rejected. No files were removed.' : 'Change rejected. No files were modified.';
        const chatModel = $('model').value;
        const assistantMessage = { role: 'assistant', content: rejectedText, model: chatModel };
        const assistant = addMessage('assistant', rejectedText, [], [], chatModel);
        history.push(assistantMessage);
        historyItemByArticle.set(assistant.article, assistantMessage);
        await saveConversationSafely('Change rejected.');
      });

      card.append(header, body);
      article.append(card);
    }

    // ── addMessage — renders a message article in this pane ───────────────────
    function addMessage(role, text, fileEdits = [], approvalRequests = [], model = null) {
      // Remove empty state if present
      messagesEl.querySelector(`#empty-${tabId}`)?.remove();
      const article = document.createElement('article');
      article.className = `message ${role}`;
      const heading = document.createElement('div');
      heading.className = 'role';
      heading.textContent = role === 'user' ? 'You' : 'Assistant';
      const content = document.createElement('div');
      content.className = 'content';
      setMessageContent(content, text, role === 'assistant');
      if (role === 'assistant') {
        const modelIndicator = document.createElement('div');
        modelIndicator.className = 'model-indicator';
        const indicatorModel = (model ?? $('model').value) || '';
        modelIndicator.textContent = indicatorModel ? modelLabel(indicatorModel) : '—';
        content.append(modelIndicator);
      }
      article.append(heading, content);

      // Action bar
      const actionsBar = document.createElement('div');
      actionsBar.className = 'msg-actions';

      // Checkpoint button
      const checkpointBtn = document.createElement('button');
      checkpointBtn.type = 'button'; checkpointBtn.className = 'msg-action-btn';
      checkpointBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 1.5A5.5 5.5 0 1 1 8 13.5 5.5 5.5 0 0 1 8 2.5zM8 4a.75.75 0 0 1 .75.75v3.5l2 1.15a.75.75 0 0 1-.75 1.3l-2.25-1.3A.75.75 0 0 1 7.25 8.5v-3.75A.75.75 0 0 1 8 4z"/></svg> Checkpoint`;
      checkpointBtn.title = 'Save a checkpoint here';
      checkpointBtn.addEventListener('click', () => {
        if (busy || approvalBusy) return;
        const idx = historyIndexForArticle(article);
        if (idx < 0) { status('Cannot checkpoint this message.', true); return; }
        const label = (history[idx].display || history[idx].content || '').slice(0, 60) || `Turn ${idx + 1}`;
        saveCheckpoint(idx, label);
      });
      actionsBar.append(checkpointBtn);

      // Edit button (user only)
      if (role === 'user') {
        const editBtn = document.createElement('button');
        editBtn.type = 'button'; editBtn.className = 'msg-action-btn';
        editBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M11.013 1.427a1.75 1.75 0 0 1 2.474 0l1.086 1.086a1.75 1.75 0 0 1 0 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 0 1-.927-.928l.929-3.25a1.75 1.75 0 0 1 .445-.757l8.61-8.61zm1.414 1.06a.25.25 0 0 0-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 0 0 0-.354l-1.086-1.086zM11.189 6.25 9.75 4.81 4.198 10.363a.25.25 0 0 0-.064.108l-.618 2.162 2.162-.617a.25.25 0 0 0 .108-.064L11.19 6.25z"/></svg> Edit`;
        editBtn.title = 'Edit this prompt and resend from here';
        const editWrap = document.createElement('div');
        editWrap.className = 'edit-prompt-wrap';
        const editTa = document.createElement('textarea');
        editTa.className = 'edit-prompt-ta'; editTa.rows = 3;
        const editActions = document.createElement('div');
        editActions.className = 'edit-prompt-actions';
        const resendBtn = document.createElement('button');
        resendBtn.type = 'button'; resendBtn.className = 'edit-resend-btn'; resendBtn.textContent = 'Resend ↑';
        const cancelBtn = document.createElement('button');
        cancelBtn.type = 'button'; cancelBtn.className = 'edit-cancel-btn'; cancelBtn.textContent = 'Cancel';
        editActions.append(resendBtn, cancelBtn);
        editWrap.append(editTa, editActions);
        editBtn.addEventListener('click', () => {
          if (busy || approvalBusy) return;
          const idx = historyIndexForArticle(article);
          if (idx < 0) { status('Cannot edit this message.', true); return; }
          editTa.value = history[idx].display ?? history[idx].content ?? '';
          editWrap.classList.add('open');
          editTa.focus();
          editTa.selectionStart = editTa.selectionEnd = editTa.value.length;
        });
        cancelBtn.addEventListener('click', () => editWrap.classList.remove('open'));
        editTa.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); resendBtn.click(); }
          if (e.key === 'Escape') cancelBtn.click();
        });
        resendBtn.addEventListener('click', async () => {
          if (busy || approvalBusy || readingFiles || !connected) return;
          const newPrompt = editTa.value.trim();
          if (!newPrompt) { status('Prompt cannot be empty.', true); return; }
          const idx = historyIndexForArticle(article);
          if (idx < 0) { status('Cannot resend: message not found.', true); return; }
          if (!checkpoint) {
            const currentLabel = (history[history.length - 1]?.display || history[history.length - 1]?.content || '').slice(0, 60) || 'before edit';
            saveCheckpoint(history.length - 1, currentLabel, true);
          }
          history.splice(idx);
          const allArticles = Array.from(messagesEl.querySelectorAll('article.message'));
          for (let i = idx; i < allArticles.length; i++) allArticles[i].remove();
          editWrap.classList.remove('open');
          promptEl.value = newPrompt;
          composerEl.requestSubmit();
        });
        actionsBar.append(editBtn);
        article.append(actionsBar, editWrap);
      } else {
        article.append(actionsBar);
      }

      if (Array.isArray(fileEdits) && fileEdits.length) for (const edit of fileEdits) showFileEdit(edit);
      if (Array.isArray(approvalRequests) && approvalRequests.length) for (const approval of approvalRequests) renderApprovalRequest(approval, article);

      messagesEl.append(article);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return { article, content };
    }

    // Copy code handler (per-pane)
    messagesEl.addEventListener('click', async (event) => {
      const copy = event.target.closest('.copy-code');
      if (copy && !copy.disabled) {
        try {
          if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(copy.dataset.copy || '');
          else {
            const inp = document.createElement('textarea');
            inp.value = copy.dataset.copy || ''; inp.setAttribute('readonly', '');
            inp.style.cssText = 'position:absolute;left:-9999px'; document.body.append(inp);
            inp.select(); document.execCommand('copy'); inp.remove();
          }
          copy.dataset.copied = 'true'; copy.textContent = 'Copied';
          setTimeout(() => { copy.dataset.copied = 'false'; copy.textContent = 'Copy'; }, 1600);
        } catch (error) { status(`Could not copy code: ${error.message}`, true); }
      }
      // Suggestion buttons
      const suggBtn = event.target.closest('button[data-prompt]');
      if (suggBtn && !busy && !readingFiles) {
        promptEl.value = suggBtn.dataset.prompt; controls(); promptEl.focus();
      }
    });

    // ── Checkpoint ─────────────────────────────────────────────────────────────
    function removeCheckpointBanner() {
      if (checkpoint?.bannerEl) { checkpoint.bannerEl.remove(); checkpoint.bannerEl = null; }
    }

    function saveCheckpoint(historyIndex, label, silent = false) {
      removeCheckpointBanner();
      const snap = history.slice(0, historyIndex + 1).map(item => ({ ...item }));
      if (silent) { checkpoint = { historySnapshot: snap, label, bannerEl: null, pendingBanner: true }; return; }
      const banner = document.createElement('div');
      banner.className = 'checkpoint-banner';
      banner.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 1.5A5.5 5.5 0 1 1 8 13.5 5.5 5.5 0 0 1 8 2.5zM8 4a.75.75 0 0 1 .75.75v3.5l2 1.15a.75.75 0 0 1-.75 1.3l-2.25-1.3A.75.75 0 0 1 7.25 8.5v-3.75A.75.75 0 0 1 8 4z"/></svg><span class="checkpoint-banner-text">Checkpoint: <strong>${label}</strong></span><button type="button" class="checkpoint-restore-btn">Restore</button><button type="button" class="checkpoint-discard-btn">Discard</button>`;
      banner.querySelector('.checkpoint-restore-btn').addEventListener('click', () => { if (!busy && !approvalBusy) restoreCheckpoint(); });
      banner.querySelector('.checkpoint-discard-btn').addEventListener('click', () => { checkpoint = null; banner.remove(); status('Checkpoint discarded.'); });
      composerEl.insertAdjacentElement('beforebegin', banner);
      checkpoint = { historySnapshot: snap, label, bannerEl: banner, pendingBanner: false };
      status(`Checkpoint saved at "${label}".`);
    }

    function restoreCheckpoint() {
      if (!checkpoint) return;
      const snap = checkpoint.historySnapshot;
      const label = checkpoint.label;
      $('file-diff-dialog').close();
      history.length = 0;
      history.push(...snap.map(item => ({ ...item })));
      messagesEl.replaceChildren();
      fileChangeListEl.replaceChildren();
      chatFileChanges.clear();
      fileChangesEl.hidden = true;
      for (const item of history) {
        const rendered = addMessage(item.role, item.display ?? item.content, item.fileEdits, item.approvalRequests);
        historyItemByArticle.set(rendered.article, item);
        const chips = buildAttachedChips(item);
        if (chips) rendered.article.append(chips);
      }
      checkpoint = null;
      status(`Conversation restored to checkpoint: "${label}".`);
      saveConversationSafely('Checkpoint restored.');
    }

    // ── Conversation persistence ───────────────────────────────────────────────
    async function saveConversation() {
      if (!storageLoaded) throw new Error('SQLite chat history is not available.');
      const id = activeChatId || crypto.randomUUID();
      const chat = {
        id, title: (history[0].display || history[0].content).slice(0, 80),
        model: $('model').value, updatedAt: new Date().toISOString(),
        messages: history.map(item => ({ ...item, model: item.model ?? $('model').value }))
      };
      await chatRequest(`/api/chats/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(chat) });
      activeChatId = id;
      savedChats = [...savedChats.filter(item => item.id !== id), chat];
      showHistoryError('');
      renderChats();
    }

    async function saveConversationSafely(successMessage) {
      try {
        await chatHistoryReady;
        await saveConversation();
        if (successMessage) status(successMessage);
      } catch (error) {
        showHistoryError(`Chat history was not saved: ${error.message}`);
        if (successMessage) status(`${successMessage} Chat history could not be saved.`, true);
      }
    }

    function newConversation() {
      $('file-diff-dialog').close();
      activeChatId = null;
      history.length = 0;
      checkpoint = null;
      // Remove checkpoint banner if any
      paneEl.querySelectorAll('.checkpoint-banner').forEach(b => b.remove());
      messagesEl.replaceChildren();
      // Restore empty state
      const empty = document.createElement('div');
      empty.id = `empty-${tabId}`;
      empty.innerHTML = `<h2>How can I help you today?</h2><p>Think through an idea, explore your code, or make sense of your logs.</p><div class="suggestions" aria-label="Suggested prompts"><button type="button" data-prompt="Help me investigate an error in my application logs.">Investigate an error <span>Find the cause in your logs</span></button><button type="button" data-prompt="Help me review and explain my code.">Understand code <span>Explore how something works</span></button><button type="button" data-prompt="Help me plan an implementation step by step.">Plan a solution <span>Turn an idea into next steps</span></button></div>`;
      messagesEl.append(empty);
      fileChangeListEl.replaceChildren();
      chatFileChanges.clear();
      fileChangesEl.hidden = true;
      attachments.length = 0;
      renderAttachments();
      promptEl.value = '';
      controls();
      renderChats();
    }

    function reopenChat(chat) {
      if (busy || approvalBusy || readingFiles) return;
      const model = $('model');
      const savedModelInstalled = Array.from(model.options).some(o => o.value === chat.model);
      if (savedModelInstalled) model.value = chat.model;
      else if (!model.value && model.options.length) model.value = model.options[0].value;
      newConversation();
      activeChatId = chat.id;
      updateModelPill(); updatePaneModelPill();
      history.push(...chat.messages.map(item => ({ ...item })));
      for (const item of history) {
        const rendered = addMessage(item.role, item.display ?? item.content, item.fileEdits, item.approvalRequests, item.model);
        historyItemByArticle.set(rendered.article, item);
        const chips = buildAttachedChips(item);
        if (chips) rendered.article.append(chips);
      }
      renderChats(); controls();
      if (!connected || !model.value)
        status('Saved conversation reopened. Connect to a configured provider to continue.', true);
      else if (!savedModelInstalled)
        status(`Saved conversation reopened. ${modelLabel(chat.model)} is unavailable; using ${modelLabel(model.value)} instead.`);
      else
        status('Saved conversation reopened. You can continue chatting.');
    }

    // ── File input ─────────────────────────────────────────────────────────────
    filesEl.addEventListener('change', async () => {
      const files = Array.from(filesEl.files);
      if (!files.length) return;
      readingFiles = true; controls(); status('Reading attachments locally...');
      try {
        const total = [...attachments, ...files].reduce((sum, f) => sum + f.size, 0);
        if (total > maxAttachmentBytes) throw new Error('Attachments exceed 1 MB total. Remove a file or choose smaller excerpts.');
        const additions = await Promise.all(files.map(async (file) => {
          if (file.size === 0) throw new Error(`${file.name} is empty.`);
          const bytes = await file.arrayBuffer();
          let text;
          try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
          catch (error) { throw new Error(`${file.name} is not valid UTF-8 text.`, { cause: error }); }
          if (/[\u0000-\u0008\u000e-\u001f\u007f]/.test(text) || /^(?:%PDF-|GIF8[79]a|PK\u0003\u0004)/.test(text) ||
              /^(?:image\/|audio\/|video\/|application\/pdf)/.test(file.type))
            throw new Error(`${file.name} is not a supported text file.`);
          return { name: file.name, size: file.size, text };
        }));
        attachments.push(...additions);
        renderAttachments();
        status(`Attached ${attachments.length} file(s). Full contents will be sent locally with your next message.`);
      } catch (error) {
        status(error.message, true);
      } finally {
        filesEl.value = ''; readingFiles = false; controls();
      }
    });

    // ── Submit ─────────────────────────────────────────────────────────────────
    stopBtn.addEventListener('click', () => controller?.abort());
    promptEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!busy && connected) composerEl.requestSubmit();
      }
    });

    composerEl.addEventListener('submit', async (event) => {
      event.preventDefault();
      const prompt = promptEl.value.trim();
      const pending = pendingApprovals();

      if (!attachments.length && pending.length === 1 && isApprovalIntent(prompt)) {
        promptEl.value = '';
        const approval = pending[0];
        approvalBusy = true; controls();
        status(approval.kind === 'delete' ? 'Deleting file after approval...' : 'Applying approved change...');
        const user = addMessage('user', prompt);
        const userMessage = { role: 'user', content: prompt, display: prompt };
        history.push(userMessage); historyItemByArticle.set(user.article, userMessage);
        try {
          const response = await chatRequest('/api/assistant/approval', { method: 'POST', body: JSON.stringify({ approval }) });
          setApprovalStatus(approval.id, 'approved'); updateApprovalCardState(approval.id, 'approved', 'Approved and applied.');
          const appliedText = (typeof response.message === 'string' && response.message.trim()) ? response.message.trim() : 'Approved change applied.';
          const chatModel = $('model').value;
          const assistantMessage = { role: 'assistant', content: appliedText, fileEdits: response.fileEdit ? [response.fileEdit] : [], model: chatModel };
          const assistant = addMessage('assistant', appliedText, assistantMessage.fileEdits, [], chatModel);
          history.push(assistantMessage); historyItemByArticle.set(assistant.article, assistantMessage);
          await saveConversationSafely('Approved change applied.');
        } catch (error) { history.pop(); user.article.remove(); status(`Could not apply approved change: ${error.message}`, true); }
        finally { approvalBusy = false; controls(); promptEl.focus(); }
        return;
      }

      if (!attachments.length && pending.length === 1 && isRejectionIntent(prompt)) {
        promptEl.value = '';
        controls();
        const approval = pending[0];
        setApprovalStatus(approval.id, 'rejected'); updateApprovalCardState(approval.id, 'rejected', 'Rejected. No file changes were applied.');
        const user = addMessage('user', prompt);
        const userMessage = { role: 'user', content: prompt, display: prompt };
        history.push(userMessage); historyItemByArticle.set(user.article, userMessage);
        const rejectedText = approval.kind === 'delete' ? 'Deletion rejected. No files were removed.' : 'Change rejected. No files were modified.';
        const chatModel = $('model').value;
        const assistantMessage = { role: 'assistant', content: rejectedText, model: chatModel };
        const assistant = addMessage('assistant', rejectedText, [], [], chatModel);
        history.push(assistantMessage); historyItemByArticle.set(assistant.article, assistantMessage);
        await saveConversationSafely('Change rejected.'); promptEl.focus();
        return;
      }

      if (busy || approvalBusy || readingFiles || !connected) return;
      if (!prompt && !attachments.length) { status('Type a message or attach a file before sending.', true); return; }
      if (attachments.reduce((sum, f) => sum + f.size, 0) > largeAttachmentWarningBytes &&
          !window.confirm('These attachments are large. Send anyway?')) return;

      const sentFiles = [...attachments];
      const hasAttachments = sentFiles.length > 0 || history.some(item => item.files?.length > 0);
      const mode = hasAttachments ? $('attachment-mode').value : 'agent';
      const generatingStatus = mode === 'attachment'
        ? 'Reading attachments with streaming LangGraph...'
        : 'Working with the selected Ollama model + code assistant...';
      const instruction = prompt || 'Analyze the attached files.';
      const content = sentFiles.length
        ? `${instruction}\n\n---ATTACHED FILES START---\nThe following file(s) were attached by the user. Read their contents carefully and use them to answer the request above.\n\n${sentFiles.map(f => `File: ${f.name}\n---\n${f.text}\n---`).join('\n\n')}\n---ATTACHED FILES END---`
        : instruction;

      busy = true; controller = new AbortController(); controls();
      const user = addMessage('user', instruction);
      if (sentFiles.length) {
        const names = document.createElement('div');
        names.className = 'hint'; names.style.cssText = 'display:flex;flex-wrap:wrap;align-items:center;gap:6px';
        names.append(Object.assign(document.createElement('span'), { textContent: 'Attached:' }));
        for (const file of sentFiles) {
          const chip = document.createElement('button');
          chip.type = 'button'; chip.className = 'attachment-chip-preview'; chip.style.fontSize = '11px';
          chip.innerHTML = `<svg viewBox="0 0 16 16" width="11" height="11" fill="currentColor"><path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/><path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8zm8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/></svg> ${file.name}`;
          chip.addEventListener('click', () => openAttachmentPreviewGlobal(file));
          names.append(chip);
        }
        user.article.append(names);
      }
      const chatModel = $('model').value;
      const reply = addMessage('assistant', 'Waiting for assistant...', [], [], chatModel);
      promptEl.value = ''; status(generatingStatus);
      let answer = '', thinking = '', thinkingContent, completed = false;
      const fileEdits = [], approvalRequests = [];
      const userMessage = { role: 'user', content, display: instruction, files: sentFiles.map(f => f.name), fileContents: sentFiles.map(f => ({ name: f.name, size: f.size, text: f.text })) };

      // Update tab label with first message
      updateTabLabel(tabId, instruction.slice(0, 40));

      try {
        const response = await fetch('/api/assistant/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            endpoint: endpointForModel(chatModel), model: chatModel, mode,
            messages: [...history, userMessage].map(({ role, content }) => ({ role, content })),
            stream: true
          }),
          signal: controller.signal
        });
        await readStream(response, (chunk) => {
          if (chunk.file_edit) { fileEdits.push(chunk.file_edit); showFileEdit(chunk.file_edit); }
          if (chunk.approval_request) { approvalRequests.push(chunk.approval_request); renderApprovalRequest(chunk.approval_request, reply.article); }
          if (chunk.message?.thinking) {
            thinking += chunk.message.thinking;
            if (!thinkingContent) {
              const details = document.createElement('details');
              const summary = document.createElement('summary');
              summary.textContent = 'Model reasoning';
              thinkingContent = document.createElement('div');
              thinkingContent.className = 'content plain';
              details.append(summary, thinkingContent);
              reply.article.insertBefore(details, reply.content);
            }
            setMessageContent(thinkingContent, thinking);
          }
          if (typeof chunk.message?.content === 'string') {
            const follow = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 100;
            answer += chunk.message.content;
            if (answer) { setMessageContent(reply.content, answer, true); status(generatingStatus); }
            if (follow) messagesEl.scrollTop = messagesEl.scrollHeight;
          }
          if (chunk.done === true) completed = true;
        });
        if (!completed) throw new Error('Ollama stream ended before the reply completed.');
        if (!answer.trim() && !fileEdits.length && !approvalRequests.length) throw new Error('The model returned no answer.');
        if (!answer.trim() && fileEdits.length) answer = 'Files updated successfully.';
        if (!answer.trim() && approvalRequests.length) answer = 'Approval required before changing files.';
        const assistantMessage = { role: 'assistant', content: answer, fileEdits, approvalRequests, model: chatModel };
        history.push(userMessage, assistantMessage);
        historyItemByArticle.set(user.article, userMessage);
        historyItemByArticle.set(reply.article, assistantMessage);
        attachments.length = 0; renderAttachments();
        try {
          await chatHistoryReady; await saveConversation(); status('Reply complete.');
        } catch (error) {
          showHistoryError(`Chat history was not saved: ${error.message}`);
          status('Reply complete, but chat history could not be saved.', true);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          if (!answer) setMessageContent(reply.content, 'Generation stopped.');
          status('Stopped. This incomplete exchange was not added to model history.');
        } else {
          const errorMessage = chatErrorMessage(error);
          if (!answer) setMessageContent(reply.content, errorMessage);
          else {
            const errorNotice = document.createElement('p');
            errorNotice.textContent = `Reply incomplete. ${errorMessage}`;
            errorNotice.setAttribute('role', 'alert');
            reply.article.append(errorNotice);
          }
          status(`${errorMessage} This exchange was not added to model history.`, true);
        }
      } finally {
        if (!completed || !answer.trim()) promptEl.value = prompt;
        busy = false; controller = null; controls(); promptEl.focus();
      }
    });

    // ── Expose pane interface ──────────────────────────────────────────────────
    return {
      el: paneEl,
      get busy() { return busy; },
      get approvalBusy() { return approvalBusy; },
      get readingFiles() { return readingFiles; },
      get activeChatId() { return activeChatId; },
      newConversation,
      reopenChat,
      refreshControls: controls,
      refreshModelPill: updatePaneModelPill,
      focusPrompt: () => promptEl.focus(),
      onActivated() {
        updatePaneModelPill();
        controls();
      }
    };
  } // end createChatPane

  // ── Attachment preview (global dialog) ────────────────────────────────────────
  function openAttachmentPreviewGlobal(attachment) {
    $('attachment-preview-title').textContent = attachment.name;
    $('attachment-preview-meta').textContent =
      `${attachment.size.toLocaleString()} bytes · ${attachment.text.split('\n').length} lines`;
    const body = $('attachment-preview-body');
    body.replaceChildren();
    const lines = attachment.text.split('\n');
    const wrap = document.createElement('div');
    wrap.className = 'attachment-preview-lines';
    const lineNums = document.createElement('div');
    lineNums.className = 'attachment-preview-line-numbers';
    lineNums.setAttribute('aria-hidden', 'true');
    lineNums.textContent = lines.map((_, i) => i + 1).join('\n');
    const lineContent = document.createElement('div');
    lineContent.className = 'attachment-preview-line-content';
    lineContent.textContent = attachment.text;
    wrap.append(lineNums, lineContent);
    body.append(wrap);
    $('attachment-preview-dialog').showModal();
  }
  $('attachment-preview-close').addEventListener('click', () => $('attachment-preview-dialog').close());
  $('attachment-preview-dialog').addEventListener('close', () => $('attachment-preview-body').replaceChildren());
  $('file-diff-close').addEventListener('click', () => $('file-diff-dialog').close());

  // ── Tab management ────────────────────────────────────────────────────────────
  function updateTabLabel(tabId, label) {
    const tab = tabs.find(t => t.id === tabId);
    if (!tab) return;
    const nameEl = tab.tabEl.querySelector('.tab-name');
    if (nameEl) nameEl.textContent = label || `Chat ${tab.num}`;
  }

  function activateTab(tabId) {
    activeTabId = tabId;
    for (const tab of tabs) {
      const isActive = tab.id === tabId;
      tab.tabEl.classList.toggle('active', isActive);
      tab.tabEl.setAttribute('aria-selected', String(isActive));
      tab.pane.el.style.display = isActive ? '' : 'none';
    }
    const pane = getActivePane();
    if (pane) pane.onActivated();
  }

  function closeTab(tabId) {
    const idx = tabs.findIndex(t => t.id === tabId);
    if (idx < 0) return;
    const tab = tabs[idx];
    if (tab.pane.busy && !window.confirm('This tab is still generating. Close it anyway? The current response will be cancelled.')) return;
    tab.tabEl.remove();
    tab.pane.el.remove();
    tabs.splice(idx, 1);
    if (tabs.length === 0) {
      addTab();
    } else {
      const nextTab = tabs[Math.min(idx, tabs.length - 1)];
      activateTab(nextTab.id);
    }
  }

  function addTab(chatToOpen = null) {
    const tabId = `tab-${nextTabNum}`;
    const tabNum = nextTabNum++;
    const pane = createChatPane(tabId);
    $('tab-panes').append(pane.el);

    // Build tab button
    const tabEl = document.createElement('button');
    tabEl.type = 'button';
    tabEl.className = 'tab-btn';
    tabEl.role = 'tab';
    tabEl.dataset.tabId = tabId;
    tabEl.setAttribute('aria-controls', tabId);
    tabEl.setAttribute('aria-selected', 'false');
    tabEl.num = tabNum;

    const nameEl = document.createElement('span');
    nameEl.className = 'tab-name';
    nameEl.textContent = `Chat ${tabNum}`;

    const busyDot = document.createElement('span');
    busyDot.className = 'tab-busy-dot';
    busyDot.setAttribute('aria-hidden', 'true');

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'tab-close';
    closeBtn.setAttribute('aria-label', `Close Chat ${tabNum}`);
    closeBtn.title = 'Close tab';
    closeBtn.innerHTML = `<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06z"/></svg>`;
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      closeTab(tabId);
    });

    tabEl.append(nameEl, busyDot, closeBtn);
    tabEl.addEventListener('click', () => activateTab(tabId));

    $('tab-list').append(tabEl);
    tabs.push({ id: tabId, num: tabNum, pane, tabEl });

    activateTab(tabId);
    if (chatToOpen) pane.reopenChat(chatToOpen);
    else pane.focusPrompt();
    return pane;
  }

  // ── "New tab" button and sidebar "New chat" button ────────────────────────────
  $('new-tab-btn').addEventListener('click', () => addTab());
  $('clear').addEventListener('click', () => {
    getActivePane()?.newConversation();
    status(connected ? 'New conversation. Ready.' : 'Click Connect / refresh to reconnect.');
    getActivePane()?.focusPrompt();
  });

  // ── Global settings controls ─────────────────────────────────────────────────
  $('connect').addEventListener('click', connect);
  $('provider').value = selectedProvider;
  $('provider').addEventListener('change', () => {
    selectedProvider = $('provider').value;
    localStorage.setItem('assistant-provider-v1', selectedProvider);
    applyProviderModels('');
  });
  $('endpoint').addEventListener('input', () => {
    connectionAttempt++;
    connected = false;
    syncGlobalControls();
    for (const tab of tabs) tab.pane.refreshControls();
    status('Server changed. Click Connect / refresh.');
  });
  $('model').addEventListener('change', () => {
    updateModelPill();
    for (const t of tabs) t.pane.refreshModelPill();
  });
  $('update').addEventListener('click', async () => {
    const pane = getActivePane();
    if (pane?.busy || !connected || !$('model').value) return;
    const model = $('model').value;
    if (!window.confirm(`Download the latest ${model}? This may download several GB.`)) return;
    const origBusy = pane?.busy;
    connected = false; syncGlobalControls(); status(`Updating ${model}...`);
    for (const tab of tabs) tab.pane.refreshControls();
    let succeeded = false;
    const ctrl = new AbortController();
    try {
      const endpoint = baseUrl();
      const response = await fetch('/api/ollama/pull', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpoint, model, stream: true }), signal: ctrl.signal
      });
      await readStream(response, (chunk) => {
        if (typeof chunk.status !== 'string') throw new Error('Invalid model update response.');
        const progress = Number.isFinite(chunk.total) && chunk.total > 0 && Number.isFinite(chunk.completed)
          ? ` (${Math.min(100, Math.floor(chunk.completed / chunk.total * 100))}%)` : '';
        status(`Updating ${model}: ${chunk.status}${progress}`);
        if (chunk.status === 'success') succeeded = true;
      });
      if (!succeeded) throw new Error('Ollama stream ended before model update completed.');
      getActivePane()?.newConversation();
      await connect();
      if (connected) status(`${model} updated successfully.`);
    } catch (error) {
      status(ctrl.signal.aborted ? `Update cancelled for ${model}.` : `Model update failed: ${error.message}`, !ctrl.signal.aborted);
    } finally {
      syncGlobalControls();
    }
  });

  // ── Connect ────────────────────────────────────────────────────────────────────
  async function connect() {
    const attempt = ++connectionAttempt;
    connected = false;
    installedModels = [];
    platformAiModels = [];
    platformAiError = '';
    ollamaError = '';
    syncGlobalControls();
    for (const tab of tabs) tab.pane.refreshControls();
    status('Connecting to Ollama and PlatformAI...');
    try {
      let platformModels = [];
      let platformError = '';
      try {
        const response = await fetch('/api/platformai/models', { signal: AbortSignal.timeout(10000) });
        if (response.status === 404) {
          throw new Error(
            'The running server does not support PlatformAI. Restart the Node server from this project, then click Connect / refresh.'
          );
        }
        if (!response.ok) throw new Error(`PlatformAI model configuration returned HTTP ${response.status}.`);
        const data = await response.json();
        if (!Array.isArray(data.models) || data.models.some(item =>
          typeof item.id !== 'string' || typeof item.name !== 'string')) {
          throw new Error('PlatformAI returned an invalid model list.');
        }
        platformModels = data.models;
      } catch (error) {
        platformError = error.message;
      }

      let ollamaModels = [];
      let ollamaConnectionError = '';
      try {
        const endpoint = baseUrl();
        const response = await fetch(`/api/ollama/tags?endpoint=${encodeURIComponent(endpoint)}`, { signal: AbortSignal.timeout(10000) });
        if (!response.ok) throw new Error(`Ollama returned HTTP ${response.status}.`);
        const data = await response.json();
        if (!Array.isArray(data.models) || data.models.some(item => typeof item.name !== 'string')) {
          throw new Error('Ollama returned an invalid model list.');
        }
        ollamaModels = data.models;
      } catch (error) {
        ollamaConnectionError = error.message;
      }

      if (attempt !== connectionAttempt) return;
      installedModels = ollamaModels;
      platformAiModels = platformModels;
      platformAiError = platformError;
      ollamaError = ollamaConnectionError;
      applyProviderModels(
        selectedProvider === 'ollama'
          ? (smallestInstalledModel(ollamaModels)?.name ?? '')
          : ''
      );
    } catch (error) {
      if (attempt !== connectionAttempt) return;
      status(`${error.message} Check Ollama or configure PlatformAI in the server's .env file.`, true);
    }
    syncGlobalControls();
    for (const t of tabs) t.pane.onActivated();
  }

  // ── Init ────────────────────────────────────────────────────────────────────────
  // Dialogs
  $('attachment-preview-dialog').addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); $('attachment-preview-dialog').close(); }
  });
  $('file-diff-dialog').addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); $('file-diff-dialog').close(); }
  });

  const chatHistoryReady = loadChats();
  const connectionReady = connect();

  // Create the first tab
  addTab();

  // Auto-load chat from URL (?chatId=...)
  (function autoLoadChatFromUrl() {
    const params = new URLSearchParams(location.search);
    const chatId = params.get('chatId');
    if (!chatId) return;
    window.history.replaceState(null, '', location.pathname);
    Promise.all([chatHistoryReady, connectionReady]).then(() => {
      const chat = savedChats.find(c => c.id === chatId);
      if (!chat) { status(`Chat "${chatId}" not found.`, true); return; }
      getActivePane()?.reopenChat(chat);
    });
  })();

}); // end DOMContentLoaded
