'use strict';

const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const start = html.indexOf('    function fileLines(');
const end = html.indexOf('    function renderDiffLines(', start);
assert.ok(start > 0 && end > start);
const context = vm.createContext({});
vm.runInContext(html.slice(start, end), context);
const lines = text => Array.from(context.fileLines(text));
const diff = (before, after) => context.inlineFileDiff(lines(before), lines(after));
const original = edit => Array.from(context.originalFileLines(edit));

test('UI script parses successfully', () => {
  const script = html.match(/<script>([\s\S]*?)<\/script>/);
  assert.ok(script);
  assert.doesNotThrow(() => new vm.Script(script[1]));
});

test('created file cumulative diff only contains latest content', () => {
  const baseline = original({ status: 'created', content: 'first\n', diff: '' });
  const result = context.inlineFileDiff(baseline, lines('final\nsecond\n'));
  assert.equal(result.additions, 2);
  assert.equal(result.deletions, 0);
  assert.ok(result.diff.includes('+final'));
  assert.ok(!result.diff.includes('first'));
});

test('reconstructs modified file from agent diff with retained newlines', () => {
  const edit = {
    status: 'modified',
    content: 'keep\nnew\n\nlast',
    diff: '--- a/file\n+++ b/file\n@@ -1,4 +1,4 @@\n keep\n\n-old\n\n+new\n\n \n\n last'
  };
  assert.deepEqual(original(edit), lines('keep\nold\n\nlast'));
});

test('reconstructs omitted context across multiple hunks', () => {
  const before = Array.from({ length: 30 }, (_, i) => `line ${i}\n`).join('');
  const after = before.replace('line 2\n', 'replacement\n').replace('line 25\n', '');
  const result = diff(before, after);
  assert.equal(result.diff.match(/@@ -/g).length, 2);
  assert.deepEqual(original({ status: 'modified', content: after, diff: result.diff }), lines(before));
});

test('handles zero-length insertion and deletion hunks', () => {
  for (const [before, after] of [['', 'new\n'], ['old\n', ''], ['a\n', 'b\na\n']]) {
    const result = diff(before, after);
    assert.deepEqual(original({ status: 'modified', content: after, diff: result.diff }), lines(before));
  }
});

test('reverted edits have no net additions or deletions', () => {
  const result = diff('original\n', 'original\n');
  assert.equal(result.diff, '');
  assert.equal(result.additions, 0);
  assert.equal(result.deletions, 0);
});

test('empty diff uses current content as original', () => {
  assert.deepEqual(original({ status: 'modified', content: 'unchanged', diff: '' }), ['unchanged']);
});

test('preserves newline-only changes and explicit missing-newline markers', () => {
  for (const [before, after] of [['line', 'line\n'], ['line\n', 'line'], ['a\r\n', 'a\n']]) {
    const result = diff(before, after);
    assert.equal(result.additions, 1);
    assert.equal(result.deletions, 1);
    assert.deepEqual(original({ status: 'modified', content: after, diff: result.diff }), lines(before));
  }
});

test('preserves removed rows beginning with dash characters', () => {
  const result = diff('-- heading\n', 'new\n');
  assert.ok(result.diff.includes('--- heading'));
  assert.deepEqual(original({ status: 'modified', content: 'new\n', diff: result.diff }), ['-- heading\n']);
});

test('rejects malformed or mismatched baseline diffs', () => {
  assert.throws(() => original({ status: 'modified', content: 'wrong\n', diff: '@@ -1 +1 @@\n-old\n+new\n' }));
  assert.throws(() => original({ status: 'modified', content: 'new\n', diff: '@@ -1,3 +1 @@\n-old\n+new\n' }));
  assert.throws(() => original({ status: 'modified', content: 'new\n', diff: 'invalid' }));
});

test('repeated-line diffs round-trip and use minimal change counts', () => {
  let seed = 42;
  const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed; };
  function lcs(a, b) {
    const row = new Array(b.length + 1).fill(0);
    for (const value of a) {
      let previous = 0;
      for (let j = 1; j <= b.length; j++) {
        const above = row[j];
        row[j] = value === b[j - 1] ? previous + 1 : Math.max(row[j], row[j - 1]);
        previous = above;
      }
    }
    return row[b.length];
  }
  for (let i = 0; i < 100; i++) {
    const before = Array.from({ length: random() % 20 }, () => `${random() % 4}\n`);
    const after = Array.from({ length: random() % 20 }, () => `${random() % 4}\n`);
    const result = context.inlineFileDiff(before, after);
    const common = lcs(before, after);
    assert.equal(result.deletions, before.length - common);
    assert.equal(result.additions, after.length - common);
    assert.deepEqual(original({ status: 'modified', content: after.join(''), diff: result.diff }), before);
  }
});
