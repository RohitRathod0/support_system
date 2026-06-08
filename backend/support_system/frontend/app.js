const API = 'http://localhost:8000';

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  sessionId: null,
  userId: `user_${Math.random().toString(36).slice(2,10)}`,
  isStreaming: false,
  metrics: { totalQueries: 0, cacheHits: 0, avgTime: 0, times: [] },
  lastEval: null,
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const messagesEl   = $('messages');
const inputEl      = $('queryInput');
const sendBtn      = $('sendBtn');
const welcomeEl    = $('welcome');
const nodesEl      = $('pipelineNodes');
const tracesEl     = $('traceList');
const fcrEl        = $('fcrPanel');

// ─── Pipeline node definitions ────────────────────────────────────────────────
const NODES = [
  { id:'guardrails',           icon:'🛡️',  label:'Guardrails',          msg:'Validating input...', badge:null },
  { id:'classify_ticket',      icon:'🏷️',  label:'Ticket Classifier',   msg:'Classifying...', badge:null },
  { id:'manage_session',       icon:'👤',  label:'Session Manager',      msg:'Loading context...', badge:null },
  { id:'parallel_retrieval',   icon:'⚡',  label:'PARALLEL RETRIEVAL',   msg:'KB + Policy + Web simultaneously', badge:'parallel' },
  { id:'fuse_information',     icon:'🔗',  label:'Information Fusion',   msg:'Merging sources...', badge:null },
  { id:'generate_solution',    icon:'💡',  label:'Solution Generator',   msg:'Generating...', badge:null },
  { id:'personalize_response', icon:'✍️',  label:'Dynamic Responder',    msg:'Personalizing...', badge:null },
  { id:'qa_review',            icon:'⭐',  label:'QA Review',            msg:'Quality check...', badge:'conditional' },
  { id:'persist_conversation', icon:'💾',  label:'Conversation Persister',msg:'Storing to ChromaDB+Redis...', badge:null },
  { id:'escalation_coordinator',icon:'🚨',  label:'Escalation (parallel)',msg:'Checking escalation...', badge:'parallel' },
  { id:'cx_optimizer',         icon:'📊',  label:'CX Optimizer (parallel)',msg:'CX analysis...', badge:'parallel' },
  { id:'evaluation',           icon:'🔍',  label:'FCR Evaluation',       msg:'Scoring response...', badge:null },
];

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  renderPipelineNodes();
  await loadAnalytics();
  await loadGraphVisualization();
  setInterval(loadAnalytics, 30000);
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  inputEl.addEventListener('input', autoResize);
});

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

// ─── Pipeline Nodes Render ────────────────────────────────────────────────────
function renderPipelineNodes(activeNode = null, doneNodes = [], timings = {}) {
  nodesEl.innerHTML = '';
  NODES.forEach((n, i) => {
    let status = 'idle';
    if (doneNodes.includes(n.id)) status = 'done';
    else if (n.id === activeNode)  status = 'running';

    const timing = timings[n.id] ? `${timings[n.id].toFixed(2)}s` : '';

    const el = document.createElement('div');
    el.className = `pipeline-node ${status} ${n.badge === 'parallel' ? 'parallel' : ''}`;
    el.id = `node-${n.id}`;
    el.innerHTML = `
      <span class="node-icon">${n.icon}</span>
      <div class="node-info">
        <div class="node-name">${n.label}</div>
        <div class="node-msg">${n.msg}</div>
      </div>
      ${n.badge ? `<span class="node-badge badge-${n.badge}">${n.badge}</span>` : ''}
      ${timing ? `<span class="node-timing">${timing}</span>` : ''}
    `;
    nodesEl.appendChild(el);
    if (i < NODES.length - 1) {
      const conn = document.createElement('div');
      conn.className = 'pipeline-connector';
      nodesEl.appendChild(conn);
    }
  });
}

function setNodeStatus(nodeId, status, extra = '') {
  const el = document.getElementById(`node-${nodeId}`);
  if (!el) return;
  el.className = `pipeline-node ${status} ${el.classList.contains('parallel') ? 'parallel' : ''}`;
  if (extra) {
    const msgEl = el.querySelector('.node-msg');
    if (msgEl) msgEl.textContent = extra;
  }
}

// ─── Send Message ─────────────────────────────────────────────────────────────
async function sendMessage(query = null) {
  const q = (query || inputEl.value).trim();
  if (!q || state.isStreaming) return;

  if (welcomeEl) welcomeEl.style.display = 'none';
  inputEl.value = '';
  inputEl.style.height = 'auto';

  appendUserMessage(q);
  state.isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.innerHTML = '⏳';

  const typingId = appendTyping();
  const doneNodes = [];
  const timings = {};
  renderPipelineNodes(null, doneNodes, timings);

  try {
    const res = await fetch(`${API}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: q,
        user_id: state.userId,
        session_id: state.sessionId,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      removeTyping(typingId);
      appendBotMessage(`❌ Error: ${err.detail || 'Unknown error'}`, {});
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const chunk = JSON.parse(line.slice(6));
          handleStreamChunk(chunk, doneNodes, timings);
          if (chunk.node === 'final' && chunk.status === 'completed') {
            finalData = chunk.partial_data;
          }
        } catch (e) { /* skip malformed */ }
      }
    }

    removeTyping(typingId);

    if (finalData) {
      state.sessionId = finalData.session_id;
      appendBotMessage(finalData.response || 'No response generated.', finalData);
      updateMetrics(finalData);
      if (finalData.evaluation) updateFCRPanel(finalData.evaluation);
    }

    renderPipelineNodes(null, doneNodes, timings);

  } catch (err) {
    removeTyping(typingId);
    appendBotMessage(`⚠️ Connection error: ${err.message}. Is the backend running on port 8000?`, {});
    console.error(err);
  } finally {
    state.isStreaming = false;
    sendBtn.disabled = false;
    sendBtn.innerHTML = '<span>Send</span><span>↑</span>';
  }
}

function handleStreamChunk(chunk, doneNodes, timings) {
  const { node, status, message, partial_data } = chunk;

  if (node === 'start' || node === 'final') return;

  if (status === 'running') {
    setNodeStatus(node, 'running', message);
  } else if (status === 'completed') {
    if (!doneNodes.includes(node)) doneNodes.push(node);
    if (partial_data?.node_timings) Object.assign(timings, partial_data.node_timings);
    setNodeStatus(node, 'done', message);
    addTraceEntry(node, message, timings[node]);
  } else if (status === 'error') {
    setNodeStatus(node, 'error', message);
  }
}

// ─── Messages ────────────────────────────────────────────────────────────────
function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'message user';
  el.innerHTML = `
    <div class="avatar user">👤</div>
    <div class="bubble user">${escHtml(text)}</div>
  `;
  messagesEl.appendChild(el);
  scrollBottom();
}

function appendBotMessage(text, meta = {}) {
  const chips = buildChips(meta);
  const escalation = meta.escalation_needed
    ? `<div class="escalation-banner">🚨 This issue has been escalated. ${meta.escalation_report?.customer_communication || 'A specialist will contact you.'}</div>`
    : '';

  const el = document.createElement('div');
  el.className = 'message bot';
  el.innerHTML = `
    <div class="avatar bot">🤖</div>
    <div>
      <div class="bubble bot">${formatResponse(text)}${escalation}</div>
      <div class="bubble-meta">${chips}</div>
    </div>
  `;
  messagesEl.appendChild(el);
  scrollBottom();
}

function buildChips(meta) {
  const chips = [];
  if (meta.cache_hit) chips.push(`<span class="meta-chip chip-cache">⚡ Cache Hit</span>`);
  if (meta.urgency_level) chips.push(`<span class="meta-chip chip-urgency-${meta.urgency_level}">🔴 ${meta.urgency_level}</span>`);
  if (meta.issue_category) chips.push(`<span class="meta-chip chip-category">📁 ${meta.issue_category}</span>`);
  if (meta.qa_score)  chips.push(`<span class="meta-chip chip-score">⭐ QA ${meta.qa_score}/10</span>`);
  if (meta.processing_time_seconds) chips.push(`<span class="meta-chip chip-time">⏱ ${meta.processing_time_seconds.toFixed(2)}s</span>`);
  if (meta.evaluation?.overall_fcr_score) chips.push(`<span class="meta-chip chip-fcr">🎯 FCR ${meta.evaluation.overall_fcr_score}/10</span>`);
  return chips.join('');
}

function appendTyping() {
  const id = 'typing-' + Date.now();
  const el = document.createElement('div');
  el.id = id;
  el.className = 'message bot';
  el.innerHTML = `
    <div class="avatar bot">🤖</div>
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  messagesEl.appendChild(el);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function formatResponse(text) {
  return escHtml(text)
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px;font-family:var(--mono)">$1</code>');
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scrollBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

// ─── Traces ───────────────────────────────────────────────────────────────────
const traceLog = [];

function addTraceEntry(node, msg, timing) {
  traceLog.unshift({ node, msg, timing: timing?.toFixed(2) });
  if (traceLog.length > 15) traceLog.pop();
  renderTraces();
}

function renderTraces() {
  if (!tracesEl) return;
  tracesEl.innerHTML = traceLog.map(t => `
    <div class="trace-entry">
      <span class="trace-node">${t.node}</span>
      <span class="trace-time">${t.timing ? t.timing + 's' : '—'}</span>
    </div>
  `).join('');
}

// ─── FCR Panel ────────────────────────────────────────────────────────────────
function updateFCRPanel(evaluation) {
  if (!fcrEl || !evaluation) return;
  const score = evaluation.overall_fcr_score || 0;
  const verdict = evaluation.verdict || 'Unknown';
  const cls = verdict === 'Fully Resolved' ? 'verdict-full'
            : verdict === 'Partially Resolved' ? 'verdict-partial' : 'verdict-not';

  fcrEl.innerHTML = `
    <div class="panel-title">🎯 FCR Evaluation</div>
    <div class="fcr-score">${score.toFixed(1)}</div>
    <div class="fcr-verdict ${cls}">${verdict}</div>
    ${evaluation.gaps?.length ? `<div style="font-size:11px;color:var(--text-muted);margin-top:8px">Gaps: ${evaluation.gaps.slice(0,2).join(', ')}</div>` : ''}
    ${evaluation.recommendation ? `<div style="font-size:11px;color:var(--text-secondary);margin-top:4px">${evaluation.recommendation}</div>` : ''}
  `;
}

// ─── Metrics ─────────────────────────────────────────────────────────────────
function updateMetrics(data) {
  state.metrics.totalQueries++;
  if (data.cache_hit) state.metrics.cacheHits++;
  if (data.processing_time_seconds) {
    state.metrics.times.push(data.processing_time_seconds);
    state.metrics.avgTime = state.metrics.times.reduce((a, b) => a + b, 0) / state.metrics.times.length;
  }

  const totalEl = $('metricTotal');
  const cacheEl = $('metricCache');
  const timeEl  = $('metricTime');
  const qaEl    = $('metricQA');

  if (totalEl) totalEl.textContent = state.metrics.totalQueries;
  if (cacheEl) cacheEl.textContent = `${Math.round(state.metrics.cacheHits / state.metrics.totalQueries * 100)}%`;
  if (timeEl)  timeEl.textContent  = `${state.metrics.avgTime.toFixed(1)}s`;
  if (qaEl && data.qa_score) qaEl.textContent = `${data.qa_score}/10`;
}

// ─── Analytics ────────────────────────────────────────────────────────────────
async function loadAnalytics() {
  try {
    const res = await fetch(`${API}/analytics/trends`);
    if (!res.ok) return;
    const data = await res.json();
    renderTrending(data.trending_categories || []);
  } catch { /* backend not running yet */ }
}

function renderTrending(categories) {
  const el = $('trendingBars');
  if (!el || !categories.length) return;
  const max = Math.max(...categories.map(c => c.count_24h), 1);
  el.innerHTML = categories.slice(0, 6).map(c => `
    <div class="trend-bar">
      <span class="trend-label">${c.category}</span>
      <div class="trend-track">
        <div class="trend-fill ${c.is_trending ? 'trending' : ''}" style="width:${Math.round(c.count_24h/max*100)}%"></div>
      </div>
      <span class="trend-count">${c.count_24h}</span>
    </div>
  `).join('');
}

// ─── Graph Visualization ──────────────────────────────────────────────────────
async function loadGraphVisualization() {
  try {
    const res = await fetch(`${API}/health/graph`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.mermaid) renderMermaid(data.mermaid);
  } catch { /* skip if backend not running */ }
}

function renderMermaid(mermaidCode) {
  const el = $('mermaidGraph');
  if (!el || typeof mermaid === 'undefined') return;
  el.innerHTML = `<div class="mermaid">${mermaidCode}</div>`;
  mermaid.run({ querySelector: '#mermaidGraph .mermaid' });
}

// ─── Quick query suggestion ───────────────────────────────────────────────────
function setQuery(text) {
  inputEl.value = text;
  inputEl.focus();
  autoResize();
}

// ─── Expose globals for HTML onclick ─────────────────────────────────────────
window.sendMessage = sendMessage;
window.setQuery    = setQuery;
