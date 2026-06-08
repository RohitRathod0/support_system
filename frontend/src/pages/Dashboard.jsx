import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Send, Trash2, Home, Activity, Database, Zap } from 'lucide-react'
import { motion } from 'framer-motion'
import { useChat } from '../hooks/useChat'
import MessageBubble from '../components/MessageBubble'
import PipelinePanel from '../components/PipelinePanel'
import ImageUpload from '../components/ImageUpload'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const { messages, isLoading, pipelineNodes, lastMeta, sendMessage, clearChat, sessionId } = useChat()
  const [input, setInput] = useState('')
  const [image, setImage] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const bottomRef = useRef()
  const textRef = useRef()

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // Fetch analytics once on mount, then every 30s only
  useEffect(() => {
    const load = () => fetch('/analytics/trends').then(r => r.json()).then(setAnalytics).catch(() => {})
    load()
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [])

  const submit = () => {
    const q = input.trim()
    if (!q && !image) return
    sendMessage(q || 'Image complaint', image?.base64 ?? null, image?.url ?? null)
    setInput('')
    setImage(null)
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  // Stats bar
  const doneNodes = pipelineNodes.filter(n => n.status === 'done' || n.status === 'completed').length
  const runningNode = pipelineNodes.find(n => n.status === 'running')

  return (
    <div className={styles.page}>

      {/* ── Header ─────────────────────────────────────────── */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.brandIcon}><Activity size={16} /></div>
          <span className={styles.brandName}>Agent Dashboard</span>
          <span className={styles.badge}>Internal</span>
        </div>
        <div className={styles.pills}>
          <Pill color="#6366f1">⚡ LangGraph</Pill>
          <Pill color="#ef4444">📦 Redis</Pill>
          <Pill color="#10b981">🔍 ChromaDB</Pill>
          {lastMeta?.cache_hit && <Pill color="#10b981">✅ Cache Hit</Pill>}
        </div>
        <div className={styles.headerRight}>
          {messages.length > 0 && (
            <button className={styles.iconBtn} onClick={clearChat} title="Clear"><Trash2 size={14}/></button>
          )}
          <Link to="/" className={styles.backBtn}><Home size={13}/> Customer View</Link>
        </div>
      </header>

      {/* ── 3-column layout ────────────────────────────────── */}
      <div className={styles.body}>

        {/* LEFT — Metrics Sidebar */}
        <aside className={styles.sidebar}>
          <Section title="📊 Session Metrics">
            <div className={styles.metricGrid}>
              <Metric label="Messages" value={messages.filter(m=>m.role==='user').length} />
              <Metric label="Cache Hit" value={lastMeta?.cache_hit ? 'Yes' : 'No'} />
              <Metric label="QA Score" value={lastMeta?.qa_score ? `${lastMeta.qa_score}/10` : '—'} />
              <Metric label="Avg Time" value={lastMeta?.processing_time_seconds ? `${lastMeta.processing_time_seconds.toFixed(1)}s` : '—'} />
            </div>
          </Section>

          {lastMeta && (
            <Section title="🏷️ Last Classification">
              <InfoRow k="Category" v={lastMeta.issue_category || '—'} />
              <InfoRow k="Urgency"  v={lastMeta.urgency_level  || '—'} color={
                lastMeta.urgency_level === 'Critical' ? '#ef4444' :
                lastMeta.urgency_level === 'High'     ? '#f59e0b' : '#6366f1'
              } />
              <InfoRow k="Sentiment" v={lastMeta.sentiment || '—'} />
              <InfoRow k="Escalated" v={lastMeta.escalation_needed ? 'Yes 🚨' : 'No'} />
            </Section>
          )}

          <Section title="🔥 Trending (24h)">
            {analytics?.trending_categories?.length ? (
              analytics.trending_categories.slice(0, 5).map(c => (
                <div key={c.category} className={styles.trendRow}>
                  <span className={styles.trendLabel}>{c.category}</span>
                  <div className={styles.trendBar}>
                    <div className={styles.trendFill}
                      style={{ width: `${Math.min(100, c.count_24h * 20)}%` }} />
                  </div>
                  <span className={styles.trendCount}>{c.count_24h}</span>
                </div>
              ))
            ) : <div className={styles.muted}>Send queries to see trends</div>}
          </Section>

          <Section title="⚡ Architecture">
            <div className={styles.archNote}><span style={{color:'#3b82f6'}}>■</span> Phase 3: KB + Policy + Web — simultaneous</div>
            <div className={styles.archNote}><span style={{color:'#3b82f6'}}>■</span> Phase 9: Escalation + CX — simultaneous</div>
            <div className={styles.archNote}><span style={{color:'#f59e0b'}}>◆</span> QA retry if score &lt; 7 (max 2×)</div>
          </Section>

          <div className={styles.quickLinks}>
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className={styles.ql}>📖 API Docs</a>
            <a href="http://localhost:8000/health" target="_blank" rel="noreferrer" className={styles.ql}>❤️ Health</a>
            <a href="http://localhost:8000/analytics/traces" target="_blank" rel="noreferrer" className={styles.ql}>🔍 Traces</a>
          </div>
        </aside>

        {/* CENTRE — Chat */}
        <main className={styles.main}>
          {/* Status bar */}
          <div className={styles.statusBar}>
            <span className={styles.nodeProgress}>
              {runningNode ? `▶ ${runningNode.label}` : doneNodes > 0 ? `✓ ${doneNodes}/${pipelineNodes.length} nodes` : 'Ready'}
            </span>
            {sessionId && <span className={styles.sessionId}>Session: {sessionId.slice(0, 8)}…</span>}
          </div>

          <div className={styles.messages}>
            {messages.length === 0 && (
              <div className={styles.emptyState}>
                <Zap size={32} color="#6366f1" />
                <p>Send a test query to see the full pipeline in action</p>
              </div>
            )}
            {messages.map(m => <MessageBubble key={m.id} message={m} />)}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className={styles.inputArea}>
            {image && (
              <div className={styles.attachPreview}>
                <img src={image.url.startsWith('/uploads') ? `http://localhost:8000${image.url}` : image.url} alt="att" />
                <button onClick={() => setImage(null)}>✕</button>
              </div>
            )}
            <div className={styles.inputRow}>
              <ImageUpload onImageReady={setImage} onClear={() => setImage(null)} disabled={isLoading} />
              <textarea
                ref={textRef}
                className={styles.textarea}
                placeholder="Test any query to see all 10 nodes fire…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKey}
                rows={1}
                disabled={isLoading}
              />
              <button className={styles.sendBtn} onClick={submit} disabled={isLoading || (!input.trim() && !image)}>
                {isLoading
                  ? <motion.div className={styles.spinner} animate={{rotate:360}} transition={{duration:1,repeat:Infinity,ease:'linear'}}/>
                  : <Send size={16}/>}
              </button>
            </div>
          </div>
        </main>

        {/* RIGHT — Live Pipeline */}
        <aside className={styles.pipeline}>
          <PipelinePanel nodes={pipelineNodes} lastMeta={lastMeta} />
        </aside>
      </div>
    </div>
  )
}

function Pill({ color, children }) {
  return (
    <span className={styles.pill} style={{'--c': color}}>
      {children}
    </span>
  )
}

function Section({ title, children }) {
  return (
    <div className={styles.section}>
      <div className={styles.sectionTitle}>{title}</div>
      {children}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className={styles.metricCard}>
      <div className={styles.metricValue}>{value}</div>
      <div className={styles.metricLabel}>{label}</div>
    </div>
  )
}

function InfoRow({ k, v, color }) {
  return (
    <div className={styles.infoRow}>
      <span className={styles.infoKey}>{k}</span>
      <span className={styles.infoVal} style={color ? {color} : {}}>{v}</span>
    </div>
  )
}
