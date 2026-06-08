import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Send, Trash2, LayoutDashboard, Headphones } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useChat } from '../hooks/useChat'
import MessageBubble from '../components/MessageBubble'
import ImageUpload from '../components/ImageUpload'
import styles from './CustomerPortal.module.css'

const QUICK = [
  { icon: '💳', text: 'Payment was declined' },
  { icon: '🔄', text: 'I want to request a refund' },
  { icon: '🔐', text: 'I cannot log into my account' },
  { icon: '📦', text: 'My order has not arrived' },
  { icon: '❌', text: 'I want to cancel my subscription' },
  { icon: '📱', text: 'The app keeps crashing' },
]

export default function CustomerPortal() {
  const { messages, isLoading, sendMessage, clearChat, lastMeta } = useChat()
  const [input, setInput] = useState('')
  const [image, setImage] = useState(null)   // { url, base64, contentType }
  const bottomRef = useRef()
  const textRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = () => {
    const q = input.trim()
    if (!q && !image) return
    sendMessage(q || 'I have attached an image with my complaint.', image?.base64 ?? null, image?.url ?? null)
    setInput('')
    setImage(null)
    textRef.current?.focus()
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  const isEmpty = messages.length === 0

  return (
    <div className={styles.page}>
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.brandIcon}><Headphones size={18} /></div>
          <div>
            <div className={styles.brandName}>Support AI</div>
            <div className={styles.brandSub}>Powered by LangGraph · Always here to help</div>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.statusPill}>
            <span className={styles.statusDot} />
            Online
          </div>
          {messages.length > 0 && (
            <button className={styles.iconBtn} onClick={clearChat} title="New conversation">
              <Trash2 size={15} />
            </button>
          )}
          <Link to="/dashboard" className={styles.dashBtn}>
            <LayoutDashboard size={14} /> Dashboard
          </Link>
        </div>
      </header>

      {/* ── Chat area ──────────────────────────────────────────────── */}
      <main className={styles.chat}>
        <AnimatePresence>
          {isEmpty ? (
            <motion.div
              className={styles.welcome}
              key="welcome"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className={styles.welcomeIcon}>🤖</div>
              <h1 className={styles.welcomeTitle}>How can I help you today?</h1>
              <p className={styles.welcomeSub}>
                Describe your issue in detail — or attach a <strong>product photo</strong> to help us understand the problem.
                Our AI processes your request through 10 specialized agents and stays strictly within company policy.
              </p>
              <div className={styles.quickGrid}>
                {QUICK.map((q) => (
                  <button
                    key={q.text}
                    className={styles.quickBtn}
                    onClick={() => { setInput(q.text); textRef.current?.focus() }}
                  >
                    <span>{q.icon}</span> {q.text}
                  </button>
                ))}
              </div>
            </motion.div>
          ) : (
            <div className={styles.messages} key="messages">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </AnimatePresence>
      </main>

      {/* ── Input bar ──────────────────────────────────────────────── */}
      <footer className={styles.inputBar}>
        {/* Image attachment */}
        {image && (
          <div className={styles.attachPreview}>
            <img src={image.url.startsWith('/uploads') ? `http://localhost:8000${image.url}` : image.url} alt="attachment" />
            <button onClick={() => setImage(null)}>✕</button>
          </div>
        )}

        <div className={styles.inputRow}>
          <ImageUpload
            onImageReady={(img) => setImage(img)}
            onClear={() => setImage(null)}
            disabled={isLoading}
          />
          <textarea
            ref={textRef}
            className={styles.textarea}
            placeholder="Describe your issue… (Enter to send, Shift+Enter for new line)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={1}
            disabled={isLoading}
          />
          <button
            className={styles.sendBtn}
            onClick={submit}
            disabled={isLoading || (!input.trim() && !image)}
          >
            {isLoading ? (
              <motion.div
                className={styles.spinner}
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              />
            ) : <Send size={17} />}
          </button>
        </div>

        <div className={styles.inputFooter}>
          <span>🛡️ Policy-bound AI · 📎 Image upload supported · ⚡ Parallel processing</span>
          {lastMeta?.processing_time_seconds && (
            <span>Last response: {lastMeta.processing_time_seconds.toFixed(1)}s</span>
          )}
        </div>
      </footer>
    </div>
  )
}
