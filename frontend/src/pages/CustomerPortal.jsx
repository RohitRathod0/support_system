import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Send, Trash2, LayoutDashboard, Headphones } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useChat } from '../hooks/useChat'
import MessageBubble from '../components/MessageBubble'
import ImageUpload from '../components/ImageUpload'
import VideoReturnSession from '../components/VideoReturnSession'
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
  const { messages, isLoading, sendMessage, clearChat, submitFeedback, lastMeta } = useChat()
  const [input, setInput] = useState('')
  const [image, setImage] = useState(null)   // { url, base64, contentType }
  const [showFeedback, setShowFeedback] = useState(false)
  const [rating, setRating] = useState(0)
  const [comment, setComment] = useState('')
  const [videoTriggerPending, setVideoTriggerPending] = useState(false)
  const [showVideoSession, setShowVideoSession] = useState(false)
  const bottomRef = useRef()
  const textRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (lastMeta?.trigger_video) {
      setVideoTriggerPending(true)
    } else {
      setVideoTriggerPending(false)
    }
  }, [lastMeta])

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

  if (showVideoSession) {
    return (
      <div style={{ width: '100vw', height: '100vh' }}>
        <VideoReturnSession 
          customerId={userId} 
          orderId={sessionId || "ORD-" + Math.floor(Math.random()*10000)}
          productCategory={lastMeta?.issue_category || "General"}
          orderValue={100}
        />
      </div>
    )
  }

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
            <button className={styles.iconBtn} onClick={() => setShowFeedback(true)} title="End conversation & give feedback" style={{ width: 'auto', padding: '0 12px', fontSize: '13px' }}>
              End Chat
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
              {messages.map((m) => {
                const isVideoTrigger = m.role === 'bot' && m.meta?.trigger_video;
                
                if (isVideoTrigger && videoTriggerPending) {
                  const overrideMessage = {
                    ...m,
                    text: "I can see this might be a product damage issue. To help you faster, let's do a quick video check. Please click below to show us the product."
                  };
                  return (
                    <div key={m.id}>
                      <MessageBubble message={overrideMessage} />
                      <div style={{ paddingLeft: '48px', marginTop: '-8px', marginBottom: '16px' }}>
                        <button 
                          className={styles.dashBtn} 
                          style={{ background: '#3b82f6', color: 'white', border: 'none', cursor: 'pointer', padding: '8px 16px', borderRadius: '8px', fontSize: '14px', fontWeight: '500' }}
                          onClick={() => setShowVideoSession(true)}
                        >
                          Start Video Return
                        </button>
                      </div>
                    </div>
                  )
                }
                
                return <MessageBubble key={m.id} message={m} />
              })}
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

      {/* ── Feedback Modal ──────────────────────────────────────────── */}
      {showFeedback && (
        <div className={styles.modalOverlay}>
          <motion.div 
            className={styles.modalContent}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <div className={styles.modalTitle}>How did we do?</div>
            <div className={styles.modalSub}>Your feedback helps us improve our AI support agent.</div>
            
            <div className={styles.starRating}>
              {[1, 2, 3, 4, 5].map((s) => (
                <button 
                  key={s} 
                  className={`${styles.star} ${rating >= s ? styles.active : ''}`}
                  onClick={() => setRating(s)}
                >
                  ★
                </button>
              ))}
            </div>

            <textarea 
              className={styles.commentArea}
              placeholder="Tell us what you liked or what could be better (optional)"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
            />

            <div className={styles.modalActions}>
              <button 
                className={styles.btnCancel} 
                onClick={() => setShowFeedback(false)}
              >
                Cancel
              </button>
              <button 
                className={styles.btnSubmit} 
                disabled={rating === 0}
                onClick={async () => {
                  const escalated = lastMeta?.escalation_needed || false
                  await submitFeedback(rating, comment, escalated)
                  setShowFeedback(false)
                  setRating(0)
                  setComment('')
                  clearChat()
                }}
              >
                Submit Feedback
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}
