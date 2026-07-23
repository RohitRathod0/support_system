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

const CATEGORY_EMOJI = {
  'Home Appliances': '🫙',
  'Clothing': '👕',
  'Electronics': '🎧',
}

export default function CustomerPortal() {
  const { messages, isLoading, sendMessage, clearChat, lastMeta, userId, sessionId } = useChat()
  const [input, setInput] = useState('')
  const [image, setImage] = useState(null)   // { url, base64, contentType }
  const [showFeedback, setShowFeedback] = useState(false)
  const [rating, setRating] = useState(0)
  const [comment, setComment] = useState('')
  const [videoTriggerPending, setVideoTriggerPending] = useState(false)
  const [showVideoSession, setShowVideoSession] = useState(false)
  // Resolution + inline feedback
  const [resolutionDetected, setResolutionDetected] = useState(false)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false)
  const [feedbackRating, setFeedbackRating] = useState({ resolved: null, stars: null })
  const [feedbackComment, setFeedbackComment] = useState('')
  const [hoverStar, setHoverStar] = useState(0)
  // Orders
  const [orders, setOrders] = useState([])
  const [selectedOrder, setSelectedOrder] = useState(null)
  const bottomRef = useRef()
  const textRef = useRef()

  // Fetch mock orders on mount
  useEffect(() => {
    fetch('/api/orders/')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data?.orders) setOrders(data.orders) })
      .catch(() => {})
  }, [])

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

  // Resolution detection
  useEffect(() => {
    if (lastMeta?.resolution_detected && !resolutionDetected && !feedbackSubmitted) {
      setResolutionDetected(true)
      // Fire and forget the resolve call
      fetch('/chat/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, customer_id: userId, resolved_by: 'ai' }),
      }).catch(() => {})
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastMeta])

  const submit = async () => {
    const q = input.trim()
    if (!q && !image) return

    const visibleQuery = q || 'I have attached an image with my complaint.'

    // Always include selected order context and live courier data if selected
    let orderPrefix = ''
    let trackingContext = ''
    if (selectedOrder) {
      orderPrefix = `[Selected Order Context: ID=${selectedOrder.order_id} | Name=${selectedOrder.product_name} | Status=${selectedOrder.status} | Expected Delivery=${selectedOrder.expected_delivery} | Tracking=${selectedOrder.tracking_number}] `
      
      try {
        const res = await fetch(`/api/orders/tracking/${selectedOrder.tracking_number}`)
        if (res.ok) {
          const data = await res.json()
          trackingContext = ` [LIVE COURIER PARTNER DATA — use this to answer: ${data.agent_message}]`
        }
      } catch (_) {}
    }

    const backendQuery = orderPrefix + visibleQuery + trackingContext

    sendMessage(
      visibleQuery,
      backendQuery,
      image?.base64 ?? null,
      image?.url ?? null
    )
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
                Select an order below to get support for that purchase, then describe your issue.
                Our AI stays strictly within your order context.
              </p>

              {/* ── Order Cards ── */}
              {orders.length > 0 && (
                <div className={styles.ordersSection}>
                  <div className={styles.ordersSectionTitle}>📦 Your Recent Orders</div>
                  <div className={styles.ordersGrid}>
                    {orders.map(order => (
                      <button
                        key={order.order_id}
                        className={`${styles.orderCard} ${selectedOrder?.order_id === order.order_id ? styles.selected : ''}`}
                        onClick={() => {
                          setSelectedOrder(order)
                          textRef.current?.focus()
                        }}
                      >
                        <span className={styles.orderCardEmoji}>{CATEGORY_EMOJI[order.category] || '📦'}</span>
                        <div className={styles.orderCardName}>{order.product_name}</div>
                        <div className={styles.orderCardMeta}>
                          <span className={styles.orderCardId}>{order.order_id}</span>
                          <span className={styles.orderCardDate}>
                            Ordered: {new Date(order.order_date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
                          </span>
                          <span className={styles.orderCardDate} style={{ color: order.status === 'In Transit' ? '#f59e0b' : '#10b981' }}>
                            {order.status === 'Delivered' ? '✓ Delivered:' : '🚚 Expected:'}{' '}
                            {new Date(order.expected_delivery).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
                          </span>
                          <span className={styles.orderCardId} style={{ color: '#64748b', fontSize: '9px', letterSpacing: '0.06em' }}>
                            🔍 {order.tracking_number}
                          </span>
                        </div>

                        {/* Mini shipment timeline for In Transit orders */}
                        {order.status === 'In Transit' && order.timeline && (
                          <div style={{ marginTop: '10px', marginBottom: '4px' }}>
                            {order.timeline.map((step, i) => (
                              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '3px' }}>
                                <div style={{
                                  width: '12px', height: '12px', borderRadius: '50%', flexShrink: 0, marginTop: '1px',
                                  background: step.done ? '#6366f1' : 'rgba(255,255,255,0.1)',
                                  border: step.done ? 'none' : '1px solid rgba(255,255,255,0.2)',
                                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}>
                                  {step.done && <span style={{ fontSize: '7px', color: 'white' }}>✓</span>}
                                </div>
                                <span style={{ fontSize: '10px', color: step.done ? '#cbd5e1' : '#475569', lineHeight: 1.3 }}>
                                  {step.event}
                                  <span style={{ color: '#334155', marginLeft: '4px' }}>· {step.date}</span>
                                </span>
                              </div>
                            ))}
                          </div>
                        )}

                        <div className={styles.orderCardFooter}>
                          <span className={styles.orderCardAmount}>₹{order.amount_rupees.toLocaleString('en-IN')}</span>
                          <span className={`${styles.orderCardStatus} ${order.status === 'Delivered' ? styles.delivered : styles.transit}`}>
                            {order.status}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* ── Quick actions (only shown if an order is selected) ── */}
              {selectedOrder && (
                <>
                  <p style={{ fontSize: '12px', color: '#6366f1', fontWeight: 600, marginBottom: '-8px' }}>
                    ✓ {selectedOrder.product_name} selected — pick a quick issue or type below
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
                </>
              )}
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

              {/* ── Feedback Questionnaire — shown once after resolution ── */}
              {resolutionDetected && !feedbackSubmitted && (
                <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '20px' }}>
                  <div style={{ width: 32, height: 32, borderRadius: '50%', background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 16 }}>🤖</div>
                  <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                    style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.08))', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '16px', padding: '20px 24px', maxWidth: '480px', width: '100%' }}
                  >
                    <p style={{ margin: '0 0 4px', color: '#c7d2fe', fontSize: '13px', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Quick Feedback</p>
                    <p style={{ margin: '0 0 18px', color: '#e2e8f0', fontSize: '15px', fontWeight: 600 }}>
                      Help us improve our AI agent 😊
                    </p>

                    {/* Q1 — Was your query resolved? */}
                    <div style={{ marginBottom: '18px' }}>
                      <p style={{ margin: '0 0 10px', color: '#94a3b8', fontSize: '13px' }}>
                        1. Was your query resolved?
                      </p>
                      <div style={{ display: 'flex', gap: '10px' }}>
                        {[{ label: '✅ Yes', val: true }, { label: '❌ No', val: false }].map(({ label, val }) => (
                          <button
                            key={String(val)}
                            onClick={() => setFeedbackRating(prev => ({ ...prev, resolved: val }))}
                            style={{
                              padding: '8px 20px', borderRadius: '8px', border: '1px solid',
                              cursor: 'pointer', fontSize: '13px', fontWeight: 600, transition: 'all 0.15s',
                              borderColor: (feedbackRating?.resolved === val) ? (val ? '#10b981' : '#ef4444') : 'rgba(255,255,255,0.12)',
                              background: (feedbackRating?.resolved === val) ? (val ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)') : 'rgba(255,255,255,0.04)',
                              color: (feedbackRating?.resolved === val) ? (val ? '#34d399' : '#f87171') : '#94a3b8',
                            }}
                          >{label}</button>
                        ))}
                      </div>
                    </div>

                    {/* Q2 — Star rating */}
                    <div style={{ marginBottom: '18px' }}>
                      <p style={{ margin: '0 0 10px', color: '#94a3b8', fontSize: '13px' }}>
                        2. How was your experience with the agent? <span style={{ color: '#ef4444' }}>*</span>
                      </p>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        {[1, 2, 3, 4, 5].map(s => (
                          <button
                            key={s}
                            onMouseEnter={() => setHoverStar(s)}
                            onMouseLeave={() => setHoverStar(0)}
                            onClick={() => setFeedbackRating(prev => ({ ...prev, stars: s }))}
                            style={{
                              background: 'none', border: 'none', cursor: 'pointer', padding: '2px',
                              fontSize: '28px', lineHeight: 1, transition: 'transform 0.1s',
                              transform: (hoverStar === s || feedbackRating?.stars === s) ? 'scale(1.2)' : 'scale(1)',
                              color: s <= (hoverStar || feedbackRating?.stars || 0) ? '#f59e0b' : 'rgba(255,255,255,0.15)',
                              filter: s <= (hoverStar || feedbackRating?.stars || 0) ? 'drop-shadow(0 0 6px rgba(245,158,11,0.6))' : 'none',
                            }}
                          >★</button>
                        ))}
                      </div>
                      {feedbackRating?.stars && (
                        <p style={{ margin: '6px 0 0', fontSize: '12px', color: '#f59e0b' }}>
                          {['', 'Poor 😞', 'Fair 😐', 'Good 🙂', 'Great 😊', 'Excellent 🌟'][feedbackRating.stars]}
                        </p>
                      )}
                    </div>

                    {/* Q3 — Improvements / comments */}
                    <div style={{ marginBottom: '18px' }}>
                      <p style={{ margin: '0 0 8px', color: '#94a3b8', fontSize: '13px' }}>
                        3. Any improvements or comments? <span style={{ color: '#64748b', fontSize: '11px' }}>(optional)</span>
                      </p>
                      <textarea
                        value={feedbackComment}
                        onChange={e => {
                          if (typeof e.target.value === 'string') setFeedbackComment(e.target.value)
                        }}
                        placeholder="Share your thoughts… (text only)"
                        style={{
                          width: '100%', boxSizing: 'border-box', minHeight: '72px', resize: 'vertical',
                          background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: '10px', color: '#e2e8f0', padding: '10px 14px', fontSize: '13px',
                          lineHeight: 1.5, outline: 'none', fontFamily: 'inherit',
                          transition: 'border-color 0.15s',
                        }}
                        onFocus={e => e.target.style.borderColor = 'rgba(99,102,241,0.5)'}
                        onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
                      />
                    </div>

                    {/* Submit */}
                    <button
                      disabled={!feedbackRating?.stars}
                      onClick={async () => {
                        if (!feedbackRating?.stars) return
                        try {
                          await fetch('/api/chat/feedback', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              session_id:    sessionId,
                              customer_id:   userId,
                              rating:        feedbackRating.stars,
                              comment:       feedbackComment,
                              query_resolved: feedbackRating.resolved ?? null,
                            }),
                          })
                        } catch (_) {}
                        setFeedbackSubmitted(true)
                      }}
                      style={{
                        width: '100%', padding: '10px 0', borderRadius: '10px', border: 'none',
                        cursor: feedbackRating?.stars ? 'pointer' : 'not-allowed',
                        background: feedbackRating?.stars ? 'linear-gradient(135deg,#6366f1,#8b5cf6)' : 'rgba(255,255,255,0.06)',
                        color: feedbackRating?.stars ? 'white' : '#475569',
                        fontSize: '14px', fontWeight: 700, transition: 'all 0.2s',
                        boxShadow: feedbackRating?.stars ? '0 4px 15px rgba(99,102,241,0.4)' : 'none',
                      }}
                    >
                      {feedbackRating?.stars ? 'Submit Feedback →' : 'Please rate your experience ★'}
                    </button>
                  </motion.div>
                </div>
              )}

              {/* Thank-you after submit */}
              {resolutionDetected && feedbackSubmitted && (
                <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '16px' }}>
                  <div style={{ width: 32, height: 32, borderRadius: '50%', background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>🤖</div>
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '16px', padding: '14px 18px' }}
                  >
                    <p style={{ margin: 0, color: '#34d399', fontSize: '14px', fontWeight: 600 }}>
                      Thank you for your feedback! ⭐ Your response has been recorded and will help us improve the AI agent.
                    </p>
                  </motion.div>
                </div>
              )}

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
            <img src={image.url.startsWith('/uploads') ? `http://127.0.0.1:8000${image.url}` : image.url} alt="attachment" />
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
