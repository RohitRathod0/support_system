import { useState, useEffect, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'
import {
  Shield, AlertTriangle, CheckCircle2, XCircle, Clock,
  TrendingUp, Package, Bot, MessageSquare, LogOut,
  RefreshCw, ChevronRight, Star, BarChart2, Activity,
  AlertCircle, Home, Filter, Loader2, ThumbsUp, ThumbsDown,
  DollarSign, TicketCheck
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './AdminDashboard.module.css'

const API = ''

function authHeaders() {
  const token = localStorage.getItem('admin_token')
  return { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
}

// ─── Auth guard hook ──────────────────────────────────────────────────────────
function useAdminAuth() {
  const navigate = useNavigate()
  useEffect(() => {
    const token = localStorage.getItem('admin_token')
    if (!token) navigate('/admin/login')
  }, [navigate])
  const logout = useCallback(() => {
    localStorage.removeItem('admin_token')
    localStorage.removeItem('admin_email')
    navigate('/admin/login')
  }, [navigate])
  return { logout, email: localStorage.getItem('admin_email') || 'admin' }
}

// ─── Data hooks ───────────────────────────────────────────────────────────────
function useAdminData() {
  const [pending,    setPending]    = useState([])
  const [resolved,   setResolved]   = useState([])
  const [products,   setProducts]   = useState({ products: [], alert_count: 0 })
  const [aiSummary,  setAiSummary]  = useState(null)
  const [feedback,   setFeedback]   = useState({ feedback: [], summary: {} })
  const [loading,    setLoading]    = useState(true)
  const [lastRefresh, setLastRefresh] = useState(null)
  const [chatStats,  setChatStats]  = useState(null)
  const [feedbackList, setFeedbackList] = useState([])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [pRes, rRes, prRes, aiRes, fbRes, sRes, fRes] = await Promise.all([
        fetch(`${API}/api/admin/pending`,               { headers: authHeaders() }),
        fetch(`${API}/api/admin/resolved?limit=20`,     { headers: authHeaders() }),
        fetch(`${API}/api/admin/product-analytics`,     { headers: authHeaders() }),
        fetch(`${API}/api/admin/ai-resolution-summary`, { headers: authHeaders() }),
        fetch(`${API}/api/admin/feedback-feed`,         { headers: authHeaders() }),
        fetch(`${API}/api/chat/stats`),
        fetch(`${API}/api/chat/feedback/list`),
      ])
      if (pRes.status === 401) { localStorage.removeItem('admin_token'); window.location.href = '/admin/login'; return }
      const [p, r, pr, ai, fb] = await Promise.all([pRes.json(), rRes.json(), prRes.json(), aiRes.json(), fbRes.json()])
      setPending(p.pending  || [])
      setResolved(r.resolved || [])
      setProducts(pr)
      setAiSummary(ai)
      setFeedback(fb)
      if (sRes.ok) setChatStats(await sRes.json())
      if (fRes.ok) setFeedbackList(await fRes.json())
      setLastRefresh(new Date())
    } catch (e) {
      console.error('Admin data fetch error:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])
  useEffect(() => {
    const t = setInterval(refresh, 30000)
    return () => clearInterval(t)
  }, [refresh])

  return { pending, resolved, products, aiSummary, feedback, loading, lastRefresh, refresh, chatStats, feedbackList }
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function AdminDashboard() {
  const { logout, email } = useAdminAuth()
  const { pending, resolved, products, aiSummary, feedback, loading, lastRefresh, refresh, chatStats, feedbackList } = useAdminData()
  const [activeTab,    setActiveTab]    = useState('approvals')
  const [rejectModal,  setRejectModal]  = useState(null)   // ticket being rejected
  const [rejectReason, setRejectReason] = useState('')
  const [actionLoading, setActionLoading] = useState(null) // ticket_id being actioned
  const [feedbackFilter, setFeedbackFilter] = useState('all')
  const navigate = useNavigate()

  // ── Approve ticket ─────────────────────────────────────────────────────────
  const approveTicket = async (ticket) => {
    setActionLoading(ticket.ticket_id)
    if (ticket.session_id) {
      try {
        await axios.post(`${API}/api/return/decision/${ticket.session_id}`, {
          session_id: ticket.session_id,
          decision: 'approve'
        })
      } catch (e) {
        console.error('Failed agent decision update:', e)
      }
    }
    try {
      await fetch(`${API}/api/admin/approve/${ticket.ticket_id}`, { method: 'POST', headers: authHeaders() })
      await refresh()
    } catch (e) { console.error(e) }
    setActionLoading(null)
  }

  // ── Reject ticket ──────────────────────────────────────────────────────────
  const rejectTicket = async () => {
    if (!rejectModal || !rejectReason.trim()) return
    setActionLoading(rejectModal.ticket_id)
    if (rejectModal.session_id) {
      try {
        await axios.post(`${API}/api/return/decision/${rejectModal.session_id}`, {
          session_id: rejectModal.session_id,
          decision: 'reject'
        })
      } catch (e) {
        console.error('Failed agent decision update:', e)
      }
    }
    try {
      await fetch(`${API}/api/admin/reject/${rejectModal.ticket_id}`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ reason: rejectReason }),
      })
      await refresh()
    } catch (e) { console.error(e) }
    setRejectModal(null)
    setRejectReason('')
    setActionLoading(null)
  }

  const tabs = [
    { id: 'approvals',  label: 'Pending Approvals', icon: TicketCheck,  badge: pending.length },
    { id: 'products',   label: 'Product Analytics', icon: Package,       badge: products.alert_count || 0, badgeColor: '#f59e0b' },
    { id: 'ai',         label: 'AI Resolution',     icon: Bot,           badge: null },
    { id: 'feedback',   label: 'Customer Feedback', icon: MessageSquare, badge: null },
  ]

  return (
    <div className={styles.page}>
      {/* Background */}
      <div className={styles.bgGlow1} />
      <div className={styles.bgGlow2} />

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.logoIcon}><Shield size={18} /></div>
          <div>
            <div className={styles.logoTitle}>Admin Dashboard</div>
            <div className={styles.logoSub}>Human-in-the-Loop Control Center</div>
          </div>
          {pending.length > 0 && (
            <motion.div
              className={styles.urgentBadge}
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              🔴 {pending.length} Pending
            </motion.div>
          )}
        </div>
        <div className={styles.headerRight}>
          <span className={styles.adminEmail}>👤 {email}</span>
          {lastRefresh && (
            <span className={styles.refreshTime}>
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button
            id="refresh-btn"
            className={styles.iconBtn}
            onClick={refresh}
            disabled={loading}
            title="Refresh"
          >
            <RefreshCw size={14} className={loading ? styles.spin : ''} />
          </button>
          <Link to="/" className={styles.navLink}><Home size={13} /> Customer Portal</Link>
          <Link to="/dashboard" className={styles.navLink}><Activity size={13} /> Agent View</Link>
          <button className={styles.logoutBtn} onClick={logout}><LogOut size={14} /> Logout</button>
        </div>
      </header>

      {/* ── Tab Bar ─────────────────────────────────────────────────────────── */}
      <div className={styles.tabBar}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            className={`${styles.tab} ${activeTab === tab.id ? styles.tabActive : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <tab.icon size={15} />
            {tab.label}
            {tab.badge !== null && tab.badge > 0 && (
              <span className={styles.tabBadge} style={tab.badgeColor ? { background: tab.badgeColor } : {}}>
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── Content ─────────────────────────────────────────────────────────── */}
      <main className={styles.content}>
        <AnimatePresence mode="wait">

          {/* ═══ TAB A: PENDING APPROVALS ════════════════════════════════════ */}
          {activeTab === 'approvals' && (
            <motion.div key="approvals" className={styles.panel}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
            >
              <SectionHeader
                icon={<TicketCheck size={18} />}
                title="Pending Approvals"
                subtitle="Money-related issues flagged by AI — require your decision before any action is taken"
                color="#ef4444"
              />

              {pending.length === 0 ? (
                <EmptyState icon={<CheckCircle2 size={40} color="#10b981" />}
                  message="No pending approvals" sub="All money-matter tickets have been actioned." />
              ) : (
                <div className={styles.ticketGrid}>
                  {pending.map(ticket => (
                    <TicketCard
                      key={ticket.ticket_id}
                      ticket={ticket}
                      onApprove={() => approveTicket(ticket)}
                      onReject={() => { setRejectModal(ticket); setRejectReason('') }}
                      actioning={actionLoading === ticket.ticket_id}
                    />
                  ))}
                </div>
              )}

              {/* Resolved history */}
              {resolved.length > 0 && (
                <div className={styles.resolvedSection}>
                  <div className={styles.sectionDivider}>
                    <span>Recently Resolved ({resolved.length})</span>
                  </div>
                  <div className={styles.resolvedGrid}>
                    {resolved.slice(0, 6).map(ticket => (
                      <ResolvedCard key={ticket.ticket_id} ticket={ticket} />
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ═══ TAB B: PRODUCT ANALYTICS ════════════════════════════════════ */}
          {activeTab === 'products' && (
            <motion.div key="products" className={styles.panel}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
            >
              <SectionHeader
                icon={<Package size={18} />}
                title="Product Analytics"
                subtitle="Complaint volume per product/category — ⚠️ Alert triggered when ≥3 complaints on the same product"
                color="#f59e0b"
              />

              {products.alert_count > 0 && (
                <motion.div
                  className={styles.alertBanner}
                  animate={{ opacity: [0.85, 1, 0.85] }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  <AlertTriangle size={16} />
                  <span>
                    <strong>{products.alert_count} product{products.alert_count > 1 ? 's' : ''}</strong> flagged with high complaint volume — review urgently!
                  </span>
                </motion.div>
              )}

              {!products.products?.length ? (
                <EmptyState icon={<Package size={40} color="#f59e0b" />}
                  message="No product data yet" sub="Complaints will appear here as customers submit them." />
              ) : (
                <div className={styles.productTable}>
                  <div className={styles.tableHeader}>
                    <span>Product / Category</span>
                    <span>Complaints</span>
                    <span>Pending Review</span>
                    <span>Last Reported</span>
                    <span>Status</span>
                  </div>
                  {products.products.map((p, i) => (
                    <motion.div
                      key={p.category}
                      className={`${styles.tableRow} ${p.has_alert ? styles.tableRowAlert : ''}`}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                    >
                      <span className={styles.productName}>
                        {p.has_alert && <AlertCircle size={13} color="#f59e0b" />}
                        {p.category}
                      </span>
                      <span className={styles.countBadge} style={{
                        background: p.total_complaints >= 5 ? 'rgba(239,68,68,0.15)' :
                                    p.total_complaints >= 3 ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.1)',
                        color:      p.total_complaints >= 5 ? '#f87171' :
                                    p.total_complaints >= 3 ? '#fbbf24' : '#34d399',
                      }}>
                        {p.total_complaints}
                      </span>
                      <span className={styles.tableCell}>
                        {p.pending_approvals > 0
                          ? <span className={styles.pendingPill}>{p.pending_approvals} pending</span>
                          : <span className={styles.clearPill}>none</span>
                        }
                      </span>
                      <span className={styles.tableCell} style={{ color: '#64748b', fontSize: 12 }}>
                        {p.last_reported ? new Date(p.last_reported).toLocaleDateString() : '—'}
                      </span>
                      <span>
                        {p.has_alert
                          ? <span className={styles.alertPill}><AlertTriangle size={11} /> Alert</span>
                          : <span className={styles.okPill}><CheckCircle2 size={11} /> Normal</span>
                        }
                      </span>
                    </motion.div>
                  ))}
                </div>
              )}

              {/* Bar chart */}
              {products.products?.length > 0 && (
                <div className={styles.barChartSection}>
                  <div className={styles.barChartTitle}>Complaint Volume by Category</div>
                  <div className={styles.barChart}>
                    {products.products.slice(0, 8).map((p, i) => {
                      const max = products.products[0]?.total_complaints || 1
                      const pct = Math.max(4, (p.total_complaints / max) * 100)
                      return (
                        <div key={p.category} className={styles.barItem}>
                          <div className={styles.barLabel}>{p.category}</div>
                          <div className={styles.barTrack}>
                            <motion.div
                              className={styles.barFill}
                              style={{
                                background: p.has_alert
                                  ? 'linear-gradient(90deg,#ef4444,#f97316)'
                                  : 'linear-gradient(90deg,#6366f1,#8b5cf6)',
                              }}
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ delay: i * 0.06, duration: 0.6, ease: 'easeOut' }}
                            />
                          </div>
                          <div className={styles.barCount}>{p.total_complaints}</div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ═══ TAB C: AI RESOLUTION SUMMARY ════════════════════════════════ */}
          {activeTab === 'ai' && (
            <motion.div key="ai" className={styles.panel}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
            >
              <SectionHeader
                icon={<Bot size={18} />}
                title="AI Resolution Summary"
                subtitle="How many issues the AI handled, escalated, and its quality metrics"
                color="#6366f1"
              />

              {!aiSummary ? (
                <EmptyState icon={<Bot size={40} color="#6366f1" />}
                  message="No AI data yet" sub="Process some customer queries to see AI metrics here." />
              ) : (
                <>
                  {/* KPI row — prefer real chatStats (customer-feedback driven) over aiSummary fallback */}
                  <div className={styles.kpiRow}>
                    <KpiCard
                      label="Total Conversations"
                      value={chatStats?.total_conversations ?? aiSummary.overview?.total_conversations ?? 0}
                      icon={<Activity size={20} />}
                      color="#6366f1"
                    />
                    <KpiCard
                      label="AI Resolved ✅"
                      value={chatStats?.ai_resolved ?? aiSummary.overview?.ai_resolved ?? 0}
                      icon={<CheckCircle2 size={20} />}
                      color="#10b981"
                      sub="Customer-confirmed resolutions"
                    />
                    <KpiCard
                      label="Not Resolved ❌"
                      value={chatStats?.ai_unresolved ?? 0}
                      icon={<XCircle size={20} />}
                      color="#ef4444"
                      sub="Customers said 'No' in feedback"
                    />
                    <KpiCard
                      label="Escalated to Human"
                      value={aiSummary.overview?.escalated_to_human ?? 0}
                      icon={<AlertTriangle size={20} />}
                      color="#f59e0b"
                    />
                    <KpiCard
                      label="Resolved by Admin"
                      value={aiSummary.overview?.resolved_by_admin ?? 0}
                      icon={<TicketCheck size={20} />}
                      color="#8b5cf6"
                    />
                  </div>

                  {/* Quality + FCR + Rating Distribution */}
                  <div className={styles.aiRow2}>
                    <div className={styles.qualityCard}>
                      <div className={styles.cardTitle}>Quality Metrics</div>
                      <div className={styles.qualityGrid}>
                        <QualityMetric label="Avg QA Score" value={`${aiSummary.quality?.avg_qa_score ?? '—'}/10`} color="#6366f1" />
                        <QualityMetric label="Avg Response Time" value={`${aiSummary.quality?.avg_processing_time_s ?? '—'}s`} color="#10b981" />
                        <QualityMetric label="Cache Hit Rate" value={`${aiSummary.quality?.cache_hit_rate ?? 0}%`} color="#f59e0b" />
                        <QualityMetric label="Avg Rating" value={chatStats?.avg_rating ? `${chatStats.avg_rating}/5 ★` : '—'} color="#f59e0b" />
                        <QualityMetric
                          label="Resolution Rate"
                          value={(() => {
                            const res = chatStats?.ai_resolved ?? 0
                            const unr = chatStats?.ai_unresolved ?? 0
                            const total = res + unr
                            return total > 0 ? `${Math.round(res / total * 100)}%` : '—'
                          })()}
                          color="#10b981"
                        />
                        <QualityMetric label="Total Feedback" value={chatStats?.total_feedback ?? 0} color="#6366f1" />
                      </div>

                      {/* Star rating distribution */}
                      {chatStats?.rating_distribution && (
                        <div style={{ marginTop: '20px' }}>
                          <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Rating Distribution (Customer Feedback)
                          </div>
                          {[5, 4, 3, 2, 1].map(star => {
                            const count = chatStats.rating_distribution[String(star)] ?? 0
                            const total = chatStats.total_feedback || 1
                            const pct = Math.round(count / total * 100)
                            return (
                              <div key={star} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                                <span style={{ fontSize: '12px', color: '#f59e0b', width: '16px', textAlign: 'right' }}>{star}★</span>
                                <div style={{ flex: 1, height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${pct}%` }}
                                    transition={{ duration: 0.6, ease: 'easeOut' }}
                                    style={{
                                      height: '100%', borderRadius: '4px',
                                      background: star >= 4 ? 'linear-gradient(90deg,#10b981,#34d399)' : star === 3 ? '#f59e0b' : 'linear-gradient(90deg,#ef4444,#f97316)',
                                    }}
                                  />
                                </div>
                                <span style={{ fontSize: '12px', color: '#64748b', width: '36px' }}>{count} ({pct}%)</span>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>

                    <div className={styles.fcrCard}>
                      <div className={styles.cardTitle}>Resolution Verdicts (FCR)</div>
                      <FCRDonut data={aiSummary.fcr_distribution || {}} />
                    </div>
                  </div>
                </>
              )}
            </motion.div>
          )}

          {/* ═══ TAB D: CUSTOMER FEEDBACK ════════════════════════════════════ */}
          {activeTab === 'feedback' && (
            <motion.div key="feedback" className={styles.panel}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
            >
              <SectionHeader
                icon={<MessageSquare size={18} />}
                title="Customer Feedback"
                subtitle="Customer satisfaction ratings and comments — use this to improve the system"
                color="#10b981"
              />

              {/* Summary pills — prefer chatStats */}
              <div className={styles.feedbackSummary}>
                <div className={styles.summaryPill} style={{ borderColor: '#6366f1' }}>
                  <Star size={14} color="#6366f1" fill="#6366f1" />
                  <span>Avg: <strong>{chatStats?.avg_rating ?? feedback.summary?.avg_rating ?? '—'}/5</strong></span>
                </div>
                <div className={styles.summaryPill} style={{ borderColor: '#10b981' }}>
                  <CheckCircle2 size={14} color="#10b981" />
                  <span>Resolved: <strong>{chatStats?.ai_resolved ?? 0}</strong></span>
                </div>
                <div className={styles.summaryPill} style={{ borderColor: '#ef4444' }}>
                  <XCircle size={14} color="#ef4444" />
                  <span>Unresolved: <strong>{chatStats?.ai_unresolved ?? 0}</strong></span>
                </div>
                <div className={styles.summaryPill} style={{ borderColor: '#64748b' }}>
                  <ThumbsUp size={14} color="#64748b" />
                  <span>Total responses: <strong>{chatStats?.total_feedback ?? 0}</strong></span>
                </div>
              </div>

              {/* Filter */}
              <div className={styles.filterRow}>
                <Filter size={13} color="#64748b" />
                {['all', 'low_rated', 'escalated'].map(f => (
                  <button
                    key={f}
                    className={`${styles.filterBtn} ${feedbackFilter === f ? styles.filterActive : ''}`}
                    onClick={() => setFeedbackFilter(f)}
                  >
                    {f === 'all' ? 'All' : f === 'low_rated' ? '⚠️ Low Rated' : '🚨 Escalated'}
                  </button>
                ))}
              </div>

              {/* Use feedbackList (real data) if available, fall back to aiSummary feed */}
              {(() => {
                const items = feedbackList.length > 0 ? feedbackList : (feedback.feedback || [])
                return items.length === 0 ? (
                  <EmptyState icon={<MessageSquare size={40} color="#10b981" />}
                    message="No feedback yet" sub="Customer ratings will appear here after interactions." />
                ) : (
                  <div className={styles.feedbackList}>
                    {items.slice(0, 50).map((fb, i) => (
                      <motion.div
                        key={fb.session_id || i}
                        className={styles.feedbackCard}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04 }}
                      >
                        <div className={styles.feedbackTop}>
                          <div className={styles.stars}>
                            {[1,2,3,4,5].map(s => (
                              <Star
                                key={s} size={14}
                                fill={s <= (fb.rating || 0) ? '#f59e0b' : 'transparent'}
                                color={s <= (fb.rating || 0) ? '#f59e0b' : '#334155'}
                              />
                            ))}
                            <span className={styles.ratingNum}>{fb.rating}/5</span>
                          </div>
                          <span className={styles.feedbackMeta}>
                            {fb.session_id?.slice(0, 8) ?? '—'} ·{' '}
                            {fb.timestamp ? new Date(fb.timestamp).toLocaleString() : '—'}
                          </span>
                        </div>

                        {/* Query resolved status */}
                        {fb.query_resolved !== undefined && fb.query_resolved !== null && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: '6px 0', fontSize: '12px' }}>
                            <span style={{ color: '#64748b' }}>Query resolved:</span>
                            {fb.query_resolved === true
                              ? <span style={{ color: '#34d399', fontWeight: 600 }}>✅ Yes</span>
                              : <span style={{ color: '#f87171', fontWeight: 600 }}>❌ No</span>
                            }
                          </div>
                        )}

                        {fb.comment && <p className={styles.feedbackComment}>"{fb.comment}"</p>}
                        <div className={styles.feedbackTags}>
                          {fb.query_resolved === false && <span className={styles.tagRed}>Unresolved</span>}
                          {fb.query_resolved === true && <span className={styles.tagGreen}>Resolved ✓</span>}
                          {fb.rating <= 2 && <span className={styles.tagRed}>Low satisfaction</span>}
                          {fb.rating >= 4 && !fb.query_resolved && <span className={styles.tagGreen}>Satisfied</span>}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )
              })()}
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* ── Reject Modal ─────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {rejectModal && (
          <motion.div
            className={styles.modalOverlay}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => setRejectModal(null)}
          >
            <motion.div
              className={styles.modal}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
            >
              <div className={styles.modalHeader}>
                <XCircle size={22} color="#ef4444" />
                <h3>Reject Ticket {rejectModal.ticket_id}</h3>
              </div>
              <p className={styles.modalSub}>
                Rejecting: <em>"{rejectModal.query?.slice(0, 80)}…"</em>
              </p>
              <textarea
                id="reject-reason-input"
                className={styles.modalTextarea}
                placeholder="Reason for rejection (required)…"
                value={rejectReason}
                onChange={e => setRejectReason(e.target.value)}
                rows={3}
                autoFocus
              />
              <div className={styles.modalActions}>
                <button className={styles.cancelBtn} onClick={() => setRejectModal(null)}>Cancel</button>
                <button
                  id="confirm-reject-btn"
                  className={styles.rejectConfirmBtn}
                  onClick={rejectTicket}
                  disabled={!rejectReason.trim() || actionLoading}
                >
                  {actionLoading ? <Loader2 size={14} className={styles.spin} /> : <XCircle size={14} />}
                  Confirm Rejection
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}


// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionHeader({ icon, title, subtitle, color }) {
  return (
    <div className={styles.sectionHeader}>
      <div className={styles.sectionIcon} style={{ background: `${color}18`, color }}>
        {icon}
      </div>
      <div>
        <h2 className={styles.sectionTitle}>{title}</h2>
        <p className={styles.sectionSub}>{subtitle}</p>
      </div>
    </div>
  )
}

function EmptyState({ icon, message, sub }) {
  return (
    <div className={styles.emptyState}>
      {icon}
      <p className={styles.emptyMsg}>{message}</p>
      <p className={styles.emptySub}>{sub}</p>
    </div>
  )
}

function TicketCard({ ticket, onApprove, onReject, actioning }) {
  const [expanded, setExpanded] = useState(false)
  const urgencyColor = {
    Critical: '#ef4444', High: '#f97316', Medium: '#f59e0b', Low: '#10b981'
  }[ticket.urgency] || '#6366f1'

  return (
    <motion.div
      className={styles.ticketCard}
      style={{ borderColor: urgencyColor + '40' }}
      layout
    >
      {/* Ticket header */}
      <div className={styles.ticketTop}>
        <div className={styles.ticketId}>
          <DollarSign size={14} color="#ef4444" />
          {ticket.ticket_id}
        </div>
        <div className={styles.ticketMeta}>
          <span className={styles.urgencyBadge} style={{ background: urgencyColor + '20', color: urgencyColor }}>
            {ticket.urgency}
          </span>
          <span className={styles.categoryBadge}>{ticket.category}</span>
        </div>
      </div>

      {/* Issue summary */}
      <p className={styles.ticketQuery}>"{ticket.query}"</p>

      {/* AI response preview */}
      <div className={styles.aiResponsePreview}>
        <div className={styles.aiLabel}><Bot size={11} /> AI Response</div>
        <p className={styles.aiText}>{ticket.ai_response?.slice(0, 160)}…</p>
      </div>

      {/* Expand for details */}
      <button className={styles.expandBtn} onClick={() => setExpanded(v => !v)}>
        {expanded ? 'Show less' : 'Show full details'} <ChevronRight size={13} style={{ transform: expanded ? 'rotate(90deg)' : '', transition: '0.2s' }} />
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            className={styles.expandedDetails}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <Detail label="User ID" value={ticket.user_id} />
            <Detail label="Session ID" value={ticket.session_id?.slice(0, 12) + '…'} />
            <Detail label="Trace ID" value={ticket.trace_id?.slice(0, 12) + '…'} />
            <Detail label="Timestamp" value={new Date(ticket.timestamp).toLocaleString()} />
            {ticket.ai_recommendation && <Detail label="AI Recommendation" value={ticket.ai_recommendation} />}
            <div className={styles.fullResponse}>
              <div className={styles.aiLabel}>Full AI Response</div>
              <p>{ticket.ai_response}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions */}
      <div className={styles.ticketActions}>
        <button
          id={`reject-${ticket.ticket_id}`}
          className={styles.rejectBtn}
          onClick={onReject}
          disabled={actioning}
        >
          <XCircle size={14} /> Reject
        </button>
        <button
          id={`approve-${ticket.ticket_id}`}
          className={styles.approveBtn}
          onClick={onApprove}
          disabled={actioning}
        >
          {actioning
            ? <Loader2 size={14} className={styles.spin} />
            : <CheckCircle2 size={14} />
          }
          Approve
        </button>
      </div>

      <p className={styles.ticketDisclaimer}>
        ⚖️ This action will be logged with your admin email and cannot be undone.
      </p>
    </motion.div>
  )
}

function ResolvedCard({ ticket }) {
  const isApproved = ticket.decision === 'APPROVED'
  return (
    <div className={`${styles.resolvedCard} ${isApproved ? styles.resolvedApproved : styles.resolvedRejected}`}>
      {isApproved
        ? <CheckCircle2 size={14} color="#10b981" />
        : <XCircle size={14} color="#ef4444" />
      }
      <div className={styles.resolvedInfo}>
        <span className={styles.resolvedId}>{ticket.ticket_id}</span>
        <span className={styles.resolvedStatus}>{ticket.decision}</span>
        {ticket.rejection_reason && <span className={styles.resolvedReason}>"{ticket.rejection_reason}"</span>}
        <span className={styles.resolvedBy}>by {ticket.actioned_by} · {ticket.actioned_at ? new Date(ticket.actioned_at).toLocaleString() : '—'}</span>
      </div>
    </div>
  )
}

function KpiCard({ label, value, icon, color, sub }) {
  return (
    <div className={styles.kpiCard}>
      <div className={styles.kpiIcon} style={{ background: `${color}18`, color }}>{icon}</div>
      <div className={styles.kpiValue}>{value}</div>
      <div className={styles.kpiLabel}>{label}</div>
      {sub && <div className={styles.kpiSub}>{sub}</div>}
    </div>
  )
}

function QualityMetric({ label, value, color }) {
  return (
    <div className={styles.qualityItem}>
      <div className={styles.qualityValue} style={{ color }}>{value}</div>
      <div className={styles.qualityLabel}>{label}</div>
    </div>
  )
}

function FCRDonut({ data }) {
  const colors = {
    'Fully Resolved':    '#10b981',
    'Partially Resolved':'#f59e0b',
    'Not Resolved':      '#ef4444',
  }
  const total = Object.values(data).reduce((a, b) => a + b, 0) || 1

  return (
    <div className={styles.fcrList}>
      {Object.entries(data).map(([verdict, count]) => {
        const pct = Math.round((count / total) * 100)
        return (
          <div key={verdict} className={styles.fcrItem}>
            <div className={styles.fcrLabelRow}>
              <span className={styles.fcrDot} style={{ background: colors[verdict] || '#6366f1' }} />
              <span className={styles.fcrLabel}>{verdict}</span>
              <span className={styles.fcrCount}>{count}</span>
            </div>
            <div className={styles.fcrBar}>
              <motion.div
                className={styles.fcrFill}
                style={{ background: colors[verdict] || '#6366f1' }}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.7, ease: 'easeOut' }}
              />
            </div>
            <span className={styles.fcrPct}>{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div className={styles.detailRow}>
      <span className={styles.detailKey}>{label}</span>
      <span className={styles.detailVal}>{value}</span>
    </div>
  )
}
