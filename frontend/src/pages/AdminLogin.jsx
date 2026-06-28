import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Shield, Mail, KeyRound, ArrowRight, Loader2, CheckCircle2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './AdminLogin.module.css'

const API = ''

export default function AdminLogin() {
  const navigate = useNavigate()
  const [step, setStep]       = useState('email')   // 'email' | 'otp' | 'done'
  const [email, setEmail]     = useState('')
  const [otp, setOtp]         = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [shake, setShake]     = useState(false)

  const triggerShake = () => {
    setShake(true)
    setTimeout(() => setShake(false), 500)
  }

  const sendOTP = async (e) => {
    e.preventDefault()
    if (!email.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API}/api/admin/auth/send-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim() }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Failed to send OTP')
      setStep('otp')
    } catch (err) {
      setError(err.message)
      triggerShake()
    } finally {
      setLoading(false)
    }
  }

  const verifyOTP = async (e) => {
    e.preventDefault()
    if (otp.length !== 6) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API}/api/admin/auth/verify-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), code: otp.trim() }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Invalid OTP')
      const data = await res.json()
      localStorage.setItem('admin_token', data.token)
      localStorage.setItem('admin_email', data.email)
      setStep('done')
      setTimeout(() => navigate('/admin'), 800)
    } catch (err) {
      setError(err.message)
      triggerShake()
      setOtp('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      {/* Animated background orbs */}
      <div className={styles.orb1} />
      <div className={styles.orb2} />
      <div className={styles.orb3} />

      <motion.div
        className={`${styles.card} ${shake ? styles.shake : ''}`}
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      >
        {/* Header */}
        <div className={styles.cardHeader}>
          <div className={styles.shieldIcon}>
            <Shield size={28} />
          </div>
          <h1 className={styles.title}>Admin Access</h1>
          <p className={styles.subtitle}>Human-in-the-Loop Control Center</p>
        </div>

        <AnimatePresence mode="wait">

          {/* Step 1: Email */}
          {step === 'email' && (
            <motion.form
              key="email-step"
              onSubmit={sendOTP}
              className={styles.form}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              <div className={styles.stepLabel}>
                <span className={styles.stepDot} />
                Step 1 — Enter your admin email
              </div>
              <div className={styles.inputGroup}>
                <Mail size={16} className={styles.inputIcon} />
                <input
                  id="admin-email-input"
                  className={styles.input}
                  type="email"
                  placeholder="admin@yourcompany.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  autoFocus
                  required
                />
              </div>
              {error && <p className={styles.error}>{error}</p>}
              <button
                id="send-otp-btn"
                className={styles.btn}
                type="submit"
                disabled={loading || !email.trim()}
              >
                {loading
                  ? <><Loader2 size={16} className={styles.spin} /> Sending code…</>
                  : <><span>Send OTP to Gmail</span> <ArrowRight size={16} /></>
                }
              </button>
              <p className={styles.hint}>
                A 6-digit code will be emailed to your registered Gmail address.
              </p>
            </motion.form>
          )}

          {/* Step 2: OTP */}
          {step === 'otp' && (
            <motion.form
              key="otp-step"
              onSubmit={verifyOTP}
              className={styles.form}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <div className={styles.stepLabel}>
                <span className={styles.stepDot} style={{ background: '#10b981' }} />
                Step 2 — Enter the code sent to {email}
              </div>
              <div className={styles.otpWrapper}>
                <KeyRound size={16} className={styles.inputIcon} />
                <input
                  id="otp-input"
                  className={`${styles.input} ${styles.otpInput}`}
                  type="text"
                  inputMode="numeric"
                  maxLength={6}
                  placeholder="• • • • • •"
                  value={otp}
                  onChange={e => setOtp(e.target.value.replace(/\D/g, ''))}
                  autoFocus
                />
              </div>
              {error && <p className={styles.error}>{error}</p>}
              <button
                id="verify-otp-btn"
                className={styles.btn}
                type="submit"
                disabled={loading || otp.length !== 6}
              >
                {loading
                  ? <><Loader2 size={16} className={styles.spin} /> Verifying…</>
                  : <><span>Verify & Enter Dashboard</span> <ArrowRight size={16} /></>
                }
              </button>
              <button
                type="button"
                className={styles.backLink}
                onClick={() => { setStep('email'); setError(''); setOtp('') }}
              >
                ← Use a different email
              </button>
            </motion.form>
          )}

          {/* Done */}
          {step === 'done' && (
            <motion.div
              key="done-step"
              className={styles.doneState}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, type: 'spring' }}
            >
              <CheckCircle2 size={48} color="#10b981" />
              <p>Authenticated! Redirecting…</p>
            </motion.div>
          )}

        </AnimatePresence>

        {/* Security badge */}
        <div className={styles.securityBadge}>
          🔐 OTP valid for 10 minutes · Session expires in 2 hours
        </div>
      </motion.div>
    </div>
  )
}
