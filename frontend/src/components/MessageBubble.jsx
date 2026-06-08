import { motion } from 'framer-motion'
import styles from './MessageBubble.module.css'

const urgencyColor = { Critical: '#ef4444', High: '#f59e0b', Medium: '#6366f1', Low: '#10b981' }

export default function MessageBubble({ message }) {
  const { role, text, imageUrl, meta, timestamp } = message

  if (role === 'typing') {
    return (
      <motion.div
        className={styles.row}
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
      >
        <div className={styles.avatar}>🤖</div>
        <div className={styles.typingBubble}>
          <span className={styles.dot} />
          <span className={styles.dot} />
          <span className={styles.dot} />
        </div>
      </motion.div>
    )
  }

  const isUser = role === 'user'

  return (
    <motion.div
      className={`${styles.row} ${isUser ? styles.userRow : ''}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      {!isUser && <div className={styles.avatar}>🤖</div>}

      <div className={styles.bubbleWrap}>
        {/* Image attachment (user side) */}
        {imageUrl && (
          <div className={`${styles.imageAttach} ${isUser ? styles.imageUser : ''}`}>
            <img src={imageUrl} alt="Product complaint" />
            <span>📎 Product image attached</span>
          </div>
        )}

        {/* Message bubble */}
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.botBubble}`}>
          <div
            className={styles.text}
            dangerouslySetInnerHTML={{
              __html: text
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                .replace(/\n/g, '<br/>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            }}
          />
        </div>

        {/* Meta chips — bot messages only */}
        {!isUser && meta && (
          <div className={styles.chips}>
            {meta.cache_hit && <Chip color="#10b981">⚡ Cache Hit</Chip>}
            {meta.urgency_level && (
              <Chip color={urgencyColor[meta.urgency_level] || '#6366f1'}>
                {meta.urgency_level}
              </Chip>
            )}
            {meta.issue_category && <Chip color="#06b6d4">{meta.issue_category}</Chip>}
            {meta.qa_score && <Chip color="#f59e0b">⭐ QA {meta.qa_score}/10</Chip>}
            {meta.processing_time_seconds && (
              <Chip color="#475569">⏱ {meta.processing_time_seconds.toFixed(1)}s</Chip>
            )}
            {meta.evaluation?.overall_fcr_score && (
              <Chip color="#6366f1">🎯 FCR {meta.evaluation.overall_fcr_score}/10</Chip>
            )}
            {meta.escalation_needed && <Chip color="#ef4444">🚨 Escalated</Chip>}
          </div>
        )}

        {/* Escalation notice */}
        {!isUser && meta?.escalation_needed && (
          <div className={styles.escalation}>
            🚨 This case has been escalated to our specialist team.
            {meta.escalation_report?.customer_communication && (
              <span> {meta.escalation_report.customer_communication}</span>
            )}
          </div>
        )}

        <div className={styles.time}>
          {timestamp?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>

      {isUser && <div className={`${styles.avatar} ${styles.userAvatar}`}>👤</div>}
    </motion.div>
  )
}

function Chip({ color, children }) {
  return (
    <span className={styles.chip} style={{ '--chip-color': color }}>
      {children}
    </span>
  )
}
