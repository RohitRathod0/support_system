import { motion, AnimatePresence } from 'framer-motion'
import styles from './PipelinePanel.module.css'

const statusColor = {
  idle: '#2d2d3a',
  running: '#6366f1',
  completed: '#10b981',
  done: '#10b981',
  error: '#ef4444',
}

export default function PipelinePanel({ nodes, lastMeta }) {
  const fcr = lastMeta?.evaluation

  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span className={styles.titleDot} />
        Live Pipeline
      </div>

      <div className={styles.nodes}>
        {nodes.map((node, i) => (
          <div key={node.id}>
            <motion.div
              className={`${styles.node} ${node.parallel ? styles.parallel : ''}`}
              animate={{
                borderColor: statusColor[node.status] ?? '#2d2d3a',
                backgroundColor: node.status === 'running'
                  ? 'rgba(99,102,241,0.10)'
                  : node.status === 'completed' || node.status === 'done'
                  ? 'rgba(16,185,129,0.07)'
                  : node.status === 'error'
                  ? 'rgba(239,68,68,0.07)'
                  : 'rgba(255,255,255,0.02)',
              }}
              transition={{ duration: 0.3 }}
            >
              <span className={styles.nodeIcon}>{node.icon}</span>
              <div className={styles.nodeInfo}>
                <div className={styles.nodeName}>{node.label}</div>
                <div className={styles.nodeStatus}>
                  {node.status === 'running' && '⟳ Processing…'}
                  {(node.status === 'done' || node.status === 'completed') && '✓ Done'}
                  {node.status === 'idle' && '—'}
                  {node.status === 'error' && '✗ Error'}
                </div>
              </div>
              <div className={styles.badges}>
                {node.parallel && <span className={styles.badge} style={{ background: 'rgba(59,130,246,0.2)', color: '#60a5fa' }}>parallel</span>}
                {node.conditional && <span className={styles.badge} style={{ background: 'rgba(245,158,11,0.2)', color: '#f59e0b' }}>retry</span>}
              </div>
              {node.status === 'running' && (
                <motion.div
                  className={styles.runningBar}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: 2, ease: 'linear' }}
                />
              )}
            </motion.div>
            {i < nodes.length - 1 && (
              <div className={styles.connector}
                style={{ background: node.status === 'done' || node.status === 'completed' ? '#10b981' : '#2d2d3a' }}
              />
            )}
          </div>
        ))}
      </div>

      {/* FCR Score */}
      <AnimatePresence>
        {fcr && (
          <motion.div
            className={styles.fcrCard}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className={styles.fcrTitle}>🎯 FCR Score</div>
            <div className={styles.fcrScore}>{fcr.overall_fcr_score?.toFixed(1) ?? '—'}</div>
            <div className={`${styles.fcrVerdict} ${
              fcr.verdict === 'Fully Resolved' ? styles.verdictGreen :
              fcr.verdict === 'Partially Resolved' ? styles.verdictYellow : styles.verdictRed
            }`}>{fcr.verdict}</div>
            {fcr.recommendation && (
              <div className={styles.fcrNote}>{fcr.recommendation}</div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Node timings */}
      {lastMeta?.node_timings && (
        <div className={styles.timings}>
          <div className={styles.timingsTitle}>⏱ Node Timings</div>
          {Object.entries(lastMeta.node_timings).map(([k, v]) => (
            <div key={k} className={styles.timingRow}>
              <span className={styles.timingNode}>{k.replace(/_/g, ' ')}</span>
              <span className={styles.timingVal}>{v}s</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
