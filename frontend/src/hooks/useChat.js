import { useState, useRef, useCallback } from 'react'

const API = ''  // empty = uses Vite proxy

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [pipelineNodes, setPipelineNodes] = useState([])
  const [lastMeta, setLastMeta] = useState(null)
  const userId = useRef(`user_${Math.random().toString(36).slice(2, 10)}`)

  const NODES = [
    { id: 'guardrails',              label: 'Guardrails',           icon: '🛡️', parallel: false },
    { id: 'classify_ticket',         label: 'Ticket Classifier',    icon: '🏷️', parallel: false },
    { id: 'manage_session',          label: 'Session Manager',      icon: '👤', parallel: false },
    { id: 'parallel_retrieval',      label: 'Parallel Retrieval',   icon: '⚡', parallel: true  },
    { id: 'fuse_information',        label: 'Info Fusion',          icon: '🔗', parallel: false },
    { id: 'generate_solution',       label: 'Solution Generator',   icon: '💡', parallel: false },
    { id: 'personalize_response',    label: 'Response Writer',      icon: '✍️', parallel: false },
    { id: 'qa_review',               label: 'QA Review',            icon: '⭐', parallel: false, conditional: true },
    { id: 'persist_conversation',    label: 'Save to ChromaDB',     icon: '💾', parallel: false },
    { id: 'escalation_coordinator',  label: 'Escalation Check',     icon: '🚨', parallel: true  },
    { id: 'cx_optimizer',            label: 'CX Optimizer',         icon: '📊', parallel: true  },
    { id: 'evaluation',              label: 'FCR Evaluation',       icon: '🔍', parallel: false },
  ]

  const resetNodes = () => setPipelineNodes(NODES.map(n => ({ ...n, status: 'idle' })))

  const sendMessage = useCallback(async (query, imageBase64 = null, imageUrl = null) => {
    if (!query.trim() || isLoading) return

    // Add user message
    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'user',
      text: query,
      imageUrl,
      timestamp: new Date(),
    }])

    setIsLoading(true)
    resetNodes()

    // Add typing indicator
    const typingId = Date.now() + 1
    setMessages(prev => [...prev, { id: typingId, role: 'typing' }])

    try {
      const body = {
        query: imageBase64
          ? `${query}\n\n[Customer attached a product image for context]`
          : query,
        user_id: userId.current,
        session_id: sessionId,
        image_base64: imageBase64 || undefined,
      }

      const res = await fetch(`${API}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!res.ok) {
        throw new Error(`Server error ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalData = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const chunk = JSON.parse(line.slice(6))
            // Update pipeline node status
            if (chunk.node && chunk.node !== 'start' && chunk.node !== 'final') {
              setPipelineNodes(prev => prev.map(n =>
                n.id === chunk.node ? { ...n, status: chunk.status } : n
              ))
            }
            if (chunk.node === 'final' && chunk.status === 'completed') {
              finalData = chunk.partial_data
            }
          } catch (_) {}
        }
      }

      // Remove typing indicator, add bot response
      setMessages(prev => prev.filter(m => m.id !== typingId))

      if (finalData) {
        if (finalData.session_id) setSessionId(finalData.session_id)
        setLastMeta(finalData)
        setMessages(prev => [...prev, {
          id: Date.now(),
          role: 'bot',
          text: finalData.response || 'Sorry, I could not generate a response.',
          meta: finalData,
          timestamp: new Date(),
        }])
        setPipelineNodes(prev => prev.map(n => ({ ...n, status: n.status === 'running' ? 'done' : n.status === 'idle' ? 'idle' : n.status })))
      }

    } catch (err) {
      setMessages(prev => prev.filter(m => m.id !== typingId))
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'bot',
        text: `⚠️ Connection error: ${err.message}. Make sure the backend is running on port 8000.`,
        meta: {},
        timestamp: new Date(),
      }])
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, sessionId])

  const uploadImage = useCallback(async (file) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${API}/chat/upload-image`, { method: 'POST', body: form })
    if (!res.ok) throw new Error('Image upload failed')
    return await res.json()
  }, [])

  const submitFeedback = useCallback(async (rating, comment, escalated = false) => {
    if (!sessionId) return
    try {
      await fetch(`${API}/chat/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          rating,
          comments: comment,
          escalated
        })
      })
    } catch (err) {
      console.error('Failed to submit feedback', err)
    }
  }, [sessionId])

  return {
    messages,
    isLoading,
    pipelineNodes: pipelineNodes.length ? pipelineNodes : NODES.map(n => ({ ...n, status: 'idle' })),
    lastMeta,
    userId: userId.current,
    sessionId,
    sendMessage,
    uploadImage,
    submitFeedback,
    clearChat: () => { setMessages([]); setSessionId(null); setLastMeta(null); resetNodes() },
  }
}
