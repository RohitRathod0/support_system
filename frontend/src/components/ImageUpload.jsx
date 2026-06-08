import { useState, useRef } from 'react'
import { ImagePlus, X, Upload } from 'lucide-react'
import styles from './ImageUpload.module.css'

export default function ImageUpload({ onImageReady, onClear, disabled }) {
  const [preview, setPreview] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

  const processFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setUploading(true)

    // Local preview
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)

    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/chat/upload-image', { method: 'POST', body: form })
      if (!res.ok) throw new Error('Upload failed')
      const data = await res.json()
      onImageReady({ url: data.url, base64: data.full_base64, contentType: file.type })
    } catch (err) {
      console.error('Image upload error:', err)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    processFile(file)
  }

  const clear = () => {
    setPreview(null)
    onClear()
    if (inputRef.current) inputRef.current.value = ''
  }

  if (preview) {
    return (
      <div className={styles.preview}>
        <img src={preview} alt="Complaint attachment" className={styles.previewImg} />
        {uploading && <div className={styles.uploadingOverlay}><span>Uploading…</span></div>}
        <button className={styles.clearBtn} onClick={clear} title="Remove image">
          <X size={14} />
        </button>
        <span className={styles.previewLabel}>📎 Image attached</span>
      </div>
    )
  }

  return (
    <div
      className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''} ${disabled ? styles.disabled : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      title="Attach a product image"
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => processFile(e.target.files[0])}
        disabled={disabled}
      />
      {dragOver
        ? <Upload size={18} className={styles.icon} />
        : <ImagePlus size={18} className={styles.icon} />
      }
    </div>
  )
}
