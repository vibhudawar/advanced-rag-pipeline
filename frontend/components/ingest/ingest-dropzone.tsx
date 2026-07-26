"use client"

import { Loader2, UploadCloud } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRef, useState } from "react"
import { toast } from "sonner"

import { ingestDocument } from "@/lib/api"
import { createClient } from "@/lib/supabase/client"
import { cn } from "@/lib/utils"

const ACCEPT = ".pdf,.docx,.txt,.md"
const MAX_MB = 5

// Drag-and-drop / click upload. Uploads each file to /ingest, then refreshes the server-rendered
// documents table. Guardrail hints mirror the backend limits.
export function IngestDropzone() {
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState<string | null>(null)

  async function upload(files: FileList | File[]) {
    const list = Array.from(files)
    if (list.length === 0) return

    const supabase = createClient()
    const { data } = await supabase.auth.getSession()
    const token = data.session?.access_token
    if (!token) {
      toast.error("Your session expired. Please sign in again.")
      return
    }

    for (const file of list) {
      setUploading(file.name)
      try {
        const { num_chunks } = await ingestDocument(file, token)
        toast.success(`Indexed ${file.name} (${num_chunks} chunks)`)
      } catch (err) {
        toast.error(err instanceof Error ? err.message : `Failed to upload ${file.name}`)
      } finally {
        router.refresh() // reflect the new row (success or recorded failure)
      }
    }
    setUploading(null)
    if (inputRef.current) inputRef.current.value = ""
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (!uploading) void upload(e.dataTransfer.files)
      }}
      onClick={() => !uploading && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !uploading) {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-colors outline-none",
        "hover:border-primary/50 hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring",
        dragging && "border-primary bg-accent/60",
        uploading && "pointer-events-none opacity-70",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        className="hidden"
        onChange={(e) => e.target.files && void upload(e.target.files)}
      />
      {uploading ? (
        <>
          <Loader2 className="size-8 animate-spin text-primary" />
          <p className="mt-3 text-sm font-medium">Indexing {uploading}…</p>
          <p className="mt-1 text-xs text-muted-foreground">Parsing, chunking, and embedding.</p>
        </>
      ) : (
        <>
          <UploadCloud className="size-8 text-muted-foreground" />
          <p className="mt-3 text-sm font-medium">
            Drop files here, or <span className="text-primary">browse</span>
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            PDF, DOCX, TXT, or Markdown · up to {MAX_MB} MB each
          </p>
        </>
      )}
    </div>
  )
}
