// The single typed client for the FastAPI backend. Every backend call goes through here —
// no `fetch` scattered in components (see frontend/CLAUDE.md § Backend contract).
//
// Identity: the backend derives the user from the verified Supabase token, so every call
// carries `Authorization: Bearer <access_token>`. Server Components pass the token from the
// session; the client stream reader passes the browser session token.

const API_URL = process.env.NEXT_PUBLIC_API_URL

export type Citation = {
  n: number
  source: string
  snippet?: string
  score?: number | null
}

export type ChatMessage = {
  role: "user" | "assistant"
  content: string
  citations?: Citation[]
}

export type ConversationSummary = {
  id: string
  title: string | null
  updated_at: string
}

export type DocumentRow = {
  id: string
  filename: string
  file_type: string | null
  file_size: number | null
  num_chunks: number | null
  status: "pending" | "success" | "failed"
  ingestion_time_s: number | null
  error: string | null
  created_at: string
}

function baseUrl(): string {
  if (!API_URL) throw new Error("NEXT_PUBLIC_API_URL is not set")
  return API_URL.replace(/\/$/, "")
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` }
}

/** Server-side read: the current user's conversation list (sidebar). */
export async function listConversations(token: string): Promise<ConversationSummary[]> {
  const res = await fetch(`${baseUrl()}/conversations`, {
    headers: authHeaders(token),
    cache: "no-store",
  })
  if (!res.ok) throw new Error(`listConversations: ${res.status}`)
  const data = (await res.json()) as { conversations: ConversationSummary[] }
  return data.conversations ?? []
}

/** Server-side read: full turns for one thread the user owns. Null if 404 (missing/not theirs). */
export async function getConversation(
  id: string,
  token: string,
): Promise<ChatMessage[] | null> {
  const res = await fetch(`${baseUrl()}/conversations/${id}`, {
    headers: authHeaders(token),
    cache: "no-store",
  })
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`getConversation: ${res.status}`)
  const data = (await res.json()) as { messages: ChatMessage[] }
  return data.messages ?? []
}

/** Rename a conversation the user owns. */
export async function renameConversation(id: string, title: string, token: string): Promise<void> {
  const res = await fetch(`${baseUrl()}/conversations/${id}`, {
    method: "PATCH",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  })
  if (!res.ok) throw new Error(`renameConversation: ${res.status}`)
}

/** Delete a conversation the user owns (its messages cascade). */
export async function deleteConversation(id: string, token: string): Promise<void> {
  const res = await fetch(`${baseUrl()}/conversations/${id}`, {
    method: "DELETE",
    headers: authHeaders(token),
  })
  if (!res.ok) throw new Error(`deleteConversation: ${res.status}`)
}

/** Server-side read: the current user's ingested documents (Ingest tab table). */
export async function listDocuments(token: string): Promise<DocumentRow[]> {
  const res = await fetch(`${baseUrl()}/documents`, {
    headers: authHeaders(token),
    cache: "no-store",
  })
  if (!res.ok) throw new Error(`listDocuments: ${res.status}`)
  const data = (await res.json()) as { documents: DocumentRow[] }
  return data.documents ?? []
}

/**
 * Client-side upload: POST one file to /ingest as multipart. Returns the saved document row.
 * Surfaces the backend's error detail (e.g. unsupported type, too large) so the UI can toast it.
 */
export async function ingestDocument(
  file: File,
  token: string,
): Promise<{ document: DocumentRow | null; num_chunks: number }> {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${baseUrl()}/ingest`, {
    method: "POST",
    headers: authHeaders(token), // no Content-Type — the browser sets the multipart boundary
    body: form,
  })
  if (!res.ok) {
    let detail = `Upload failed (${res.status})`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // non-JSON error body — keep the generic message
    }
    throw new Error(detail)
  }
  return (await res.json()) as { document: DocumentRow | null; num_chunks: number }
}

export type StreamCallbacks = {
  onConversation?: (conversationId: string) => void
  onToken?: (token: string) => void
  onCitations?: (citations: Citation[]) => void
  onDone?: () => void
  onError?: (message: string) => void
}

/**
 * Client-side chat stream. POSTs to /stream and consumes the SSE body with a stream reader,
 * dispatching typed events. `signal` lets the caller cancel (unmount / stop button).
 */
export async function streamChat(
  params: { question: string; conversationId?: string; token: string; signal?: AbortSignal },
  cb: StreamCallbacks,
): Promise<void> {
  const res = await fetch(`${baseUrl()}/stream`, {
    method: "POST",
    headers: { ...authHeaders(params.token), "Content-Type": "application/json" },
    body: JSON.stringify({
      question: params.question,
      conversation_id: params.conversationId ?? null,
    }),
    signal: params.signal,
  })

  if (!res.ok || !res.body) {
    cb.onError?.(`stream failed (${res.status})`)
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  // Parse SSE frames: blocks separated by a blank line, each with `event:` and `data:` lines.
  const dispatch = (frame: string) => {
    let event = "message"
    const dataLines: string[] = []
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim()
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim())
    }
    if (dataLines.length === 0) return
    let data: unknown
    try {
      data = JSON.parse(dataLines.join("\n"))
    } catch {
      return
    }
    switch (event) {
      case "conversation":
        cb.onConversation?.((data as { conversation_id: string }).conversation_id)
        break
      case "token":
        cb.onToken?.(data as string)
        break
      case "citations":
        cb.onCitations?.(data as Citation[])
        break
      case "done":
        cb.onDone?.()
        break
      case "error":
        cb.onError?.(typeof data === "string" ? data : "stream error")
        break
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      if (frame.trim()) dispatch(frame)
    }
  }
  if (buffer.trim()) dispatch(buffer)
}
