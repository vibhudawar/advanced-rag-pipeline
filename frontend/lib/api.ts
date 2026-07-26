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
