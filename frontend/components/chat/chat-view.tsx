"use client"

import { useRouter } from "next/navigation"
import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"

import { streamChat, type ChatMessage } from "@/lib/api"
import { createClient } from "@/lib/supabase/client"

import { Composer } from "./composer"
import { MessageBubble } from "./message-bubble"

// The chat surface (client leaf). Holds the message list, drives the SSE stream, and keeps the
// URL + sidebar in sync when a brand-new thread is created. Keyed by conversationId at the page
// level, so switching threads via the sidebar mounts a fresh view with server-loaded history.
export function ChatView({
  conversationId,
  initialMessages,
}: {
  conversationId?: string
  initialMessages: ChatMessage[]
}) {
  const router = useRouter()
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [streaming, setStreaming] = useState(false)
  const convIdRef = useRef<string | undefined>(conversationId)
  const wasNew = useRef(conversationId === undefined)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => () => abortRef.current?.abort(), [])

  const send = useCallback(
    async (question: string) => {
      const supabase = createClient()
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      if (!token) {
        toast.error("Your session expired. Please sign in again.")
        return
      }

      setMessages((m) => [
        ...m,
        { role: "user", content: question },
        { role: "assistant", content: "" },
      ])
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      const appendToken = (t: string) =>
        setMessages((m) => {
          const next = [...m]
          const last = next[next.length - 1]
          if (last?.role === "assistant") next[next.length - 1] = { ...last, content: last.content + t }
          return next
        })

      const setCitations = (citations: ChatMessage["citations"]) =>
        setMessages((m) => {
          const next = [...m]
          const last = next[next.length - 1]
          if (last?.role === "assistant") next[next.length - 1] = { ...last, citations }
          return next
        })

      const setMeta = (meta: ChatMessage["meta"]) =>
        setMessages((m) => {
          const next = [...m]
          const last = next[next.length - 1]
          if (last?.role === "assistant") next[next.length - 1] = { ...last, meta }
          return next
        })

      await streamChat(
        { question, conversationId: convIdRef.current, token, signal: controller.signal },
        {
          onConversation: (id) => {
            convIdRef.current = id
          },
          onToken: appendToken,
          onCitations: setCitations,
          onMeta: setMeta,
          onError: (msg) => {
            toast.error(msg)
            setMessages((m) => {
              const next = [...m]
              const last = next[next.length - 1]
              if (last?.role === "assistant" && !last.content)
                next[next.length - 1] = { ...last, content: "_Something went wrong. Please try again._" }
              return next
            })
          },
          onDone: () => {},
        },
      ).catch(() => {
        // Aborted (user hit stop / unmounted) — no toast needed.
      })

      setStreaming(false)

      // First message of a brand-new thread: point the URL at the saved conversation and refresh
      // the layout so the sidebar picks it up. Existing threads just bump sidebar ordering.
      if (convIdRef.current) {
        if (wasNew.current) {
          wasNew.current = false
          router.replace(`/?c=${convIdRef.current}`)
        }
        router.refresh()
      }
    },
    [router],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setStreaming(false)
  }, [])

  const empty = messages.length === 0

  return (
    <div className="flex h-full flex-col">
      {empty ? (
        <div className="flex flex-1 items-center justify-center p-6">
          <div className="max-w-md text-center">
            <h1 className="text-2xl font-semibold tracking-tight">What do you want to know?</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Ask a question and get a grounded answer with citations from your indexed documents.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <div
            className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6"
            role="log"
            aria-live="polite"
            aria-label="Conversation"
          >
            {messages.map((msg, i) => (
              <MessageBubble
                key={i}
                message={msg}
                pending={streaming && i === messages.length - 1 && msg.role === "assistant"}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>
      )}
      <Composer onSend={send} onStop={stop} streaming={streaming} />
    </div>
  )
}
