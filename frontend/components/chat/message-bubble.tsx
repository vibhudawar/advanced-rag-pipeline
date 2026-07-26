"use client"

import type { ChatMessage } from "@/lib/api"
import { cn } from "@/lib/utils"

import { Citations } from "./citations"
import { Markdown } from "./markdown"

// Compact per-answer metrics (model · latency · tokens · cost). Full traces live in LangSmith.
function MetaLine({ meta }: { meta: NonNullable<ChatMessage["meta"]> }) {
  const parts: string[] = []
  if (meta.model) parts.push(meta.model)
  if (typeof meta.latency_ms === "number") parts.push(`${(meta.latency_ms / 1000).toFixed(1)}s`)
  if (meta.total_tokens) parts.push(`${meta.total_tokens.toLocaleString()} tokens`)
  if (meta.cost_usd)
    parts.push(meta.cost_usd < 0.0001 ? "<$0.0001" : `$${meta.cost_usd.toFixed(4)}`)
  if (parts.length === 0) return null
  return <p className="mt-2 text-xs text-muted-foreground/70">{parts.join(" · ")}</p>
}

// One turn. User messages are a right-aligned filled bubble; assistant messages are full-width
// markdown with citations (like ChatGPT). `pending` shows the retrieving shimmer before the
// first token lands.
export function MessageBubble({
  message,
  pending,
}: {
  message: ChatMessage
  pending?: boolean
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className={cn("max-w-full min-w-0")}>
        {pending && !message.content ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="size-2 animate-pulse rounded-full bg-muted-foreground/50" />
            Retrieving…
          </div>
        ) : (
          <Markdown>{message.content}</Markdown>
        )}
        {message.citations && message.citations.length > 0 && (
          <Citations citations={message.citations} />
        )}
        {message.meta && !pending && <MetaLine meta={message.meta} />}
      </div>
    </div>
  )
}
