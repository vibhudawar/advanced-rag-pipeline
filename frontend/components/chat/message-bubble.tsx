"use client"

import type { ChatMessage } from "@/lib/api"
import { cn } from "@/lib/utils"

import { Citations } from "./citations"
import { Markdown } from "./markdown"

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
      </div>
    </div>
  )
}
