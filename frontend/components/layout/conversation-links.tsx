"use client"

import Link from "next/link"
import { useSearchParams } from "next/navigation"

import type { ConversationSummary } from "@/lib/api"
import { cn } from "@/lib/utils"

// Renders the thread list and highlights the active one (from `?c=<id>`). Client-only because
// it reads the current search params.
export function ConversationLinks({ conversations }: { conversations: ConversationSummary[] }) {
  const active = useSearchParams().get("c")

  if (conversations.length === 0) {
    return <p className="px-3 py-2 text-sm text-muted-foreground">No conversations yet.</p>
  }

  return (
    <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
      {conversations.map((c) => (
        <Link
          key={c.id}
          href={`/?c=${c.id}`}
          title={c.title ?? "Untitled"}
          className={cn(
            "block truncate rounded-md px-2 py-2 text-sm transition-colors",
            "hover:bg-accent hover:text-accent-foreground",
            active === c.id && "bg-accent font-medium text-accent-foreground",
          )}
        >
          {c.title || "New chat"}
        </Link>
      ))}
    </nav>
  )
}
