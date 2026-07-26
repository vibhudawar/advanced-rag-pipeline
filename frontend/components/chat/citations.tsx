"use client"

import { FileText } from "lucide-react"
import { useState } from "react"

import type { Citation } from "@/lib/api"
import { cn } from "@/lib/utils"

// Sources behind an assistant answer. Numbered chips expand to show the grounding snippet.
export function Citations({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState<number | null>(null)
  if (citations.length === 0) return null

  return (
    <div className="mt-3 border-t pt-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <FileText className="size-3.5" />
        Sources
      </div>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((c) => (
          <button
            key={c.n}
            type="button"
            onClick={() => setOpen(open === c.n ? null : c.n)}
            className={cn(
              "inline-flex max-w-[16rem] items-center gap-1 rounded-md border px-2 py-1 text-xs transition-colors",
              "hover:bg-accent hover:text-accent-foreground",
              open === c.n && "bg-accent text-accent-foreground",
            )}
            title={c.source}
          >
            <span className="font-mono text-[0.7rem] text-muted-foreground">[{c.n}]</span>
            <span className="truncate">{c.source}</span>
          </button>
        ))}
      </div>
      {open !== null && (
        <p className="mt-2 rounded-md bg-muted p-2.5 text-xs leading-relaxed text-muted-foreground">
          {citations.find((c) => c.n === open)?.snippet ?? "No preview available."}
        </p>
      )}
    </div>
  )
}
