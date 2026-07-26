import { Plus } from "lucide-react"
import Link from "next/link"

import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// Placeholder sidebar for 7c.1. The real per-user conversation list + search arrives in 7c.2.
export function AppSidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r p-3 md:flex">
      <Link href="/" className={cn(buttonVariants({ variant: "outline" }), "justify-start gap-2")}>
        <Plus className="size-4" />
        New chat
      </Link>
      <div className="mt-4 px-1 text-xs font-medium text-muted-foreground">Recent</div>
      <p className="mt-2 px-1 text-sm text-muted-foreground">No conversations yet.</p>
    </aside>
  )
}
