import { Plus } from "lucide-react"
import Link from "next/link"

import { buttonVariants } from "@/components/ui/button"
import { listConversations, type ConversationSummary } from "@/lib/api"
import { createClient } from "@/lib/supabase/server"
import { cn } from "@/lib/utils"

import { ConversationLinks } from "./conversation-links"

// Per-user thread list (server component). Fetches the signed-in user's conversations from the
// backend using their session token. If the backend is unreachable the shell still renders.
export async function AppSidebar() {
  let conversations: ConversationSummary[] = []
  try {
    const supabase = await createClient()
    const { data } = await supabase.auth.getSession()
    const token = data.session?.access_token
    if (token) conversations = await listConversations(token)
  } catch {
    // Backend down — degrade to an empty list rather than breaking the app shell.
  }

  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r md:flex">
      <div className="p-3">
        <Link
          href="/"
          className={cn(buttonVariants({ variant: "outline" }), "w-full justify-start gap-2")}
        >
          <Plus className="size-4" />
          New chat
        </Link>
      </div>
      <div className="px-3 pb-1 text-xs font-medium text-muted-foreground">Recent</div>
      <ConversationLinks conversations={conversations} />
    </aside>
  )
}
