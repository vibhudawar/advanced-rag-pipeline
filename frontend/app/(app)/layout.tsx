import { redirect } from "next/navigation"

import { AppSidebar } from "@/components/layout/app-sidebar"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { listConversations, type ConversationSummary } from "@/lib/api"
import { createClient } from "@/lib/supabase/server"

// App shell: collapsible sidebar (logo, nav, history, account) + an inset main panel. The
// sidebar owns everything the old top bar did, so there's no navbar — just a collapse trigger
// floating at the top-left of the content.
export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) redirect("/login") // defense-in-depth; proxy already gates

  let conversations: ConversationSummary[] = []
  try {
    const { data } = await supabase.auth.getSession()
    const token = data.session?.access_token
    if (token) conversations = await listConversations(token)
  } catch {
    // Backend unreachable — render the shell with an empty history rather than failing.
  }

  return (
    <SidebarProvider>
      <AppSidebar
        user={{
          email: user.email ?? "",
          name:
            (user.user_metadata?.full_name as string | undefined) ??
            (user.user_metadata?.name as string | undefined),
          avatarUrl: (user.user_metadata?.avatar_url as string | undefined) ?? undefined,
        }}
        conversations={conversations}
      />
      <SidebarInset className="min-h-0">
        <header className="flex h-12 shrink-0 items-center px-2">
          <SidebarTrigger />
        </header>
        <div className="min-h-0 flex-1">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  )
}
