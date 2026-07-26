import { redirect } from "next/navigation"

import { AppSidebar } from "@/components/layout/app-sidebar"
import { TopBar } from "@/components/layout/top-bar"
import { createClient } from "@/lib/supabase/server"

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) redirect("/login") // defense-in-depth; middleware already gates

  return (
    <div className="flex h-svh flex-col">
      <TopBar
        user={{
          email: user.email ?? "",
          avatarUrl: (user.user_metadata?.avatar_url as string | undefined) ?? undefined,
        }}
      />
      <div className="flex min-h-0 flex-1">
        <AppSidebar />
        <main className="min-h-0 flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  )
}
