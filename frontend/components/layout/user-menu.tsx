"use client"

import { ChevronsUpDown, LogOut } from "lucide-react"
import { useRouter } from "next/navigation"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { createClient } from "@/lib/supabase/client"

// The account card at the bottom of the sidebar: avatar + name/email, opening a menu with
// Sign out. Collapses to just the avatar on the icon rail.
export function UserMenu({
  user,
}: {
  user: { email: string; name?: string; avatarUrl?: string }
}) {
  const router = useRouter()

  async function signOut() {
    await createClient().auth.signOut()
    router.replace("/login")
    router.refresh()
  }

  const initial = (user.name || user.email)?.[0]?.toUpperCase() ?? "U"
  const label = user.name || user.email

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <button
            aria-label="Account"
            className="flex w-full items-center gap-2 rounded-md p-2 text-left outline-none transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 focus-visible:ring-ring group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:p-1.5 data-[popup-open]:bg-sidebar-accent"
          />
        }
      >
        <Avatar className="size-8 shrink-0 rounded-lg">
          {user.avatarUrl ? <AvatarImage src={user.avatarUrl} alt="" /> : null}
          <AvatarFallback className="rounded-lg text-xs">{initial}</AvatarFallback>
        </Avatar>
        <div className="grid flex-1 text-left leading-tight group-data-[collapsible=icon]:hidden">
          <span className="truncate text-sm font-medium">{label}</span>
          <span className="truncate text-xs text-muted-foreground">{user.email}</span>
        </div>
        <ChevronsUpDown className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden" />
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="end" className="min-w-56">
        <div className="truncate px-2 py-1.5 text-xs text-muted-foreground">{user.email}</div>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={signOut}>
          <LogOut className="mr-2 size-4" />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
