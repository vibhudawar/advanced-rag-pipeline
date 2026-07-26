"use client"

import { MessageSquare, Plus, Upload } from "lucide-react"
import Link from "next/link"
import { usePathname, useSearchParams } from "next/navigation"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import type { ConversationSummary } from "@/lib/api"

import { ConversationItem } from "./conversation-item"
import { ThemeToggle } from "./theme-toggle"
import { UserMenu } from "./user-menu"

const NAV = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/ingest", label: "Ingest", icon: Upload },
]

// Active nav item gets the blue (primary) fill, matching the reference dashboard.
const ACTIVE_CLASSES =
  "data-[active=true]:bg-primary data-[active=true]:text-primary-foreground " +
  "data-[active=true]:hover:bg-primary/90 data-[active=true]:hover:text-primary-foreground " +
  "data-[active=true]:active:bg-primary/90 data-[active=true]:active:text-primary-foreground"

export function AppSidebar({
  user,
  conversations,
}: {
  user: { email: string; name?: string; avatarUrl?: string }
  conversations: ConversationSummary[]
}) {
  const pathname = usePathname()
  const activeConv = useSearchParams().get("c")

  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-1 py-1">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
            R
          </div>
          <span className="text-base font-semibold group-data-[collapsible=icon]:hidden">RAG</span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        {/* Primary navigation (Chat / Ingest) */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map((item) => {
                const Icon = item.icon
                const isActive =
                  item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
                return (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      isActive={isActive}
                      tooltip={item.label}
                      className={ACTIVE_CLASSES}
                      render={<Link href={item.href} />}
                    >
                      <Icon className="size-4" />
                      <span>{item.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Conversation history — hidden on the collapsed icon rail */}
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="New chat" render={<Link href="/" />}>
                  <Plus className="size-4" />
                  <span>New chat</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>

          <SidebarGroupLabel>Recent</SidebarGroupLabel>
          <SidebarGroupContent>
            {conversations.length === 0 ? (
              <p className="px-2 py-1.5 text-xs text-muted-foreground">No conversations yet.</p>
            ) : (
              <SidebarMenu>
                {conversations.map((c) => (
                  <ConversationItem key={c.id} conversation={c} active={activeConv === c.id} />
                ))}
              </SidebarMenu>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="gap-1 border-t">
        <div className="flex items-center justify-between px-1 group-data-[collapsible=icon]:justify-center">
          <span className="px-2 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
            Appearance
          </span>
          <ThemeToggle />
        </div>
        <SidebarMenu>
          <SidebarMenuItem>
            <UserMenu user={user} />
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
