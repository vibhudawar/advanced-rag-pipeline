"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

import { cn } from "@/lib/utils"

const TABS = [
  { href: "/", label: "Chat" },
  { href: "/ingest", label: "Ingest" },
] as const

export function TabSwitcher() {
  const pathname = usePathname()
  return (
    <nav className="inline-flex items-center rounded-lg bg-muted p-1">
      {TABS.map((tab) => {
        const active = tab.href === "/" ? pathname === "/" : pathname.startsWith(tab.href)
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={cn(
              "rounded-md px-4 py-1.5 text-sm font-medium transition-colors",
              active
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {tab.label}
          </Link>
        )
      })}
    </nav>
  )
}
