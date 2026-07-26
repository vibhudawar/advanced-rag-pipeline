import { TabSwitcher } from "@/components/layout/tab-switcher"
import { ThemeToggle } from "@/components/layout/theme-toggle"
import { UserMenu } from "@/components/layout/user-menu"

export function TopBar({ user }: { user: { email: string; avatarUrl?: string } }) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b px-4">
      <div className="flex items-center gap-2 font-semibold">
        <span className="grid size-7 place-items-center rounded-md bg-primary text-xs text-primary-foreground">
          R
        </span>
        <span className="hidden sm:inline">RAG</span>
      </div>
      <TabSwitcher />
      <div className="flex items-center gap-1">
        <ThemeToggle />
        <UserMenu user={user} />
      </div>
    </header>
  )
}
