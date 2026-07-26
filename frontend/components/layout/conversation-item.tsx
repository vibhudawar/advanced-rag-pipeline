"use client"

import { MoreHorizontal, Pencil, Trash2 } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState } from "react"
import { toast } from "sonner"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { SidebarMenuAction, SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar"
import { deleteConversation, renameConversation, type ConversationSummary } from "@/lib/api"
import { createClient } from "@/lib/supabase/client"

async function getToken(): Promise<string | undefined> {
  const { data } = await createClient().auth.getSession()
  return data.session?.access_token
}

// One conversation row: link + a hover "…" menu to rename (dialog) or delete (confirm). Deleting
// the currently-open thread navigates back to a fresh chat.
export function ConversationItem({
  conversation,
  active,
}: {
  conversation: ConversationSummary
  active: boolean
}) {
  const router = useRouter()
  const label = conversation.title || "New chat"
  const [renameOpen, setRenameOpen] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [title, setTitle] = useState(label)
  const [busy, setBusy] = useState(false)

  async function doRename() {
    const next = title.trim()
    if (!next) return
    const token = await getToken()
    if (!token) return toast.error("Your session expired. Please sign in again.")
    setBusy(true)
    try {
      await renameConversation(conversation.id, next, token)
      setRenameOpen(false)
      router.refresh()
    } catch {
      toast.error("Couldn't rename the conversation.")
    } finally {
      setBusy(false)
    }
  }

  async function doDelete() {
    const token = await getToken()
    if (!token) return toast.error("Your session expired. Please sign in again.")
    setBusy(true)
    try {
      await deleteConversation(conversation.id, token)
      toast.success("Conversation deleted")
      if (active) router.push("/")
      router.refresh() // row unmounts when the refreshed list drops it
    } catch {
      toast.error("Couldn't delete the conversation.")
      setBusy(false)
    }
  }

  return (
    <SidebarMenuItem>
      <SidebarMenuButton isActive={active} render={<Link href={`/?c=${conversation.id}`} />}>
        <span className="truncate">{label}</span>
      </SidebarMenuButton>

      <DropdownMenu>
        <DropdownMenuTrigger
          render={<SidebarMenuAction showOnHover aria-label="Conversation options" />}
        >
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent side="right" align="start" className="w-40">
          <DropdownMenuItem
            onClick={() => {
              setTitle(label)
              // defer so the menu finishes closing before the dialog grabs focus
              setTimeout(() => setRenameOpen(true), 0)
            }}
          >
            <Pencil className="mr-2 size-4" />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem
            className="text-destructive focus:text-destructive"
            onClick={() => setTimeout(() => setDeleteOpen(true), 0)}
          >
            <Trash2 className="mr-2 size-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Rename conversation</DialogTitle>
          </DialogHeader>
          <Input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault()
                void doRename()
              }
            }}
            maxLength={120}
            autoFocus
            aria-label="Conversation title"
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button onClick={doRename} disabled={busy || !title.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes &ldquo;{label}&rdquo; and its messages. This can&apos;t be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault()
                void doDelete()
              }}
              disabled={busy}
              className="bg-destructive text-white hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SidebarMenuItem>
  )
}
