import { CheckCircle2, FileText, XCircle } from "lucide-react"

import type { DocumentRow } from "@/lib/api"
import { cn } from "@/lib/utils"

function formatSize(bytes: number | null): string {
  if (!bytes) return "—"
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" }) +
    ", " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
}

function StatusPill({ status, error }: { status: DocumentRow["status"]; error: string | null }) {
  const map = {
    success: { icon: CheckCircle2, label: "Indexed", cls: "text-emerald-600 dark:text-emerald-400" },
    failed: { icon: XCircle, label: "Failed", cls: "text-destructive" },
    pending: { icon: FileText, label: "Pending", cls: "text-amber-600 dark:text-amber-400" },
  } as const
  const { icon: Icon, label, cls } = map[status]
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs font-medium", cls)} title={error ?? undefined}>
      <Icon className="size-3.5" />
      {label}
    </span>
  )
}

// The user's ingested documents. Server-rendered from GET /documents; refreshed after upload.
export function DocumentsTable({ documents }: { documents: DocumentRow[] }) {
  if (documents.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-8 text-center">
        <FileText className="mx-auto size-8 text-muted-foreground/50" />
        <p className="mt-3 text-sm font-medium">No documents yet</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload a PDF, DOCX, TXT, or Markdown file to make it searchable in chat.
        </p>
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded-lg border">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-xs text-muted-foreground">
          <tr>
            <th className="px-4 py-2.5 text-left font-medium">Document</th>
            <th className="px-4 py-2.5 text-left font-medium">Status</th>
            <th className="hidden px-4 py-2.5 text-right font-medium sm:table-cell">Chunks</th>
            <th className="hidden px-4 py-2.5 text-right font-medium sm:table-cell">Size</th>
            <th className="hidden px-4 py-2.5 text-right font-medium md:table-cell">Added</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {documents.map((doc) => (
            <tr key={doc.id} className="hover:bg-muted/30">
              <td className="max-w-0 px-4 py-3">
                <div className="flex items-center gap-2">
                  <FileText className="size-4 shrink-0 text-muted-foreground" />
                  <span className="truncate font-medium" title={doc.filename}>
                    {doc.filename}
                  </span>
                </div>
                {doc.status === "failed" && doc.error && (
                  <p className="mt-1 truncate pl-6 text-xs text-destructive" title={doc.error}>
                    {doc.error}
                  </p>
                )}
              </td>
              <td className="px-4 py-3">
                <StatusPill status={doc.status} error={doc.error} />
              </td>
              <td className="hidden px-4 py-3 text-right tabular-nums text-muted-foreground sm:table-cell">
                {doc.num_chunks ?? "—"}
              </td>
              <td className="hidden px-4 py-3 text-right tabular-nums text-muted-foreground sm:table-cell">
                {formatSize(doc.file_size)}
              </td>
              <td className="hidden px-4 py-3 text-right whitespace-nowrap text-muted-foreground md:table-cell">
                {formatDate(doc.created_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
