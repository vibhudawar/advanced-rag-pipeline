import { DocumentsTable } from "@/components/ingest/documents-table"
import { IngestDropzone } from "@/components/ingest/ingest-dropzone"
import { listDocuments, type DocumentRow } from "@/lib/api"
import { createClient } from "@/lib/supabase/server"

// Ingest tab. Server-loads the user's documents; the dropzone uploads and refreshes this list.
export default async function IngestPage() {
  let documents: DocumentRow[] = []
  const supabase = await createClient()
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  if (token) {
    try {
      documents = await listDocuments(token)
    } catch {
      // Backend unreachable — render an empty table; upload will surface the error.
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-6 overflow-y-auto px-4 py-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload documents to make them searchable in chat. Answers cite what they use.
        </p>
      </div>
      <IngestDropzone />
      <DocumentsTable documents={documents} />
    </div>
  )
}
