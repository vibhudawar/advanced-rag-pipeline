import { ChatView } from "@/components/chat/chat-view"
import { getConversation, type ChatMessage } from "@/lib/api"
import { createClient } from "@/lib/supabase/server"

// Chat tab. Server-loads the selected thread's history (from `?c=<id>`) and hands it to the
// client ChatView. Keyed by conversation id so switching threads mounts fresh state.
export default async function ChatPage({
  searchParams,
}: {
  searchParams: Promise<{ c?: string }>
}) {
  const { c } = await searchParams
  let initialMessages: ChatMessage[] = []

  if (c) {
    const supabase = await createClient()
    const { data } = await supabase.auth.getSession()
    const token = data.session?.access_token
    if (token) {
      try {
        const msgs = await getConversation(c, token)
        if (msgs) initialMessages = msgs
      } catch {
        // Backend unreachable — render an empty thread; the client surfaces errors on send.
      }
    }
  }

  return <ChatView key={c ?? "new"} conversationId={c} initialMessages={initialMessages} />
}
