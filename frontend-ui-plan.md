# Frontend UI Plan (WIN 7c)

A fast, modern, professional web UI for the RAG service. Two-tab workspace — **Chat** and
**Ingest** — mirroring the ChatGPT Chat/Work split. Next.js 15 (App Router) + TypeScript +
Tailwind v4 + shadcn/ui, deployed on **Vercel**; talks to the FastAPI backend on **Render**.

Scaffold (run when we start building — creates `frontend/`):
`pnpm dlx shadcn@latest init --preset b0 --template next`

## 1. Decisions (locked)

- **Theme:** light-first, **blue accent** (`primary` = blue-600), enterprise/SaaS feel; dark-mode toggle included.
- **Ingest:** real upload → parse → chunk → embed → Pinecone, **capped** (size/type/rate limits) for a public demo. Documents table from Supabase.
- **Auth:** **Google OAuth via Supabase** (enabled). History is **per-user** — the sidebar shows the logged-in user's conversations. Replaces the earlier per-browser-localStorage idea.
- **Goal:** live public demo (Vercel + Render, both free tier).
- **Repo:** single monorepo (this repo). Vercel builds `frontend/` (Root Directory setting); Render builds the root Dockerfile (frontend excluded via `.dockerignore`). No repo split.

## 2. Design language

- Base `zinc-50` / white surfaces, `zinc-900` text; **accent blue-600** for primary buttons, active tab, focus rings, citation chips.
- Rounded-`2xl` cards, hairline `zinc-200` borders, soft shadows. Comfortable density.
- Typography: Geist (ships with the b0 preset). lucide-react icons only, **no emojis** in UI.
- Every list has skeletons; every empty state has a one-line message + a CTA.
- Motion: minimal — fade/slide for streaming tokens and tab switches; nothing flashy.

## 3. App shell / wireframe

```
┌───────────────────────────────────────────────────────────────────────────┐
│ ▣ RAG            ┌───────── Chat │ Ingest ─────────┐            ◐ theme  ⌘K │  top bar
├───────────────┬───────────────────────────────────────────────────────────┤
│  + New chat   │                                                             │
│  ⌕ search     │                     (tab content)                          │
│               │                                                             │
│  Today        │                                                             │
│   • Metformin…│                                                             │
│   • Vitamin D…│                                                             │
│  Earlier      │                                                             │
│   • Tesla ris…│                                                             │
│               │                                                             │
│  [collapse ‹] │                                                             │
└───────────────┴───────────────────────────────────────────────────────────┘
  sidebar (Chat tab; collapsible)          main = active tab
```

### 3a. Chat tab — empty state
```
                         What do you want to know?

        ┌───────────────────────────────────────────────────────┐
        │  Ask about your documents…                             │
        │                                                        │
        │                                          [ Send  ↑ ]   │
        └───────────────────────────────────────────────────────┘
              Grounded answers with citations. Says "I don't
              know" when the docs don't cover it.
```

### 3b. Chat tab — active thread
```
  You                                                      What is metformin's
                                                    effect on cell proliferation?

  Assistant
  Prostaglandin D2 impedes TNF-α–triggered migration of Langerhans cells [1],
  and … [2].
     ▸ Sources (2)                                        ← collapsible
       [1] scifact:12345 · score 0.87 · "…snippet…"
       [2] scifact:67890 · score 0.81 · "…snippet…"

  (streaming: tokens append live; a "retrieving…" shimmer shows before token 1)

  ┌───────────────────────────────────────────────────────┐
  │  Ask a follow-up…                          [ Send ↑ ]  │   ← sticky composer
  └───────────────────────────────────────────────────────┘
```
- Abstention ("I don't have enough information…") renders as a plain, calm message (no error styling).
- Citations are blue chips `[n]`; clicking scrolls to / expands that source. Sources panel is collapsed by default.

### 3c. Ingest tab
```
  ┌─────────────────────────────────────────────────────────────┐
  │            ⬆  Drop PDF / TXT / MD / DOCX here                │   dropzone
  │                    or click to browse                        │
  │        Max 5 MB · up to N pages · these limits keep the      │
  │                 public demo cheap and safe                   │
  └─────────────────────────────────────────────────────────────┘
  ▸ Advanced   index: [rag-documents ▾]  chunking: [recursive ▾]  size 1000 / overlap 200

  Queued:  report.pdf  [▓▓▓▓▓▓▓░░░]  embedding…  (per-file progress)

  Documents                                                    ⟳ refresh
  ┌─────────────┬────────┬────────┬───────────┬──────────────┐
  │ filename    │ type   │ chunks │ status     │ added        │
  ├─────────────┼────────┼────────┼───────────┼──────────────┤
  │ report.pdf  │ pdf    │  42    │ ✓ success  │ 2m ago       │
  │ notes.txt   │ txt    │   8    │ ⏳ pending │ just now     │
  └─────────────┴────────┴────────┴───────────┴──────────────┘
```

## 4. Architecture

- **Server Components by default.** `"use client"` only at leaves that need it (composer, stream reader, dropzone, tab state).
- **Chat streaming:** a client component reads `POST {API}/stream` via `fetch` + `ReadableStream` (SSE), appending `token`s, capturing the `conversation` + `citations` events. On a new chat, save the returned `conversation_id` to localStorage.
- **Reads (history, documents):** through the **backend API** (not the anon key / not direct DB) — RLS is deferred, so the backend (service role) mediates. Server Components fetch from the API; TanStack Query only for client-side bits (⌘K search, optimistic sidebar).
- **Ingest:** multipart upload to `POST {API}/ingest`; poll `GET {API}/documents` (or stream progress) for status.
- **No Drizzle, no direct client DB.** Supabase schema is owned by the raw SQL migrations on the Python side; the frontend never touches Postgres directly. `supabase-js` reserved for auth (7d).

### Backend additions this win needs (small FastAPI endpoints)
| Endpoint | Purpose |
|---|---|
| `GET /conversations` | the logged-in user's conversations (filtered by verified `user_id`) |
| `GET /conversations/{id}` | title + messages for one conversation (owner-checked) |
| `GET /documents` | rows for the Ingest table (Supabase `documents`, user-scoped) |
| `POST /ingest` | upload → `ingest_documents` → Pinecone + `documents` row. **Guardrails:** max size, allowed extensions, rate limit; validate filename; generic errors. |

All authed endpoints read `Authorization: Bearer <supabase access token>`, validate via
`supabase.auth.get_user(token)`, and use the resulting `user_id`. `POST /stream` also takes the
token, stamps `conversations.user_id`, and returns `conversation_id` via the `conversation` SSE event.

## 4a. Auth (Google OAuth via Supabase)

- **Frontend:** `@supabase/ssr` for cookie-based sessions in the App Router. A Google login button
  starts the OAuth flow; `middleware.ts` refreshes the session and gates the app. The client
  attaches the access token to every backend call (via `lib/api.ts`).
- **Backend:** validate the bearer token with `supabase.auth.get_user(token)` → `user_id`; create
  conversations with that `user_id`; filter list/get/documents by it. 401 on missing/invalid token.
- **DB:** enable the RLS policies from `0001_init_schema.sql` (currently commented) and grant the
  `authenticated` role. RLS is defense-in-depth; the backend still enforces `user_id` explicitly.
- `lib/sessions.ts` (localStorage) is no longer needed — history is keyed on the user.

## 5. Folder structure (inside `frontend/`)

```
frontend/
  app/
    layout.tsx                 # shell: top bar + tabs + sidebar (RSC)
    page.tsx                   # Chat tab (default)
    ingest/page.tsx            # Ingest tab
    globals.css
  components/
    chat/                      # composer, message, sources, thread (client leaves)
    ingest/                    # dropzone, options, documents-table
    layout/                    # tab-switcher, sidebar, theme-toggle
    ui/                        # shadcn components (generated)
  lib/
    api.ts                     # typed fetch client for the FastAPI backend (SSE + REST)
    sessions.ts                # localStorage conversation-id tracking
    database.types.ts          # `supabase gen types` output (used later/for auth)
  hooks/
```
Do NOT add new top-level folders without approval.

## 6. Components to generate (shadcn CLI)

`button textarea input card tabs scroll-area sidebar tooltip dropdown-menu sheet dialog alert-dialog skeleton sonner badge table progress separator avatar resizable command collapsible`

## 7. Performance / optimization

- RSC shell → minimal client JS; push `"use client"` to leaves.
- Stream renders incrementally (first token fast); shimmer only until token 1.
- `<Suspense>` + skeletons for sidebar list and documents table.
- Markdown: `react-markdown` with **no raw HTML** (safe by default) — the assistant text is model output, treat as untrusted; never `dangerouslySetInnerHTML`.
- Dates via `Intl`, no heavyweight date lib. Lucide icons tree-shaken.
- Vercel edge/CDN for static; API calls go to Render (note first-request cold-start ~25s — show a "waking up" state on the very first request).

## 8. Build order (7c sub-phases, branch per phase)

- **7c.0 scaffold** — `frontend/` via shadcn b0, blue theme tokens, shell (top bar + tabs + collapsible sidebar), theme toggle.
- **7c.1 auth** — `@supabase/ssr` sessions + Google login + `middleware.ts`; backend token validation (`supabase.auth.get_user`) + `user_id` scoping; enable RLS policies + grant `authenticated`.
- **7c.2 chat** — streaming chat against `/stream`, message + citations UI, per-user sidebar (`GET /conversations`, `GET /conversations/{id}`).
- **7c.3 ingest** — dropzone + options + documents table + `POST /ingest` & `GET /documents` (with guardrails).
- **7c.4 polish** — empty/loading/error states, responsive (mobile: sidebar → sheet), dark mode, a11y pass.
- **7c.5 deploy** — Vercel (Root Directory `frontend`) + Render (root Dockerfile) wired via env; CORS to the Vercel origin; end-to-end live check.

## 9. Deployment (single monorepo → two services)

**Vercel (frontend):** import this repo → **Root Directory = `frontend`** (Vercel then builds only
the Next app; the Python is invisible to it). Env:
`NEXT_PUBLIC_API_URL` (Render backend URL), `NEXT_PUBLIC_SUPABASE_URL`,
`NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`, `NEXT_PUBLIC_APP_URL` (the Vercel URL).

**Render (backend):** Docker service from this repo, root `Dockerfile` (`.dockerignore` excludes
`frontend/`). Env: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY`, `SUPABASE_URL`
(=NEXT_PUBLIC_SUPABASE_URL value), `SUPABASE_SECRET_KEY`, `RAG_INDEX`, and
`ALLOWED_ORIGINS` = the Vercel origin (for CORS).

Secrets split: `SUPABASE_SECRET_KEY` + DB URLs are **backend-only** (Render), never in Vercel/frontend.
Optional: Vercel "Ignored Build Step" / Render path filters so each service only rebuilds when its
own files change.

## 10. Open questions / later

- Auth (Google via Supabase) + RLS = **7d**, deferred.
- Model routing / cost display in the UI = ties to WIN 6 (later).
- `CLAUDE.md` for the frontend lives at `frontend/CLAUDE.md` (drafted as `frontend-CLAUDE.md`, moved in after scaffold).
