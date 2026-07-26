# Frontend Conventions

> Staged as `frontend-CLAUDE.md`; move to `frontend/CLAUDE.md` after the Next.js app is scaffolded.

The frontend for the RAG service (WIN 7c). Read `frontend-ui-plan.md` for product context and
the active wireframe/scope. The backend is a **separate Python FastAPI service** (`api/`,
`src/`) — the frontend never imports it, it calls it over HTTP.

## Stack

Next.js 15 App Router • TypeScript • Tailwind v4 • shadcn/ui • `@supabase/ssr` + `supabase-js`
(auth only) • TanStack Query (client-only bits) • react-hook-form + Zod (the ingest form) •
Sonner (toasts) • lucide-react • pnpm.

Data layer: the FastAPI backend owns all app reads/writes. Supabase Postgres is written by the
backend (service role); its schema is owned by **raw SQL migrations** in `supabase/migrations/`
(Python side). **No Drizzle. No ORM in the frontend. No client DB queries.** `supabase-js`/
`@supabase/ssr` are used **only for auth** (Google OAuth + session cookies) — never for data
reads. All chat/document data flows through the backend API.

## Backend contract

- Base URL: `process.env.NEXT_PUBLIC_API_URL`.
- `POST /stream` → SSE: `conversation` (once, `{conversation_id}`), then `token` events, then
  `citations`, then `done`. Consume with `fetch` + a `ReadableStream` reader.
- `GET /conversations/{id}`, `GET /documents`, `POST /ingest` (multipart).
- All calls go through `lib/api.ts` (one typed client). No `fetch` scattered in components.

## Folder Structure

See `frontend-ui-plan.md` § 5. Do NOT introduce new top-level folders without explicit approval.

## Server vs Client Components

Default = Server Component. Only mark `"use client"` when one is true:
- Uses React hooks (`useState`, `useEffect`, `useReducer`, …)
- Uses event handlers (`onClick`, `onChange`, …)
- Uses browser-only APIs (`window`, `localStorage`, `document`)
- Uses third-party client libraries (react-hook-form, TanStack Query, dnd, …)
- Wraps shadcn primitives that are themselves client (Dialog, Sheet, DropdownMenu, Command,
  Popover, Tabs with state, …)

Push `"use client"` to the **leaf**, not the page. If a page needs one client button, only that
button is `"use client"`. The page stays a Server Component. (Chat composer, stream reader,
dropzone, tab-switcher, theme-toggle are the client leaves.)

## Data Fetching

- **Reads** in pages/layouts: call the backend from Server Components (`lib/api.ts`) — no
  `useEffect`, no client fetch, no Next API route as a middleman for internal data.
- **Mutations / chat**: the streaming call is a client leaf using `fetch` + stream reader.
  Ingest submits via a Server Action or a direct typed `POST /ingest` from a client form.
- **TanStack Query**: only for client-side interactions that need debounce/optimistic —
  ⌘K search, ingest progress polling, optimistic sidebar. Never wrap a Server Component's data
  in TanStack Query.

## URL State

Filters, the active tab, and any shareable view live in `searchParams`, not React state. The
page reads `searchParams`, calls the query, renders. This keeps refresh/share working and avoids
a client cache to invalidate. (Chat vs Ingest tab reflects in the URL, e.g. `/` and `/ingest`.)

## Streaming

Use `<Suspense fallback={<Skeleton />}>` for slow sections (sidebar list, documents table) so the
shell renders immediately. The chat answer streams token-by-token from `/stream`; show a
"retrieving…" shimmer only until the first token.

## Auth & sessions

Google OAuth via Supabase (`@supabase/ssr`, cookie-based). `middleware.ts` refreshes the session
and gates the app; login is a Google button. Every backend call from `lib/api.ts` attaches the
Supabase access token (`Authorization: Bearer`); the backend validates it and scopes data to the
`user_id`. History is **per-user** (the user's conversations), not per-browser. Never trust the
client for identity — the backend derives `user_id` from the verified token, not from a request field.

## Security

- Only `NEXT_PUBLIC_*` env vars in the frontend. The `SUPABASE_SECRET_KEY` and DB URLs are
  **backend-only** — never import or reference them client-side.
- Assistant/markdown content is model output → **untrusted**. Render with `react-markdown`
  with raw HTML disabled. Never `dangerouslySetInnerHTML` with model or user content.
- All backend errors surface as a generic `toast.error(...)`; never render raw error bodies.

## UI Conventions

- Light-first, professional; blue accent (`primary` token). Comfortable density.
- Empty states: one-line message + a CTA. Skeleton loaders for lists. Optimistic updates only
  for cheap toggles.
- Confirmations: `AlertDialog` for destructive actions, Sonner toast for success.
- **No emojis** in UI labels. lucide-react icons only.

## Definition of Done (per sub-phase)

All three states (empty / loading / error) implemented; responsive (mobile sidebar → Sheet);
dark + light both correct; a keyboard/a11y pass; no `any`; builds clean with `pnpm build`.

## Forbidden

- `any` types. Use `unknown` and narrow.
- `// TODO` left in code. Either do it or open an issue.
- New libraries without a proposal in chat first.
- Direct DB access from the client (no `supabase-js` data queries pre-auth). Always via the
  backend API.
- `console.log` in committed code (use `console.error` for errors only, or a logger).
- `"use client"` on a `page.tsx` or `layout.tsx` file. Ever.
- Next API routes as a middleman for internal data — call the backend or use RSC/Server Actions.
- `useEffect` for data fetching. Use RSC.
- Form state in `useState` for non-trivial forms. Use react-hook-form + Zod.
- Filter/tab state in `useState`. Use URL `searchParams`.
- `dangerouslySetInnerHTML` with model/user content.
- Hardcoded secrets; any non-`NEXT_PUBLIC_` secret referenced client-side.
