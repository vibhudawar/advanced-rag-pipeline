import { Skeleton } from "@/components/ui/skeleton"

// Shown while a page in the app group server-renders (e.g. switching conversations, which
// re-runs the async chat page and suspends).
export default function Loading() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6" aria-hidden>
      <div className="flex justify-end">
        <Skeleton className="h-10 w-56 rounded-2xl" />
      </div>
      <div className="space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
      </div>
      <div className="flex justify-end">
        <Skeleton className="h-10 w-40 rounded-2xl" />
      </div>
      <div className="space-y-2">
        <Skeleton className="h-4 w-1/2" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-4 w-3/5" />
      </div>
    </div>
  )
}
