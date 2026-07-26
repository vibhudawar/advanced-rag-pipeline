"use client"

import { RotateCw } from "lucide-react"
import { useEffect } from "react"

import { Button } from "@/components/ui/button"

// Error boundary for the app group. Shows a generic message (never the raw error, which may
// carry internal detail) and offers a retry.
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error(error)
  }, [error])

  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
      <p className="text-sm font-medium">Something went wrong</p>
      <p className="max-w-md text-sm text-muted-foreground">
        The app hit an unexpected error. Please try again.
      </p>
      <Button variant="outline" onClick={reset} className="mt-1 gap-2">
        <RotateCw className="size-4" />
        Try again
      </Button>
    </div>
  )
}
