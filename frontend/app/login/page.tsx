import { GoogleButton } from "@/components/auth/google-button"

export default function LoginPage() {
  return (
    <main className="flex min-h-svh items-center justify-center bg-muted/30 p-6">
      <div className="w-full max-w-sm space-y-6 rounded-2xl border bg-card p-8 shadow-sm">
        <div className="space-y-2 text-center">
          <div className="mx-auto grid size-11 place-items-center rounded-xl bg-primary text-lg font-semibold text-primary-foreground">
            R
          </div>
          <h1 className="text-xl font-semibold">Sign in to RAG</h1>
          <p className="text-sm text-muted-foreground">
            Ask your documents — grounded, cited answers.
          </p>
        </div>
        <GoogleButton />
        <p className="text-center text-xs text-muted-foreground">
          Continue with Google to create your account or sign in.
        </p>
      </div>
    </main>
  )
}
