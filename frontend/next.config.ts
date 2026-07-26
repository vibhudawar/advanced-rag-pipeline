import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  // Pin the workspace root to this app so Next doesn't walk up to a stray ~/package-lock.json.
  turbopack: { root: import.meta.dirname },
}

export default nextConfig
