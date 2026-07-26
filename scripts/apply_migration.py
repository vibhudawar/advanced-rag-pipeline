"""Apply supabase/migrations/*.sql to the database at $DATABASE_URL (one-off dev/ops task).

Reads the connection string from the environment (DATABASE_URL, falling back to
DATABASE_POOLED_URL) — never hardcoded. Migrations are idempotent (IF NOT EXISTS), so
re-running is safe. Statement splitting is dollar-quote aware so the trigger function body
isn't broken on its inner semicolons.

    python -m scripts.apply_migration
"""

from __future__ import annotations

import glob
import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()


def split_sql(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if ln.strip() and not ln.strip().startswith("--")]
    text = "\n".join(lines)
    stmts: list[str] = []
    buf: list[str] = []
    in_dollar = False
    i = 0
    while i < len(text):
        if text[i:i + 2] == "$$":
            in_dollar = not in_dollar
            buf.append("$$")
            i += 2
            continue
        ch = text[i]
        if ch == ";" and not in_dollar:
            stmt = "".join(buf).strip()
            if stmt:
                stmts.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def _connect():
    # Try direct first, then the (IPv4) transaction pooler — Supabase's direct host is
    # often IPv6-only and won't resolve on IPv4 networks.
    candidates = [("DATABASE_URL", os.getenv("DATABASE_URL")),
                  ("DATABASE_POOLED_URL", os.getenv("DATABASE_POOLED_URL"))]
    last_exc = None
    for name, url in candidates:
        if not url:
            continue
        try:
            conn = psycopg.connect(url, autocommit=True, connect_timeout=15)
            print(f"[connect] via {name}")
            return conn
        except Exception as e:  # noqa: BLE001 - try the next connection option
            print(f"[connect] {name} failed: {type(e).__name__}")
            last_exc = e
    raise RuntimeError("could not connect via DATABASE_URL or DATABASE_POOLED_URL") from last_exc


def main() -> int:
    files = sorted(glob.glob("supabase/migrations/*.sql"))
    if not files:
        print("[error] no migrations found under supabase/migrations/")
        return 2

    with _connect() as conn:
        for path in files:
            with open(path, encoding="utf-8") as f:
                stmts = split_sql(f.read())
            print(f"[apply] {path} ({len(stmts)} statements)")
            for stmt in stmts:
                conn.execute(stmt)
        rows = conn.execute(
            "select table_name from information_schema.tables "
            "where table_schema = 'public' order by table_name"
        ).fetchall()
        print("[public tables]", [r[0] for r in rows])
    print("[done] migration applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
