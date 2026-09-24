"""SQLite helpers for large IN (?, ?, ...) parameter lists.

SQLite enforces a maximum number of host parameters per statement (the
`SQLITE_MAX_VARIABLE_NUMBER` compile-time limit, often 999). A single
``IN`` clause with one placeholder per id can exceed that on large
tables (e.g. 5% of all office product rows).
"""

from __future__ import annotations

import sqlite3

# Stay below default 999; leave headroom for other parameters if extended later.
_IN_CHUNK = 500


def execute_where_in(
    conn: sqlite3.Connection,
    sql: str,
    ids: list,
    *,
    chunk_size: int = _IN_CHUNK,
) -> None:
    """Run ``sql`` once per chunk of ``ids``.

    ``sql`` must contain the literal ``{in_ph}`` once; it is replaced with a
    comma-separated list of ``?`` placeholders, e.g.::

        "UPDATE t SET c = 1 WHERE id IN ({in_ph})"
    """
    if "{in_ph}" not in sql:
        raise ValueError("sql must contain {in_ph} for the IN list")
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        statement = sql.replace("{in_ph}", placeholders, 1)
        conn.execute(statement, chunk)
