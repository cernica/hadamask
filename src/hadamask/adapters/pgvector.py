"""PostgreSQL + pgvector adapter (psycopg 3).

One table per collection: ``id TEXT PRIMARY KEY, embedding vector(D), payload JSONB``.
Collection names are validated against a strict identifier pattern and quoted with
:class:`psycopg.sql.Identifier`; every value (including field names inside filters) is a
bound parameter.

Filters on the JSONB payload:

* ``$eq`` - ``payload @> {"f": v}`` or ``payload @> {"f": [v]}`` (so list-valued fields match
  when they contain the value); served by a GIN index;
* ``$in`` - OR of ``$eq``; ``$ne``/``$nin`` - their negation, so records without the field
  match, as in MongoDB;
* ranges - ``payload -> f`` compared as ``jsonb``, restricted to values of the same JSON type
  (jsonb orders numbers numerically and strings lexicographically).

Scores follow :class:`~hadamask.adapters.base.Hit`: cosine -> ``1 - (a <=> b)``,
dot -> ``-(a <#> b)``, l2 -> ``a <-> b``.

With ``hnsw=True`` an HNSW index is built (``m=16, ef_construction=200``). Searches set
``hnsw.ef_search`` (default ``max(100, k)``) and ``hnsw.iterative_scan`` (default
``strict_order``, pgvector >= 0.8) so that filtered queries still return ``k`` rows; both can be
overridden per search: ``search(..., ef_search=400, iterative_scan="relaxed_order")``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, Optional, Union

import numpy as np

from .._deps import missing_extra

try:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql
    from psycopg.types.json import Jsonb
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise missing_extra("pgvector", exc) from exc

from ..core import Protector
from .base import And, Condition, Filter, ProtectedStore, RawHit, RawRecord

HNSW_MAX_DIM = 2000  # pgvector limit for the vector type
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_OPS = {
    "cosine": ("<=>", "vector_cosine_ops", "1 - ({col} <=> %s)"),
    "dot": ("<#>", "vector_ip_ops", "-({col} <#> %s)"),
    "l2": ("<->", "vector_l2_ops", "{col} <-> %s"),
}
_RANGE = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
_ITERATIVE_SCAN = ("off", "strict_order", "relaxed_order")


def _ident(name: str) -> sql.Identifier:
    if not _NAME.match(name):
        raise ValueError(f"invalid collection name {name!r}: use letters, digits and '_' (max 63)")
    return sql.Identifier(name)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    raise TypeError(f"range filters need a number, string or boolean, got {type(value).__name__}")


def compile_filter(flt: Optional[Filter]) -> tuple[sql.Composable, list]:
    """Translate a hadamask filter into ``(SQL boolean expression, parameters)``."""
    if flt is None:
        return sql.SQL("TRUE"), []
    if isinstance(flt, Condition):
        return _condition(flt)
    parts, params = [], []
    for item in flt.items:
        expr, p = compile_filter(item)
        parts.append(expr)
        params.extend(p)
    joiner = sql.SQL(" AND ") if isinstance(flt, And) else sql.SQL(" OR ")
    return sql.SQL("(") + joiner.join(parts) + sql.SQL(")"), params


def _eq(field: str, value: Any) -> tuple[sql.Composable, list]:
    return (
        sql.SQL("(payload @> %s::jsonb OR payload @> %s::jsonb)"),
        [json.dumps({field: value}), json.dumps({field: [value]})],
    )


def _condition(c: Condition) -> tuple[sql.Composable, list]:
    if c.op in ("eq", "ne"):
        expr, params = _eq(c.field, c.value)
    elif c.op in ("in", "nin"):
        parts = [_eq(c.field, v) for v in c.value]
        expr = sql.SQL("(") + sql.SQL(" OR ").join(p[0] for p in parts) + sql.SQL(")")
        params = [x for p in parts for x in p[1]]
    else:
        return (
            sql.SQL("(jsonb_typeof(payload -> %s) = %s AND (payload -> %s) {} %s::jsonb)").format(
                sql.SQL(_RANGE[c.op])
            ),
            [c.field, _json_type(c.value), c.field, json.dumps(c.value)],
        )
    if c.op in ("ne", "nin"):
        return sql.SQL("(NOT ") + expr + sql.SQL(")"), params
    return expr, params


class PgVectorStore(ProtectedStore):
    def __init__(self, conn: Union[psycopg.Connection, str], protector: Protector, hnsw: bool = True) -> None:
        super().__init__(protector)
        self.conn = psycopg.connect(conn, autocommit=True) if isinstance(conn, str) else conn
        self.hnsw = hnsw
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def create_collection(self, name: str, dim: int, metric: str = "cosine") -> None:
        self._check_metric(metric)
        table = _ident(name)
        stored = self.protector.stored_dim(dim)
        if self.hnsw and stored > HNSW_MAX_DIM:
            raise ValueError(
                f"pgvector HNSW indexes support at most {HNSW_MAX_DIM} dimensions, but {dim}-d vectors are stored "
                f"as {stored}-d with the hadamard transform. Use Protector(..., transform='dense') to keep {dim} "
                "dimensions, or hnsw=False for exact scans."
            )
        with self.conn.transaction():
            self.conn.execute(
                sql.SQL("CREATE TABLE {} (id TEXT PRIMARY KEY, embedding vector({}) NOT NULL, payload JSONB)").format(
                    table, sql.Literal(self.protector.stored_dim(dim))
                )
            )
            self.conn.execute(sql.SQL("CREATE INDEX ON {} USING GIN (payload jsonb_path_ops)").format(table))
            if self.hnsw:
                self.conn.execute(
                    sql.SQL("CREATE INDEX ON {} USING hnsw (embedding {}) WITH (m = 16, ef_construction = 200)").format(
                        table, sql.SQL(_OPS[metric][1])
                    )
                )
        self.register_collection(name, dim, metric)

    def drop_collection(self, name: str) -> None:
        self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(_ident(name)))

    def _upsert(self, name, ids, vectors, payloads) -> None:
        rows = [
            (str(i), np.asarray(v, dtype=np.float32), Jsonb(p)) for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        query = sql.SQL(
            "INSERT INTO {} (id, embedding, payload) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload"
        ).format(_ident(name))
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(query, rows)

    def _search(self, name, vector, k, flt, with_vectors, params) -> list[RawHit]:
        unknown = set(params) - {"ef_search", "iterative_scan"}
        if unknown:
            raise TypeError(f"unsupported pgvector search options: {sorted(unknown)}")
        ef_search = int(params.get("ef_search", max(100, k)))
        iterative_scan = params.get("iterative_scan", "strict_order")
        if iterative_scan not in _ITERATIVE_SCAN:
            raise ValueError(f"iterative_scan must be one of {_ITERATIVE_SCAN}")
        op, _, score_tpl = _OPS[self._metric(name)]
        where, where_params = compile_filter(flt)
        query = sql.SQL(
            "SELECT id, {score} AS score, payload{vec} FROM {table} WHERE {where} ORDER BY embedding {op} %s LIMIT %s"
        ).format(
            score=sql.SQL(score_tpl.format(col="embedding")),
            vec=sql.SQL(", embedding") if with_vectors else sql.SQL(""),
            table=_ident(name),
            where=where,
            op=sql.SQL(op),
        )
        q = np.asarray(vector, dtype=np.float32)
        with self.conn.transaction(), self.conn.cursor() as cur:
            if self.hnsw:
                # One round trip for both settings. On pgvector < 0.8, hnsw.iterative_scan is just
                # an unused placeholder setting, so no savepoint is needed.
                cur.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true), set_config('hnsw.iterative_scan', %s, true)",
                    [str(ef_search), iterative_scan],
                )
            cur.execute(query, [q, *where_params, q, k])
            rows = cur.fetchall()
        return [RawHit(r[0], float(r[1]), r[2], r[3].to_numpy().tolist() if with_vectors else None) for r in rows]

    def _get(self, name, ids, with_vectors) -> list[RawRecord]:
        rows = self.conn.execute(
            sql.SQL("SELECT id, payload, embedding FROM {} WHERE id = ANY(%s)").format(_ident(name)),
            [[str(i) for i in ids]],
        ).fetchall()
        return [RawRecord(r[0], r[1], r[2].to_numpy().tolist() if with_vectors else None) for r in rows]

    def _delete(self, name, ids) -> None:
        self.conn.execute(sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(_ident(name)), [[str(i) for i in ids]])

    def _delete_where(self, name, flt) -> None:
        where, params = compile_filter(flt)
        self.conn.execute(sql.SQL("DELETE FROM {} WHERE {}").format(_ident(name), where), params)

    def _count(self, name, flt) -> int:
        where, params = compile_filter(flt)
        return self.conn.execute(
            sql.SQL("SELECT count(*) FROM {} WHERE {}").format(_ident(name), where), params
        ).fetchone()[0]

    def _scroll(self, name, flt, batch_size, with_vectors) -> Iterator[RawRecord]:
        where, params = compile_filter(flt)
        query = sql.SQL("SELECT id, payload{vec} FROM {table} WHERE id > %s AND {where} ORDER BY id LIMIT %s").format(
            vec=sql.SQL(", embedding") if with_vectors else sql.SQL(""), table=_ident(name), where=where
        )
        last = ""
        while True:
            rows = self.conn.execute(query, [last, *params, batch_size]).fetchall()
            for r in rows:
                yield RawRecord(r[0], r[1], r[2].to_numpy().tolist() if with_vectors else None)
            if len(rows) < batch_size:
                return
            last = rows[-1][0]

    def close(self) -> None:
        self.conn.close()
