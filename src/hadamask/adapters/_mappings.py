"""Index mappings and filter compilation shared by the Elasticsearch and OpenSearch adapters."""

from __future__ import annotations

from typing import Optional

from ..core import Protector
from ..metadata import BLIND_PREFIX
from .base import And, Condition, Filter


def dynamic_templates(protector: Protector) -> list[dict]:
    """Index only what can be filtered: blind tokens and plaintext strings.

    Ciphertexts are stored in ``_source`` but neither indexed nor kept in doc values: the
    database can do nothing useful with them, and indexing them only costs space.
    First matching template wins, so the specific ones come first.
    """
    templates: list[dict] = [{"hadamask_blind": {"match": f"{BLIND_PREFIX}*", "mapping": {"type": "keyword"}}}]
    for i, field in enumerate(sorted(protector.policy.plaintext)):
        templates.append(
            {f"hadamask_plain_{i}": {"match": field, "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
        )
    templates.append(
        {
            "hadamask_ciphertext": {
                "match_mapping_type": "string",
                "mapping": {"type": "keyword", "index": False, "doc_values": False},
            }
        }
    )
    return templates


#: Copy of the record id kept in ``_source`` as a sortable keyword, so ``_scroll`` can page with
#: ``search_after`` (sorting on ``_id`` is disabled by default in Elasticsearch 8).
ID_FIELD = "_hm_id"


def compile_filter(flt: Optional[Filter]) -> dict:
    """Translate a hadamask filter into the Elasticsearch/OpenSearch query DSL.

    ``$ne``/``$nin`` use ``must_not``, so (as in MongoDB) records without the field match.
    ``None`` becomes ``match_all``.
    """
    if flt is None:
        return {"match_all": {}}
    if isinstance(flt, Condition):
        return _condition(flt)
    parts = [compile_filter(i) for i in flt.items]
    if isinstance(flt, And):
        return {"bool": {"filter": parts}}
    return {"bool": {"should": parts, "minimum_should_match": 1}}


def _condition(c: Condition) -> dict:
    if c.op in ("eq", "ne"):
        clause = {"term": {c.field: c.value}}
    elif c.op in ("in", "nin"):
        clause = {"terms": {c.field: list(c.value)}}
    else:
        clause = {"range": {c.field: {c.op: c.value}}}
    return {"bool": {"must_not": [clause]}} if c.op in ("ne", "nin") else clause
