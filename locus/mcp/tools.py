"""
Locus MCP tool definitions — 15 tools for vectorless RAG over MCP.
"""

TOOLS: dict[str, dict] = {
    "locus_index": {
        "description": (
            "Index a file or directory into Locus. Builds the BM25 inverted index, "
            "extracts KG triples from wikilinks, tags, and prose sentences. "
            "Skips unchanged files automatically (checksum dedup). Run this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path to index."},
                "pattern": {"type": "string", "default": "**/*.md"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["path"],
        },
    },
    "locus_retrieve": {
        "description": (
            "Five-signal retrieval: BM25 + KG entity expansion + link walking + "
            "frontmatter structural matching + recency prior, fused via weighted RRF. "
            "Query intent is auto-classified (KG-first / BM25-first / balanced). "
            "Each result carries a provenance tag explaining why it was returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "as_of": {"type": "string", "description": "Temporal filter (YYYY-MM-DD)."},
                "use_links": {"type": "boolean", "default": True},
                "full_content": {"type": "boolean", "default": False, "description": "Return full chunk text instead of truncating at 600 chars."},
                "intent": {"type": "string", "enum": ["kg_first", "bm25_first", "balanced"], "description": "Override auto-detected query intent."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["query"],
        },
    },
    "locus_add_fact": {
        "description": "Add a temporal fact (subject–predicate–object) to the knowledge graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "valid_from": {"type": "string"},
                "valid_to": {"type": "string"},
                "source": {"type": "string"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["subject", "predicate", "object"],
        },
    },
    "locus_query_entity": {
        "description": "Query all KG facts about an entity. Supports temporal filter via as_of.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "as_of": {"type": "string"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["entity"],
        },
    },
    "locus_hot_context": {
        "description": "Return the hot-tier bulletin board — frequently retrieved and pinned context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "token_limit": {"type": "integer", "default": 1500},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_promote": {
        "description": "Manually pin a chunk to Tier 0 (never auto-archived).",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["chunk_id"],
        },
    },
    "locus_session_start": {
        "description": "Warm context open: corpus + KG + bulletin + resolver stats, hot-tier content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_wrap_up": {
        "description": "Session close: tick bulletin decay, return summary stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_status": {
        "description": "Full status: corpus, KG, bulletin, budget, entity resolver.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_forget": {
        "description": "Remove a document from the Locus corpus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_path": {"type": "string"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["doc_path"],
        },
    },
    "locus_kg_stats": {
        "description": "Knowledge graph statistics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_sync": {
        "description": "Full reindex: wipe and rebuild corpus + KG from a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "pattern": {"type": "string", "default": "**/*.md"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["path"],
        },
    },
    "locus_contradictions": {
        "description": (
            "Find contradicting KG triples: same subject+predicate, different objects, "
            "overlapping validity windows. Scope to a specific entity or scan all."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Optional entity to scope the search."},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_add_alias": {
        "description": (
            "Register an entity alias so variant names resolve to the same canonical entity. "
            "e.g. alias='Jeff', canonical='Jeff Milam'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "alias": {"type": "string"},
                "canonical": {"type": "string"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["alias", "canonical"],
        },
    },
    "locus_suggest_aliases": {
        "description": (
            "Suggest entity pairs that may refer to the same thing, based on name similarity. "
            "Returns candidates for human review — use locus_add_alias to confirm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.75, "description": "Similarity threshold (0–1)."},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
}
