"""
Locus MCP tool definitions — 12 tools for vectorless RAG over MCP.
"""

TOOLS: dict[str, dict] = {
    "locus_index": {
        "description": (
            "Index a file or directory into Locus. Builds the BM25 inverted index "
            "and extracts KG triples from wikilinks and tags. Run this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path to index."},
                "pattern": {"type": "string", "default": "**/*.md", "description": "Glob pattern for directory indexing."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["path"],
        },
    },
    "locus_retrieve": {
        "description": (
            "Retrieve context for a query using BM25 + KG entity expansion + link walking, "
            "fused via Reciprocal Rank Fusion. Returns ranked chunks with provenance tags "
            "(bm25 / kg / link:hopN) so you can see why each result was returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query."},
                "limit": {"type": "integer", "default": 5, "description": "Max chunks to return."},
                "as_of": {"type": "string", "description": "Temporal filter — only facts valid on this date (YYYY-MM-DD)."},
                "use_links": {"type": "boolean", "default": True, "description": "Follow wikilinks from top hits."},
                "full_content": {"type": "boolean", "default": False, "description": "Return full chunk content instead of truncating at 600 chars."},
                "intent": {"type": "string", "enum": ["kg_first", "bm25_first", "balanced"], "description": "Override auto-detected query intent."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["query"],
        },
    },
    "locus_add_fact": {
        "description": (
            "Add a temporal fact (subject–predicate–object triple) to the Locus knowledge graph. "
            "Optionally scoped with valid_from / valid_to dates for point-in-time queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "valid_from": {"type": "string", "description": "Start date YYYY-MM-DD."},
                "valid_to": {"type": "string", "description": "End date YYYY-MM-DD."},
                "source": {"type": "string", "description": "Source document path."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["subject", "predicate", "object"],
        },
    },
    "locus_query_entity": {
        "description": (
            "Query all knowledge graph facts about an entity. "
            "Supports temporal filter via as_of to ask 'what was true on date X?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "as_of": {"type": "string", "description": "Historical date (YYYY-MM-DD)."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["entity"],
        },
    },
    "locus_hot_context": {
        "description": (
            "Return the current hot-tier bulletin board — frequently retrieved chunks "
            "and pinned context. Useful for always-on background knowledge."
        ),
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
        "description": (
            "Warm context open: corpus stats, KG stats, and hot-tier content. "
            "Run at the start of a session to orient the agent (~1K tokens)."
        ),
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
        "description": "Full status: corpus stats, KG stats, bulletin tiers, token budget.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_forget": {
        "description": "Remove a document from the Locus corpus by its indexed path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_path": {"type": "string", "description": "Document path as indexed (relative path)."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["doc_path"],
        },
    },
    "locus_kg_stats": {
        "description": "Knowledge graph statistics: triple count, entity count, source count.",
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
                "path": {"type": "string", "description": "Root directory to reindex."},
                "pattern": {"type": "string", "default": "**/*.md"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["path"],
        },
    },
}
