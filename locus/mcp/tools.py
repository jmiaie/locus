"""
Locus MCP tool definitions — 20 tools for vectorless RAG over MCP.
"""

TOOLS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Core RAG
    # ------------------------------------------------------------------
    "locus_index": {
        "description": (
            "Index a file or directory into Locus. Builds BM25 inverted index, "
            "extracts KG triples from wikilinks, tags, and prose sentences. "
            "Skips unchanged files (checksum dedup). Run this first."
        ),
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
    "locus_retrieve": {
        "description": (
            "Five-signal retrieval: BM25 + KG entity expansion + link walking + "
            "frontmatter structural matching + recency prior, fused via weighted RRF. "
            "Each result carries a provenance tag (bm25 / kg / link:hopN / structural / recency)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "as_of": {"type": "string", "description": "Temporal filter YYYY-MM-DD."},
                "use_links": {"type": "boolean", "default": True},
                "full_content": {"type": "boolean", "default": False},
                "intent": {"type": "string", "enum": ["kg_first", "bm25_first", "balanced"]},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["query"],
        },
    },
    "locus_explain": {
        "description": (
            "Explain WHY a chunk was (or would be) retrieved for a query. "
            "Reports which BM25 terms matched, which KG entities link to the document, "
            "and which structural signals fired. Returns a plain-English narrative. "
            "Unique to graph-based RAG — impossible with vector search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "Chunk ID from a prior locus_retrieve call."},
                "query": {"type": "string", "description": "The query to explain against (optional but recommended)."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["chunk_id"],
        },
    },
    # ------------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------------
    "locus_add_fact": {
        "description": "Add a temporal fact (subject–predicate–object) to the KG.",
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
        "description": "Query all KG facts about an entity. Supports as_of temporal filter.",
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
    "locus_kg_stats": {
        "description": "Knowledge graph statistics.",
        "input_schema": {
            "type": "object",
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    "locus_contradictions": {
        "description": (
            "Find contradicting KG triples: same subject+predicate, different objects, "
            "overlapping validity windows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Optional entity to scope."},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_add_alias": {
        "description": "Register an entity alias so variant names resolve to the same canonical entity.",
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
        "description": "Suggest entity name pairs that may refer to the same entity (similarity-based).",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.75},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    # ------------------------------------------------------------------
    # Bulletin / session
    # ------------------------------------------------------------------
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
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    "locus_wrap_up": {
        "description": "Session close: tick bulletin decay, return summary stats.",
        "input_schema": {
            "type": "object",
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------
    "locus_status": {
        "description": "Full status: corpus, KG, bulletin, budget, resolver.",
        "input_schema": {
            "type": "object",
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    "locus_forget": {
        "description": "Remove a document from the corpus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_path": {"type": "string"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["doc_path"],
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
    # ------------------------------------------------------------------
    # Phase 6 — Watch, Bridge, Eval
    # ------------------------------------------------------------------
    "locus_ingest_ompa": {
        "description": (
            "Import an OMPA vault into the Locus store. "
            "Indexes all markdown files and copies KG triples directly from "
            "OMPA's .palace/knowledge_graph.sqlite3 (same schema — zero transformation). "
            "Safe to re-run; unchanged files are skipped via checksum dedup."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vault_path": {"type": "string", "description": "Path to the OMPA vault root."},
                "pattern":    {"type": "string", "default": "**/*.md"},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["vault_path"],
        },
    },
    "locus_benchmark": {
        "description": (
            "Measure retrieval quality against a JSON QA file. "
            "Returns Recall@K and MRR. "
            "QA format: [{\"query\": \"...\", \"expected_docs\": [\"path/to/doc.md\"]}]"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qa_file":  {"type": "string", "description": "Path to JSON QA file."},
                "k_values": {"type": "array", "items": {"type": "integer"}, "default": [1, 3, 5]},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["qa_file"],
        },
    },
    # ------------------------------------------------------------------
    # Phase 5 — Cluster (multi-node)
    # ------------------------------------------------------------------
    "locus_cluster_retrieve": {
        "description": (
            "Query multiple Locus nodes simultaneously and fuse results via RRF. "
            "Provenance tags include the node name (e.g. 'jarv:bm25', 'kai:kg'). "
            "Designed for Jarv/Kai/Tai multi-node setups."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "nodes": {"type": "array", "items": {"type": "string"}, "description": "Subset of node names. Omit for all."},
                "as_of": {"type": "string"},
                "cluster_path": {"type": "string", "description": "Path to cluster registry JSON."},
            },
            "required": ["query", "cluster_path"],
        },
    },
    "locus_add_node": {
        "description": "Register a named node (Locus store) in the cluster registry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Node name (e.g. 'jarv')."},
                "store_path": {"type": "string"},
                "cluster_path": {"type": "string"},
            },
            "required": ["name", "store_path", "cluster_path"],
        },
    },
    "locus_remove_node": {
        "description": "Unregister a node from the cluster (does not delete its store).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "cluster_path": {"type": "string"},
            },
            "required": ["name", "cluster_path"],
        },
    },
    "locus_list_nodes": {
        "description": "List all cluster nodes with their corpus and KG stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_path": {"type": "string"},
            },
            "required": ["cluster_path"],
        },
    },
}
