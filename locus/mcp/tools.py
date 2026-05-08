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
    # ------------------------------------------------------------------
    # Phase 8 — Re-ranking, context packing, cache, prepare_context
    # ------------------------------------------------------------------
    "locus_prepare_context": {
        "description": (
            "All-in-one context preparation for LLM agents. "
            "Retrieves → re-ranks → packs into token budget → assesses confidence → "
            "adds KG facts for query entities. Single call returns everything needed "
            "to answer the query. Recommended primary tool for agent workflows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":        {"type": "string"},
                "limit":        {"type": "integer", "default": 5},
                "token_budget": {"type": "integer", "default": 4000, "description": "Max tokens for packed_context."},
                "rerank":       {"type": "boolean", "default": True, "description": "Apply heuristic re-ranking after RRF."},
                "as_of":        {"type": "string", "description": "Temporal filter YYYY-MM-DD."},
                "store_path":   {"type": "string", "default": ".locus"},
            },
            "required": ["query"],
        },
    },
    "locus_cache_stats": {
        "description": "Query cache statistics: size, hit rate, hits, misses.",
        "input_schema": {
            "type": "object",
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    "locus_clear_cache": {
        "description": "Manually invalidate the query result cache.",
        "input_schema": {
            "type": "object",
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    # ------------------------------------------------------------------
    # Phase 7 — KG traversal, doctor, export
    # ------------------------------------------------------------------
    "locus_kg_traverse": {
        "description": (
            "BFS traversal from a starting entity. Walks the KG graph up to max_depth hops, "
            "optionally filtered by predicate or direction. Returns all reachable entities "
            "and their facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start":            {"type": "string", "description": "Starting entity name."},
                "max_depth":        {"type": "integer", "default": 2},
                "predicate_filter": {"type": "array", "items": {"type": "string"}, "description": "Only follow these predicates."},
                "direction":        {"type": "string", "enum": ["out", "in", "both"], "default": "both"},
                "store_path":       {"type": "string", "default": ".locus"},
            },
            "required": ["start"],
        },
    },
    "locus_kg_match": {
        "description": (
            "Pattern match over the KG with wildcard support. "
            "'*' matches any value. "
            "Examples: match('Alice','*','*') — all Alice facts; "
            "match('*','leads','*') — all leadership; "
            "match('*','works_at','Acme') — everyone at Acme."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":   {"type": "string", "default": "*"},
                "predicate": {"type": "string", "default": "*"},
                "obj":       {"type": "string", "default": "*", "description": "Object pattern ('*' for any)."},
                "as_of":     {"type": "string", "description": "Temporal filter YYYY-MM-DD."},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_doctor": {
        "description": (
            "Run a health check on the Locus store. Checks corpus integrity, KG population, "
            "bulletin fill levels, entity resolver, store size, and version."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"store_path": {"type": "string", "default": ".locus"}},
        },
    },
    "locus_export_kg": {
        "description": (
            "Export the knowledge graph to a file. "
            "Formats: graphml (Gephi/Cytoscape), jsonl (scripting), dot (Graphviz). "
            "Format is auto-detected from the file extension if not specified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string", "description": "Output file path."},
                "format":     {"type": "string", "enum": ["graphml", "jsonl", "dot"], "description": "Export format (auto from extension if omitted)."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["path"],
        },
    },
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
    # ------------------------------------------------------------------
    # Phase 9 — Reasoning, corpus inspection, GitHub bridge
    # ------------------------------------------------------------------
    "locus_reason": {
        "description": (
            "Multi-hop KG reasoning. Extracts entities from the question, "
            "explores their KG neighbourhood, and surfaces connecting paths "
            "between them. Returns reasoning_chains, entity_neighborhood, "
            "and detected entities. Ideal for questions like "
            "'How is Alice connected to the auth system?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question":   {"type": "string"},
                "max_depth":  {"type": "integer", "default": 3},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["question"],
        },
    },
    "locus_find_paths": {
        "description": (
            "Find all shortest KG paths between two entities using BFS. "
            "Returns each path as a list of hops with a human-readable narrative. "
            "Useful for tracing how two people, systems, or concepts are connected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_a":         {"type": "string"},
                "entity_b":         {"type": "string"},
                "max_depth":        {"type": "integer", "default": 3},
                "predicate_filter": {"type": "array", "items": {"type": "string"}},
                "store_path":       {"type": "string", "default": ".locus"},
            },
            "required": ["entity_a", "entity_b"],
        },
    },
    "locus_inspect": {
        "description": (
            "Inspect a document's corpus representation: chunk count, word count, "
            "top BM25 terms, entities, and internal wikilinks. "
            "Useful for debugging indexing quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_path":   {"type": "string"},
                "limit":      {"type": "integer", "default": 20},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["doc_path"],
        },
    },
    "locus_top_terms": {
        "description": (
            "Return the corpus-wide top terms by document frequency. "
            "Each entry includes the term, how many chunks contain it, and total TF. "
            "Useful for understanding what vocabulary dominates the knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit":      {"type": "integer", "default": 20},
                "store_path": {"type": "string", "default": ".locus"},
            },
        },
    },
    "locus_ingest_github": {
        "description": (
            "Fetch markdown files from a GitHub repository and index them. "
            "Fetches the file tree, downloads matching files, and calls locus_index. "
            "Requires a GITHUB_TOKEN env var or token param for private repos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":       {"type": "string", "description": "owner/repo"},
                "branch":     {"type": "string", "default": "main"},
                "path":       {"type": "string", "default": "", "description": "Sub-directory within repo."},
                "pattern":    {"type": "string", "default": "*.md"},
                "token":      {"type": "string", "description": "GitHub PAT (or set GITHUB_TOKEN env var)."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["repo"],
        },
    },
    # ------------------------------------------------------------------
    # Phase 10 — Query intelligence + snapshot
    # ------------------------------------------------------------------
    "locus_expand_query": {
        "description": (
            "Expand a query using KG alias resolution and first-hop neighbours. "
            "If query terms match known entity aliases, their canonical names and "
            "related entities are appended to produce a richer query string. "
            "Returns original, expanded, added_terms, and entity_matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":          {"type": "string"},
                "max_expansions": {"type": "integer", "default": 5},
                "store_path":     {"type": "string", "default": ".locus"},
            },
            "required": ["query"],
        },
    },
    "locus_multi_retrieve": {
        "description": (
            "Run multiple query variants and fuse results via Reciprocal Rank Fusion. "
            "Ideal for query reformulation, typo tolerance, or multi-intent queries. "
            "Each query runs through the full six-signal pipeline independently."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries":    {"type": "array", "items": {"type": "string"}, "description": "List of query variants."},
                "limit":      {"type": "integer", "default": 5},
                "as_of":      {"type": "string", "description": "Temporal filter YYYY-MM-DD."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["queries"],
        },
    },
    "locus_timeline": {
        "description": (
            "Show a chronological timeline of all KG facts for an entity. "
            "Facts are sorted by valid_from date; undated facts appear last. "
            "Useful for tracking how knowledge about an entity evolved over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity":     {"type": "string"},
                "as_of":      {"type": "string", "description": "Temporal filter YYYY-MM-DD."},
                "store_path": {"type": "string", "default": ".locus"},
            },
            "required": ["entity"],
        },
    },
    "locus_snapshot": {
        "description": (
            "Archive the entire Locus store to a portable .tar.gz file. "
            "The snapshot captures the corpus index, KG, bulletin, entity resolver, "
            "and all SQLite databases. Can be restored with locus_restore."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Destination archive path (e.g. backup.tar.gz)."},
                "store_path":  {"type": "string", "default": ".locus"},
            },
            "required": ["output_path"],
        },
    },
    "locus_restore": {
        "description": (
            "Restore a Locus store from a snapshot archive. "
            "Extracts to store_path (default: .locus-restored). "
            "Use overwrite=true to replace an existing store."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "snapshot_path": {"type": "string", "description": "Path to the .tar.gz archive."},
                "store_path":    {"type": "string", "default": ".locus-restored"},
                "overwrite":     {"type": "boolean", "default": False},
            },
            "required": ["snapshot_path"],
        },
    },
    "locus_snapshot_info": {
        "description": "Inspect a snapshot archive: file list, archive size, uncompressed size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "snapshot_path": {"type": "string"},
            },
            "required": ["snapshot_path"],
        },
    },
}
