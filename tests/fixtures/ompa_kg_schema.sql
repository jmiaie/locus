-- OMPA knowledge-graph schema, copied verbatim from
-- jmiaie/ompa ompa/knowledge_graph.py (KnowledgeGraph._init_db) at commit 2f306ec.
-- Used by tests/test_locus.py to check that OMPABridge reads OMPA's real table.
-- If OMPA changes its schema, update this file from that function.

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT DEFAULT 'unknown',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL DEFAULT 1.0,
    source_file TEXT,
    extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Basic lookup indexes
CREATE INDEX IF NOT EXISTS idx_triples_subject   ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_object    ON triples(object);

-- Composite indexes for timeline queries and reverse lookups
CREATE INDEX IF NOT EXISTS idx_triples_subject_date
    ON triples(subject, valid_from);
CREATE INDEX IF NOT EXISTS idx_triples_object_pred
    ON triples(object, predicate);
CREATE INDEX IF NOT EXISTS idx_triples_validity
    ON triples(valid_from, valid_to);
