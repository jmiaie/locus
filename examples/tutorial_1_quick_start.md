# Tutorial 1: Quick Start — Index Your First Knowledge Base

In this tutorial, you'll index a folder of markdown documents and run your first retrieval queries.

## What You'll Build

A local knowledge base with:
- 10 sample markdown files about a fictional API
- Full-text search + knowledge graph extraction
- Explainable retrieval (see *why* results were returned)

## Prerequisites

- Python 3.11+
- `pip install locus-rag`

## Step 1: Create Sample Documents

Create a folder `my-docs/` with these files:

**my-docs/01-auth.md**
```markdown
---
date: 2025-01-15
type: guide
tags: security, authentication
---

# Authentication Guide

## JWT Tokens

JWT tokens are the primary auth mechanism. Alice developed the JWT validation layer
and it's now maintained by the Auth team.

- Alice leads the Auth team
- Tokens are validated on every request
- Token expiry is 24 hours
```

**my-docs/02-oauth.md**
```markdown
---
date: 2025-01-20
type: guide
tags: security, third-party
---

# OAuth Integration

OAuth 2.0 flow for third-party app integrations.

Alice and Bob collaborated on the OAuth implementation. It replaced the legacy
token-based system that Bob originally built.

- OAuth is the recommended path for partners
- Bob originally built auth/legacy-token.md
- Alice replaced that with OAuth
```

**my-docs/03-deployment.md**
```markdown
---
date: 2025-02-01
type: runbook
tags: operations, deployment
---

# Deployment Runbook

## Pre-deployment checklist

1. Run test suite
2. Update CHANGELOG
3. Tag release

## Deployment steps

Alice documents all deployment procedures. The system depends on both JWT
and OAuth working correctly before deploying.
```

(Create a few more files in the same pattern.)

## Step 2: Index the Documents

Open a Python shell or create `index_docs.py`:

```python
from locus import LocusEngine

# Create engine pointing to .locus folder
engine = LocusEngine(store_path=".locus")

# Index all markdown files
result = engine.index("./my-docs", pattern="*.md")

print(f"✓ Indexed {result['files']} files")
print(f"✓ Created {result['chunks']} chunks")
print(f"✓ Extracted {result['triples']} KG facts")

# Check health
print(engine.status())
```

Run it:
```bash
python index_docs.py
```

You should see:
```
✓ Indexed 3 files
✓ Created 12 chunks
✓ Extracted 47 KG facts
```

## Step 3: Retrieve and Explain

```python
from locus import LocusEngine

engine = LocusEngine(store_path=".locus")

# Query 1: Keyword search
query = "how does authentication work?"
chunks = engine.retrieve(query, limit=3)

for i, chunk in enumerate(chunks, 1):
    print(f"\n[{i}] {chunk.doc_path}")
    print(f"    Via: {chunk.provenance}")
    print(f"    Score: {chunk.score:.4f}")
    print(f"    {chunk.content[:150]}...")

# Query 2: Explain why a specific chunk was returned
print("\n" + "="*60)
print("Why was this chunk retrieved?")
print("="*60)

explanation = engine.explain(chunks[0].chunk_id, query=query)
print(explanation["narrative"])
print(f"\nMatched BM25 terms: {explanation.get('bm25_matched_terms', [])}")
print(f"Matched KG entities: {explanation.get('kg_matched_entities', [])}")
```

Output:
```
[1] my-docs/01-auth.md
    Via: bm25
    Score: 0.0521
    JWT tokens are the primary auth mechanism. Alice developed the JWT validation layer
    and it's now maintained by the Auth team....

============================================================
Why was this chunk retrieved?
============================================================
Chunk from 'my-docs/01-auth.md' (section: 'JWT Tokens'). 
Retrieved because: BM25: terms [authentication, jwt] appear in this chunk; 
KG: entity Alice links to this document.

Matched BM25 terms: ['authentication', 'jwt']
Matched KG entities: ['Alice']
```

## Step 4: Entity-Centric Query

Knowledge graphs excel at entity-based queries:

```python
# Who works on what?
chunks = engine.retrieve("who leads the auth team?", limit=5)
for chunk in chunks:
    print(f"✓ {chunk.doc_path}: {chunk.provenance}")

# Check the KG directly
facts = engine.query_entity("Alice")
print(f"\nAlice facts:")
for fact in facts["facts"][:5]:
    print(f"  - {fact['subject']} --{fact['predicate']}--> {fact['object']}")
```

Output:
```
✓ my-docs/01-auth.md: kg
✓ my-docs/02-oauth.md: kg

Alice facts:
  - Alice --leads--> Auth team
  - Alice --works_at--> Company
  - Alice --develops--> JWT validation layer
```

## Step 5: Debug Mode

Enable debug tracing to understand retrieval decisions:

```python
from locus import LocusEngine
from locus.debug import DebugTracer

engine = LocusEngine(store_path=".locus")
tracer = DebugTracer(enabled=True)

# Use tracer within engine (integration in core.py coming in next phase)
query = "authentication mechanisms"
with tracer.start_trace(query, "bm25_first") as trace:
    chunks = engine.retrieve(query, limit=5)
    tracer.finish_trace(trace, chunks, total_elapsed_ms=45, cache_hit=False)

# View summary
print(tracer.report())
# Output:
# {
#   "traces_count": 1,
#   "avg_latency_ms": 45.2,
#   "p50_latency_ms": 45.2,
#   "p95_latency_ms": 45.2,
#   "signal_usage": {"bm25": 1, "kg": 1, "link_walk": 1, ...},
#   "log_dir": ".locus/debug_logs"
# }
```

Debug logs are saved as JSON files in `.locus/debug_logs/` for later analysis.

## Step 6: Pagination

Retrieve in batches without loading everything into memory:

```python
# Get first 5 results
page1 = engine.retrieve("deployment", limit=5)

# Get next 5 (when pagination is fully integrated)
# page2 = engine.retrieve("deployment", limit=5, offset=5)

for chunk in page1:
    print(f"{chunk.doc_path}: {chunk.content[:100]}...")
```

## What You Learned

✓ Indexing markdown files  
✓ Running six-signal retrieval  
✓ Explaining why results were returned  
✓ Entity-centric queries via KG  
✓ Debug tracing for analysis  

## Next Steps

- **Tutorial 2:** [Building a Support Chatbot](tutorial_2_chatbot.md) — Turn your KB into a Q&A system
- **Tutorial 3:** [Using Locus with Cursor](tutorial_3_cursor.md) — Integrate into your IDE
- **Docs:** [Architecture Deep Dive](../docs/architecture.md) — Understand the six signals

## Troubleshooting

**Q: Low recall on my queries?**  
A: Check `engine.doctor()` — ensure you have enough chunks and entities extracted.

**Q: Slow retrieval?**  
A: Enable debug mode to see which signal is causing latency. Consider disabling link-walking for large corpora.

**Q: Why is this chunk ranked #1?**  
A: Use `engine.explain(chunk_id, query)` to see all signals that contributed.
