# Tutorial 2: Building a Support Chatbot

In this tutorial, you'll build a simple chatbot that answers questions from your knowledge base.

## Scenario

You have documentation about an API. Users ask questions, and you want answers grounded in that documentation.

## Architecture

```
User Question
     ↓
Locus Retrieve (find relevant docs)
     ↓
Format Context (pack docs into token budget)
     ↓
LLM (Claude, GPT, etc.) with Context
     ↓
Answer with Sources
```

## Step 1: Prepare Your Knowledge Base

Use the docs from Tutorial 1, or create new ones. The quality of answers depends on the KB.

```bash
mkdir -p kb/
# Add your markdown files to kb/
```

## Step 2: Create a Simple Chatbot

Create `chatbot.py`:

```python
from locus import LocusEngine
import json


class SimpleKBChatbot:
    """Answer questions from a knowledge base."""
    
    def __init__(self, kb_path: str):
        self.engine = LocusEngine(store_path=".locus")
        self.kb_path = kb_path
        
        # Index if needed
        if self.engine.status()["corpus"]["doc_count"] == 0:
            print(f"Indexing {kb_path}...")
            result = self.engine.index(kb_path)
            print(f"✓ Indexed {result['files']} files, {result['chunks']} chunks")
    
    def answer(self, question: str, use_llm: bool = False) -> dict:
        """
        Answer a question from the KB.
        
        Args:
            question: User question
            use_llm: If True, would call an LLM (implementation varies)
        
        Returns:
            Dict with answer, sources, and confidence
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.engine.retrieve(question, limit=5)
        
        # Step 2: Assess confidence
        confidence = self.engine.assess_confidence(chunks)
        
        # Step 3: Format context
        context = self.engine.format_context(chunks, include_hot=True)
        
        # Step 4: Create response
        answer = self._synthesize_answer(question, chunks, use_llm=use_llm)
        
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": [
                {
                    "doc": c.doc_path,
                    "via": c.provenance,
                    "excerpt": c.content[:200],
                }
                for c in chunks[:3]
            ],
            "context": context,
        }
    
    def _synthesize_answer(
        self,
        question: str,
        chunks: list,
        use_llm: bool = False,
    ) -> str:
        """
        Create an answer from retrieved chunks.
        
        With use_llm=True, you'd call Claude/GPT here.
        For now, we extract a simple answer.
        """
        if not chunks:
            return "I couldn't find information about that."
        
        if use_llm:
            # This is where you'd call your LLM API
            # For now, we'll skip it
            return self._llm_answer(question, chunks)
        
        # Simple extraction: return first chunk with context
        chunk = chunks[0]
        lines = chunk.content.split("\n")
        answer_snippet = "\n".join(lines[:3])
        
        return f"Based on {chunk.doc_path}:\n\n{answer_snippet}"
    
    def _llm_answer(self, question: str, chunks: list) -> str:
        """Call an LLM for synthesis (stub)."""
        # Example: Call Claude via anthropic package
        # This requires ANTHROPIC_API_KEY environment variable
        #
        # from anthropic import Anthropic
        # client = Anthropic()
        # context = "\n\n".join([c.content for c in chunks[:3]])
        # response = client.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     max_tokens=500,
        #     system="You are a helpful support agent. Answer based on the provided documentation.",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": f"Documentation:\n{context}\n\nQuestion: {question}"
        #         }
        #     ]
        # )
        # return response.content[0].text
        
        return "LLM integration not configured"
    
    def explain(self, question: str) -> None:
        """Show why a question was answered the way it was."""
        chunks = self.engine.retrieve(question, limit=1)
        if not chunks:
            print("No results to explain.")
            return
        
        explanation = self.engine.explain(chunks[0].chunk_id, query=question)
        print(f"\n=== Explanation for: {question} ===\n")
        print(explanation["narrative"])
        print(f"\nBM25 terms: {explanation.get('bm25_matched_terms', [])}")
        print(f"KG entities: {explanation.get('kg_matched_entities', [])}")
        print(f"\nDocument: {explanation['doc_path']}")
        print(f"Section: {explanation.get('section', 'N/A')}")


# Demo
if __name__ == "__main__":
    chatbot = SimpleKBChatbot("./my-docs")
    
    questions = [
        "How does authentication work?",
        "Who leads the auth team?",
        "What's the deployment process?",
    ]
    
    for q in questions:
        print("\n" + "="*70)
        print(f"Q: {q}")
        print("="*70)
        
        response = chatbot.answer(q)
        print(f"A: {response['answer']}")
        print(f"\nConfidence: {response['confidence']['level']}")
        print(f"Sources:")
        for src in response['sources']:
            print(f"  - {src['doc']} ({src['via']})")
        
        # Show why this answer
        chatbot.explain(q)
```

Run it:
```bash
python chatbot.py
```

Output:
```
======================================================================
Q: How does authentication work?
======================================================================
A: Based on my-docs/01-auth.md:

JWT tokens are the primary auth mechanism. Alice developed the JWT validation layer
and it's now maintained by the Auth team.

Confidence: ok
Sources:
  - my-docs/01-auth.md (bm25)
  - my-docs/02-oauth.md (kg)

=== Explanation for: How does authentication work? ===

Chunk from 'my-docs/01-auth.md' (section: 'JWT Tokens'). 
Retrieved because: BM25: terms [authentication, jwt] appear in this chunk; 
KG: entity Alice links to this document.

BM25 terms: ['authentication', 'jwt']
KG entities: ['Alice']

Document: my-docs/01-auth.md
Section: JWT Tokens
```

## Step 3: With LLM Integration (Optional)

To use Claude for answer synthesis:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Then modify the chatbot:

```python
def answer(self, question: str, use_llm: bool = True) -> dict:
    chunks = self.engine.retrieve(question, limit=5)
    confidence = self.engine.assess_confidence(chunks)
    
    # With LLM enabled
    answer = self._synthesize_answer(question, chunks, use_llm=True)
    
    return {
        "question": question,
        "answer": answer,  # Now from Claude
        "confidence": confidence,
        "sources": [...],
    }
```

Now when you run it, Claude will synthesize intelligent answers grounded in your KB.

## Step 4: Streaming Responses

For production, you'd want streaming (progressive results):

```python
def answer_stream(self, question: str):
    """Stream answer chunks as they become available."""
    
    chunks = self.engine.retrieve(question, limit=5)
    
    # Yield metadata first
    yield {
        "type": "metadata",
        "confidence": self.engine.assess_confidence(chunks)["level"],
        "doc_count": len(chunks),
    }
    
    # In future, stream LLM responses incrementally
    answer = self._synthesize_answer(question, chunks)
    for line in answer.split("\n"):
        yield {
            "type": "answer_line",
            "text": line,
        }
    
    # Yield sources
    for src in chunks[:3]:
        yield {
            "type": "source",
            "doc": src.doc_path,
            "via": src.provenance,
        }
```

Use in a web server:

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class ChatHandler(BaseHTTPRequestHandler):
    chatbot = SimpleKBChatbot("./my-docs")
    
    def do_POST(self):
        if self.path != "/ask":
            return
        
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        question = body["question"]
        
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        
        # Stream responses as newline-delimited JSON
        for response_chunk in self.chatbot.answer_stream(question):
            self.wfile.write(json.dumps(response_chunk).encode() + b"\n")

server = HTTPServer(("localhost", 8000), ChatHandler)
server.serve_forever()
```

## Step 5: Evaluation

Measure chatbot quality against expected answers:

```python
from locus import LocusEngine

engine = LocusEngine()

# Prepare QA pairs
qa_pairs = [
    {
        "question": "How does authentication work?",
        "expected_docs": ["my-docs/01-auth.md"]
    },
    {
        "question": "Who leads the auth team?",
        "expected_docs": ["my-docs/01-auth.md"]
    },
]

# Evaluate
results = engine.benchmark(qa_pairs)
print(results)
# Output:
# {
#   "Recall@1": 0.95,
#   "Recall@3": 1.0,
#   "MRR": 0.98,
#   "misses": []
# }
```

## What You Learned

✓ End-to-end retrieval pipeline  
✓ Confidence assessment  
✓ Context formatting for LLMs  
✓ LLM integration (optional)  
✓ Streaming responses  
✓ Evaluation methodology  

## Next Steps

- **Deploy to production** — Containerize chatbot, add logging
- **Add conversation history** — Track multi-turn dialogues
- **Collect feedback** — Let users rate answers; improve ranking
- **Tutorial 3:** [Using Locus with Cursor](tutorial_3_cursor.md)
