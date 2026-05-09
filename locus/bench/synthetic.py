"""
Synthetic corpus and QA pair generator for benchmarking.

Generates realistic markdown documents with:
  - YAML frontmatter (date, tags, type)
  - Headed sections with topical prose
  - Wikilinks between related documents
  - Named entities for KG extraction
  - Varied vocabulary per topic domain

QA pairs are derived directly from document content so ground-truth
is deterministic and reproducible without human annotation.
"""

from __future__ import annotations

import hashlib
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Topic definitions — (slug, primary_terms, entities, related_topics)
# ---------------------------------------------------------------------------

_TOPICS: list[tuple[str, list[str], list[str], list[str]]] = [
    (
        "authentication",
        ["JWT", "OAuth", "token", "session", "login", "credential", "password", "identity"],
        ["AuthService", "TokenIssuer", "Alice"],
        ["security", "api_gateway"],
    ),
    (
        "deployment",
        ["Kubernetes", "Docker", "container", "pod", "replica", "rollout", "manifest"],
        ["DevOps", "ClusterManager", "Bob"],
        ["monitoring", "networking"],
    ),
    (
        "database",
        ["PostgreSQL", "schema", "migration", "index", "query", "transaction", "replica"],
        ["DBAdmin", "SchemaRegistry", "Carol"],
        ["caching", "authentication"],
    ),
    (
        "monitoring",
        ["Prometheus", "Grafana", "metric", "alert", "dashboard", "latency", "SLO"],
        ["OnCallTeam", "AlertManager", "Dave"],
        ["deployment", "incident_response"],
    ),
    (
        "security",
        ["encryption", "TLS", "certificate", "firewall", "policy", "audit", "RBAC"],
        ["SecurityTeam", "PolicyEngine", "Eve"],
        ["authentication", "networking"],
    ),
    (
        "api_gateway",
        ["rate-limit", "routing", "proxy", "endpoint", "middleware", "load-balancer"],
        ["GatewayService", "RateLimiter", "Frank"],
        ["authentication", "networking"],
    ),
    (
        "caching",
        ["Redis", "cache", "TTL", "invalidation", "eviction", "hit-rate", "warm"],
        ["CacheLayer", "RedisCluster", "Grace"],
        ["database", "api_gateway"],
    ),
    (
        "incident_response",
        ["runbook", "postmortem", "escalation", "triage", "on-call", "SLA", "MTTR"],
        ["IncidentCommander", "OnCallRotation", "Henry"],
        ["monitoring", "deployment"],
    ),
    (
        "networking",
        ["VPC", "subnet", "DNS", "firewall", "peering", "ingress", "egress"],
        ["NetOps", "DNSResolver", "Iris"],
        ["security", "deployment"],
    ),
    (
        "data_pipeline",
        ["Kafka", "stream", "batch", "ETL", "schema", "consumer", "producer", "offset"],
        ["DataEngineer", "StreamProcessor", "Jack"],
        ["database", "monitoring"],
    ),
    (
        "release_process",
        ["changelog", "semver", "tag", "branch", "merge", "review", "approval", "deploy"],
        ["ReleaseManager", "CISystem", "Karen"],
        ["deployment", "testing"],
    ),
    (
        "testing",
        ["unit-test", "integration", "coverage", "fixture", "mock", "assertion", "flaky"],
        ["QATeam", "TestRunner", "Leo"],
        ["release_process", "deployment"],
    ),
]

_PROSE_TEMPLATES = [
    "{entity0} leads the {topic} initiative. The system uses {term0} and {term1} "
    "to handle all {topic} operations. See also [[{related0}]] for details.",

    "The {topic} service is owned by {entity0}. Configuration relies on {term0} "
    "with fallback to {term1}. Related documentation: [[{related0}]], [[{related1}]].",

    "{entity0} and {entity1} collaborate on {topic}. Key components include "
    "{term0}, {term1}, and {term2}. Escalate issues to {entity0} via {term3}.",

    "Our {topic} architecture uses {term0} at the core. {entity1} manages "
    "provisioning. For {term1} configuration, refer to [[{related0}]].",

    "{term0} and {term1} are the primary tools for {topic}. "
    "{entity0} is responsible for maintenance. SLA targets are tracked in [[{related1}]].",
]


@dataclass
class QAPair:
    query: str
    expected_doc: str
    topic: str

    def to_dict(self) -> dict:
        return {"query": self.query, "expected_docs": [self.expected_doc], "topic": self.topic}


class SyntheticCorpus:
    """Generate a reproducible synthetic corpus with ground-truth QA pairs."""

    def __init__(self, num_docs: int = 50, seed: int = 42) -> None:
        self.num_docs = num_docs
        self.seed = seed
        self._rng = random.Random(seed)
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self.doc_dir: Path | None = None
        self.qa_pairs: list[QAPair] = []
        self._docs: list[tuple[str, str]] = []  # (slug, content)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, target_dir: Path | None = None) -> Path:
        """Generate the corpus. Returns the directory containing .md files."""
        if target_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="locus_bench_")
            self.doc_dir = Path(self._tmpdir.name)
        else:
            self.doc_dir = Path(target_dir)
            self.doc_dir.mkdir(parents=True, exist_ok=True)

        self._docs = []
        self.qa_pairs = []

        topic_cycle = [_TOPICS[i % len(_TOPICS)] for i in range(self.num_docs)]

        for i, (slug, terms, entities, related) in enumerate(topic_cycle):
            doc_slug = f"{slug}_{i:03d}"
            content = self._make_doc(doc_slug, slug, terms, entities, related, i)
            path = self.doc_dir / f"{doc_slug}.md"
            path.write_text(content, encoding="utf-8")
            self._docs.append((doc_slug, content))
            self.qa_pairs.extend(self._make_qa(doc_slug, slug, terms))

        return self.doc_dir

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_doc(
        self,
        doc_slug: str,
        topic: str,
        terms: list[str],
        entities: list[str],
        related: list[str],
        idx: int,
    ) -> str:
        year = 2023 + (idx % 3)
        month = (idx % 12) + 1
        day = (idx % 28) + 1

        tags = [topic, terms[0].lower(), self._rng.choice(["ops", "dev", "infra", "platform"])]
        doc_type = self._rng.choice(["guide", "runbook", "reference", "policy"])

        # Related topic slugs → wikilink targets (approximate)
        rel_slugs = [f"{r}_000" for r in related[:2]]

        template = self._rng.choice(_PROSE_TEMPLATES)
        prose = template.format(
            topic=topic.replace("_", " "),
            entity0=entities[0] if entities else "TeamLead",
            entity1=entities[1] if len(entities) > 1 else "Engineer",
            term0=terms[0],
            term1=terms[1] if len(terms) > 1 else terms[0],
            term2=terms[2] if len(terms) > 2 else terms[0],
            term3=terms[3] if len(terms) > 3 else terms[1],
            related0=rel_slugs[0] if rel_slugs else "overview_000",
            related1=rel_slugs[1] if len(rel_slugs) > 1 else rel_slugs[0] if rel_slugs else "overview_000",
        )

        # Second paragraph — more terms
        extra_terms = terms[1:4]
        para2 = (
            f"Implementation details: {', '.join(extra_terms)} are configured "
            f"according to {entities[0] if entities else 'the team'} standards. "
            f"Refer to [[{rel_slugs[0] if rel_slugs else 'overview_000'}]] for architecture context."
        )

        return (
            f"---\n"
            f"date: {year:04d}-{month:02d}-{day:02d}\n"
            f"tags: {', '.join(tags)}\n"
            f"type: {doc_type}\n"
            f"---\n\n"
            f"# {topic.replace('_', ' ').title()} — {doc_slug}\n\n"
            f"{prose}\n\n"
            f"## Details\n\n"
            f"{para2}\n"
        )

    def _make_qa(self, doc_slug: str, topic: str, terms: list[str]) -> list[QAPair]:
        pairs: list[QAPair] = []
        # Query 1: primary topic phrase
        pairs.append(QAPair(
            query=f"how does {topic.replace('_', ' ')} work",
            expected_doc=f"{doc_slug}.md",
            topic=topic,
        ))
        # Query 2: key term
        pairs.append(QAPair(
            query=f"{terms[0]} configuration",
            expected_doc=f"{doc_slug}.md",
            topic=topic,
        ))
        # Query 3: term combination (harder)
        if len(terms) >= 2:
            pairs.append(QAPair(
                query=f"{terms[0]} {terms[1]}",
                expected_doc=f"{doc_slug}.md",
                topic=topic,
            ))
        return pairs
