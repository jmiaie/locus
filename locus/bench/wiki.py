"""
EngineeringWiki — a realistic interconnected knowledge-base corpus.

Unlike the simple SyntheticCorpus (homogeneous templates, no real links),
this corpus mimics a real engineering wiki:

  - Documents reference each other via actual wikilinks to existing slugs
  - Named entities (people, teams, services) appear across multiple documents
  - KG-extractable prose: "Alice leads the Auth team", "Redis replaced Memcached"
  - Temporal spread: 2022–2025 dates so recency signal has something to weight
  - Four document types: guide, runbook, incident, reference
  - Hub documents (high inbound link count) that link_popularity can rank on

Query types in generated QA pairs:
  term     — keyword queries (BM25 baseline)
  entity   — ask about a named person/service (KG signal advantage)
  navigate — follow a wikilink chain (link-walk advantage)
  temporal — ask for recent/dated content (recency + structural advantage)
  type     — filter by doc type (structural advantage)
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared entities that appear across documents — these feed the KG signal
# ---------------------------------------------------------------------------

PEOPLE = [
    ("Alice",   "Auth team",      "AuthService"),
    ("Bob",     "DevOps team",    "DeployPipeline"),
    ("Carol",   "Database team",  "PostgresCluster"),
    ("Dave",    "Monitoring",     "AlertManager"),
    ("Eve",     "Security",       "PolicyEngine"),
    ("Frank",   "Platform",       "APIGateway"),
    ("Grace",   "Data platform",  "KafkaBroker"),
    ("Henry",   "Incident team",  "OnCallSystem"),
    ("Iris",    "Networking",     "DNSResolver"),
    ("Jack",    "Release eng",    "CISystem"),
]

SERVICES = [s for _, _, s in PEOPLE]
TEAM_LEADS = {s: p for p, _, s in PEOPLE}

# ---------------------------------------------------------------------------
# Topic domains — each maps to a document cluster
# ---------------------------------------------------------------------------

DOMAINS: list[dict] = [
    {
        "slug":    "auth",
        "title":   "Authentication",
        "terms":   ["JWT", "OAuth2", "token", "session", "login", "credential"],
        "owner":   "Alice",
        "service": "AuthService",
        "links_to": ["security", "api_gateway"],
        "type":    "guide",
    },
    {
        "slug":    "deployment",
        "title":   "Deployment",
        "terms":   ["Kubernetes", "Helm", "rollout", "replica", "pod", "manifest"],
        "owner":   "Bob",
        "service": "DeployPipeline",
        "links_to": ["monitoring", "networking"],
        "type":    "runbook",
    },
    {
        "slug":    "database",
        "title":   "Database",
        "terms":   ["PostgreSQL", "migration", "schema", "index", "transaction"],
        "owner":   "Carol",
        "service": "PostgresCluster",
        "links_to": ["caching", "monitoring"],
        "type":    "reference",
    },
    {
        "slug":    "monitoring",
        "title":   "Monitoring",
        "terms":   ["Prometheus", "Grafana", "metric", "alert", "SLO", "latency"],
        "owner":   "Dave",
        "service": "AlertManager",
        "links_to": ["incident", "deployment"],
        "type":    "guide",
    },
    {
        "slug":    "security",
        "title":   "Security",
        "terms":   ["TLS", "certificate", "RBAC", "policy", "audit", "firewall"],
        "owner":   "Eve",
        "service": "PolicyEngine",
        "links_to": ["auth", "networking"],
        "type":    "reference",
    },
    {
        "slug":    "api_gateway",
        "title":   "API Gateway",
        "terms":   ["rate-limit", "routing", "proxy", "middleware", "load-balancer"],
        "owner":   "Frank",
        "service": "APIGateway",
        "links_to": ["auth", "monitoring"],
        "type":    "guide",
    },
    {
        "slug":    "caching",
        "title":   "Caching",
        "terms":   ["Redis", "TTL", "invalidation", "eviction", "hit-rate", "warm"],
        "owner":   "Grace",
        "service": "KafkaBroker",
        "links_to": ["database", "api_gateway"],
        "type":    "reference",
    },
    {
        "slug":    "incident",
        "title":   "Incident Response",
        "terms":   ["runbook", "postmortem", "escalation", "triage", "on-call", "MTTR"],
        "owner":   "Henry",
        "service": "OnCallSystem",
        "links_to": ["monitoring", "deployment"],
        "type":    "runbook",
    },
    {
        "slug":    "networking",
        "title":   "Networking",
        "terms":   ["VPC", "subnet", "DNS", "ingress", "egress", "peering"],
        "owner":   "Iris",
        "service": "DNSResolver",
        "links_to": ["security", "deployment"],
        "type":    "reference",
    },
    {
        "slug":    "release",
        "title":   "Release Process",
        "terms":   ["changelog", "semver", "tag", "merge", "approval", "hotfix"],
        "owner":   "Jack",
        "service": "CISystem",
        "links_to": ["deployment", "monitoring"],
        "type":    "runbook",
    },
]

_DOMAIN_BY_SLUG = {d["slug"]: d for d in DOMAINS}

# ---------------------------------------------------------------------------
# QA pair with query type for per-type recall analysis
# ---------------------------------------------------------------------------

@dataclass
class WikiQAPair:
    query: str
    expected_doc: str
    query_type: str   # term | entity | navigate | temporal | type

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected_docs": [self.expected_doc],
            "query_type": self.query_type,
        }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class EngineeringWiki:
    """
    Generates a realistic engineering wiki corpus.

    Each domain produces multiple document variants:
      - A primary guide/reference (dated 2022–2023, stable)
      - A runbook (dated 2023–2024)
      - A recent update (dated 2024–2025, preferred by recency signal)
      - Incident postmortems linking back to affected services

    Total: ~40–50 documents depending on num_domains.
    """

    def __init__(self, num_domains: int | None = None, seed: int = 42) -> None:
        self._domains = DOMAINS[:num_domains] if num_domains else DOMAINS
        self._rng = random.Random(seed)
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self.doc_dir: Path | None = None
        self.qa_pairs: list[WikiQAPair] = []
        self._doc_slugs: list[str] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, target_dir: Path | None = None) -> Path:
        if target_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="locus_wiki_")
            self.doc_dir = Path(self._tmpdir.name)
        else:
            self.doc_dir = Path(target_dir)
            self.doc_dir.mkdir(parents=True, exist_ok=True)

        self._doc_slugs = []
        self.qa_pairs = []

        # Pass 1: determine all slugs so wikilinks can reference real docs
        all_slugs: list[str] = []
        for domain in self._domains:
            all_slugs.extend([
                f"{domain['slug']}_overview",
                f"{domain['slug']}_runbook",
                f"{domain['slug']}_recent",
            ])
        all_slugs.append("incident_auth_outage")
        all_slugs.append("incident_deploy_rollback")
        all_slugs.append("platform_overview")

        self._doc_slugs = all_slugs

        # Pass 2: write files
        for domain in self._domains:
            self._write_overview(domain)
            self._write_runbook(domain)
            self._write_recent(domain)

        self._write_incident("incident_auth_outage",   "auth",       2024)
        self._write_incident("incident_deploy_rollback", "deployment", 2023)
        self._write_platform_overview()

        # Pass 3: generate typed QA pairs
        self._generate_qa()

        return self.doc_dir

    def cleanup(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ------------------------------------------------------------------
    # Document writers
    # ------------------------------------------------------------------

    def _write_overview(self, domain: dict) -> None:
        slug = f"{domain['slug']}_overview"
        year = self._rng.choice([2022, 2023])
        month = self._rng.randint(1, 12)
        links = self._linked_slugs(domain, variant="overview")
        terms = domain["terms"]
        owner = domain["owner"]
        service = domain["service"]

        body = (
            f"# {domain['title']} Overview\n\n"
            f"{owner} leads the {domain['title'].lower()} initiative. "
            f"The {service} handles all {domain['title'].lower()} operations "
            f"using {terms[0]} and {terms[1]}.\n\n"
            f"## Architecture\n\n"
            f"{service} is owned by {owner}. Key components: {', '.join(terms[:3])}. "
            f"See also: {self._wikilinks(links)}.\n\n"
            f"## Configuration\n\n"
            f"{terms[0]} configuration is managed by {owner}. "
            f"For operational procedures refer to [[{domain['slug']}_runbook]].\n"
        )
        self._write(slug, domain["type"], domain["slug"], year, month, body)

    def _write_runbook(self, domain: dict) -> None:
        slug = f"{domain['slug']}_runbook"
        year = self._rng.choice([2023, 2024])
        month = self._rng.randint(1, 12)
        links = self._linked_slugs(domain, variant="runbook")
        terms = domain["terms"]
        owner = domain["owner"]
        service = domain["service"]

        body = (
            f"# {domain['title']} Runbook\n\n"
            f"**Owner:** {owner}  **Service:** {service}\n\n"
            f"## Deployment steps\n\n"
            f"1. Verify {terms[0]} configuration\n"
            f"2. Check {terms[1]} status\n"
            f"3. Notify {owner} before changes\n\n"
            f"## Troubleshooting\n\n"
            f"Common issues with {service}: {terms[2]} failures, {terms[0]} misconfiguration. "
            f"Escalate to {owner}. Related: {self._wikilinks(links)}.\n\n"
            f"## Rollback\n\n"
            f"To rollback {service}: revert {terms[0]} settings and alert {owner}.\n"
        )
        self._write(slug, "runbook", domain["slug"], year, month, body)

    def _write_recent(self, domain: dict) -> None:
        slug = f"{domain['slug']}_recent"
        year = self._rng.choice([2024, 2025])
        month = self._rng.randint(1, 12)
        terms = domain["terms"]
        owner = domain["owner"]
        service = domain["service"]

        body = (
            f"# {domain['title']} — Recent Changes\n\n"
            f"Updated {year} by {owner}.\n\n"
            f"## Changes\n\n"
            f"{service} migrated from legacy {terms[-1]} to {terms[0]}. "
            f"{owner} led the migration. "
            f"New configuration uses {terms[1]} and {terms[2]}.\n\n"
            f"## Impact\n\n"
            f"Performance improved after {terms[0]} upgrade. "
            f"See [[{domain['slug']}_overview]] for background.\n"
        )
        self._write(slug, "changelog", domain["slug"], year, month, body)

    def _write_incident(self, slug: str, domain_slug: str, year: int) -> None:
        domain = _DOMAIN_BY_SLUG.get(domain_slug, DOMAINS[0])
        owner = domain["owner"]
        service = domain["service"]
        terms = domain["terms"]
        month = self._rng.randint(1, 12)

        body = (
            f"# Incident: {service} Outage {year}\n\n"
            f"**Incident commander:** {owner}\n"
            f"**Affected service:** {service}\n\n"
            f"## Summary\n\n"
            f"{service} experienced an outage due to {terms[0]} misconfiguration. "
            f"{owner} led the response. MTTR: 47 minutes.\n\n"
            f"## Timeline\n\n"
            f"- Alert fired in [[monitoring_overview]]\n"
            f"- {owner} paged via [[incident_runbook]]\n"
            f"- Root cause: {terms[1]} configuration drift\n\n"
            f"## Remediation\n\n"
            f"Applied {terms[0]} fix. Updated [[{domain_slug}_runbook]] with prevention steps.\n"
        )
        self._write(slug, "incident", domain_slug, year, month, body)

    def _write_platform_overview(self) -> None:
        slug = "platform_overview"
        services_list = ", ".join(SERVICES[:5])
        links = ["auth_overview", "deployment_overview", "monitoring_overview"]
        body = (
            f"# Platform Overview\n\n"
            f"Frank leads the Platform team. The platform consists of: {services_list}.\n\n"
            f"## Service Ownership\n\n"
            + "\n".join(
                f"- {service}: owned by {TEAM_LEADS.get(service, 'TBD')}"
                for service in SERVICES
            )
            + f"\n\n## Key Systems\n\n"
            f"See: {self._wikilinks(links)}.\n"
        )
        self._write(slug, "reference", "platform", 2024, 3, body)

    def _write(self, slug: str, doc_type: str, tag: str, year: int, month: int, body: str) -> None:
        content = (
            f"---\n"
            f"date: {year:04d}-{month:02d}-01\n"
            f"tags: {tag}, {doc_type}\n"
            f"type: {doc_type}\n"
            f"---\n\n"
            + body
        )
        (self.doc_dir / f"{slug}.md").write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # QA generation — typed by which signal should help
    # ------------------------------------------------------------------

    def _generate_qa(self) -> None:
        pairs: list[WikiQAPair] = []

        for domain in self._domains:
            s = domain["slug"]
            t = domain["terms"]
            o = domain["owner"]
            svc = domain["service"]
            doc_type = domain["type"]

            # Term queries — BM25 baseline
            pairs.append(WikiQAPair(f"how does {domain['title'].lower()} work", f"{s}_overview.md", "term"))
            pairs.append(WikiQAPair(f"{t[0]} configuration", f"{s}_overview.md", "term"))
            pairs.append(WikiQAPair(f"{t[0]} {t[1]}", f"{s}_overview.md", "term"))

            # Entity queries — KG signal advantage
            pairs.append(WikiQAPair(f"who owns {svc}", f"{s}_overview.md", "entity"))
            pairs.append(WikiQAPair(f"{o} responsibilities", f"{s}_overview.md", "entity"))
            pairs.append(WikiQAPair(f"{svc} configuration", f"{s}_overview.md", "entity"))

            # Navigate queries — link-walk advantage
            # Query matches the overview; correct answer is a linked runbook
            pairs.append(WikiQAPair(f"{domain['title'].lower()} runbook procedures", f"{s}_runbook.md", "navigate"))
            pairs.append(WikiQAPair(f"{t[0]} rollback steps", f"{s}_runbook.md", "navigate"))

            # Temporal queries — recency signal advantage
            # Recent docs are the correct answer for "latest" queries
            pairs.append(WikiQAPair(f"latest {domain['title'].lower()} changes", f"{s}_recent.md", "temporal"))
            pairs.append(WikiQAPair(f"recent {t[0]} migration", f"{s}_recent.md", "temporal"))

            # Type queries — structural signal advantage
            pairs.append(WikiQAPair(f"{doc_type} for {domain['title'].lower()}", f"{s}_overview.md", "type"))

        # Incident queries
        pairs.append(WikiQAPair("AuthService outage postmortem", "incident_auth_outage.md", "entity"))
        pairs.append(WikiQAPair("deployment rollback incident", "incident_deploy_rollback.md", "entity"))
        pairs.append(WikiQAPair("Frank platform team ownership", "platform_overview.md", "entity"))

        self.qa_pairs = pairs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _linked_slugs(self, domain: dict, variant: str) -> list[str]:
        """Return valid existing slugs that this document should link to."""
        result = []
        for linked in domain.get("links_to", [])[:2]:
            if linked in _DOMAIN_BY_SLUG:
                result.append(f"{linked}_overview")
        return result

    @staticmethod
    def _wikilinks(slugs: list[str]) -> str:
        if not slugs:
            return "(none)"
        return " ".join(f"[[{s}]]" for s in slugs)
