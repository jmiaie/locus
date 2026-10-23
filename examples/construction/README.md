# Locus for construction and real estate: a walkthrough for GCs and owner's reps

> **Everything in `docs/` is SYNTHETIC.** The project (the "Harbor Point Medical Office Building"), the companies, the people and all the numbers are made up for this demo.
> The set is small: 24 documents and 21 questions. Treat the results as a demonstration of the workflow, not as a benchmark of real project archives.

## The problem this demo shows

On a typical job the answer to "why is the Level 2 slab 7.5 inches?" is spread across several documents:
- an RFI;
- the engineer's response;
- an ASI that revised the spec;
- a change order;
- a submittal;
- a lease clause that made it necessary.

Specs get revised, so "what did the spec say on the day we poured?" is a different question from "what does it say now?".

Locus indexes the project's documents on your own machine. It runs with no cloud service, no GPU and no LLM. Every answer comes with the source file and section it came from.

## What's in the demo

| Folder | Documents |
|---|---|
| `docs/specs/` | 03 30 00 Concrete (Rev 0, Rev 1 via ASI-004), 07 54 23 TPO Roofing (Rev 0, Rev 1 via ASI-009), 08 71 00 Door Hardware (Rev 0, Rev 1 via ASI-011), 21 13 13 Sprinklers |
| `docs/rfis/` | RFI-007 (curtain wall embeds), RFI-011 (MRI slab vibration), RFI-012 (roof membrane), RFI-015 (MRI sprinklers), RFI-019 (Stair 2 delayed egress), RFI-021 (low cylinder breaks) |
| `docs/submittals/` | SUB-03-001 concrete mixes (R&R, then Approved as Noted), SUB-07-004 roofing (Rejected, then Approved), SUB-08-002 door hardware |
| `docs/change-orders/` | CO-002 ($48,600), CO-003 ($31,250, +5 days), CO-005 ($12,940) |
| `docs/permits/` | BP-2026-0418 conditions of approval, Amendment 1 (Saturday hours) |
| `docs/lease/` | Lease abstract, Suite 200 (imaging tenant: delivery conditions, TI, outside date) |
| `docs/meetings/` | OAC meetings 14 and 17 |

Every document has frontmatter: `type`, `date`, and for revisions `revision`, `effective` and `supersedes`.
`facts.json` holds 11 curated, dated facts, such as "Level 2 slab f'c". Each one names its source revision.
`eval_questions.json` holds 21 labelled questions: 15 lookup or RFI-trace questions and 6 point-in-time questions. They were written together with the documents, before any retrieval was run, and were not tuned afterwards.

## Run it

```bash
pip install -e .                                  # from the repo root (or the offline bundle)
python examples/construction/demo.py              # walkthrough + evaluation
python examples/construction/demo.py --eval       # metrics only (JSON)
```

It runs in well under a second on a laptop and makes no network calls.

## What you will see

**1. RFI trace with citations.** The query is *"Level 2 slab MRI vibration change order cost"*. It returns three documents, each with its file, type, date and the matching section:
- CO-002 ($48,600);
- 03 30 00 Rev 1 §2.2 (7.5 in slab for the 2,000 µin/s vibration criterion);
- RFI-011 (the question).

**2. "What did the spec say as of…".** For the same question, the answer depends on the date:
- As of 2026-02-15, the point-in-time view returns 03 30 00 **Rev 0** (4,000 psi).
- As of 2026-04-01 it returns **Rev 1** (5,000 psi, issued by ASI-004 on 2026-03-20).

The view hides documents that didn't exist yet on that date, and revisions that had already been superseded by then. It uses the `effective` and `supersedes` fields in each document.

**3. Permit compliance by date.** For "Saturday work hours":
- In March it returns the original permit condition 4 (no Saturdays).
- In May it returns Amendment 1 (Saturdays 8:00–16:00), along with the OAC minutes that mention it.

**4. Lease exposure.** The Suite 200 lease abstract (Exhibit C) requires a roof warranty of at least 20 years. The OAC 14 minutes connect that requirement to the roofing upgrade.

**5. Timeline from the knowledge graph.** `engine.query_entity("Roof membrane", as_of="2026-04-20")` returns the curated fact *60-mil, 20-year NDL*. As of 2026-06-01 it returns *80-mil, 30-year NDL*. Each answer names its source spec revision and its validity window.

## Results (BM25 default, document level, top 5)

| Question set | n | R@1 | R@5 | MRR |
|---|---|---|---|---|
| All | 21 | 0.667 | 0.952 | 0.775 |
| Lookup and RFI trace | 15 | 0.667 | 0.933 | 0.772 |
| Point-in-time (with the as-of view) | 6 | 0.667 | 1.000 | 0.783 |
| Point-in-time **without** the as-of view (R@1) | 6 | 0.167 | — | — |

Per-question results are in `eval_results.json`.

**Where it falls short:**
- **Q02** ("Why was the Level 2 slab *thickened*…") misses the top 5. The documents say "thickness", and Porter stemming does not reduce *thickened* and *thickness* to the same term. Searching for "thickness" or "vibration" finds it.
- **Short meeting minutes often rank first.** They mention many topics briefly, so they beat the primary document in Q09, Q12 and Q15. The primary document is still in the top 3.
- **Q19 and Q20, as-of questions:** the date filter keeps the right revision in the top 5. The change order or the RFI that *caused* the revision ranks above the spec itself.
- **Recall is not reading.** Locus finds the right documents. A person or an LLM still has to read them and decide.

## Taking it to a real project

1. **Export the documents to text.** Put specs, RFIs and the rest into a folder as Markdown or text files, one per document. PDF-to-text conversion is not part of Locus.
2. **Add frontmatter for dates and revisions.** Add `type`, `date`, and for revisions `revision`, `effective` and `supersedes`. A document-control export (Procore, ACC, Aconex) usually has these fields.
3. **Run it on-prem.** Index with `locus index <folder>` or run the container (see "On-prem deployment" in the main README), then connect an MCP client.
4. **Keep it inside your network.** Nothing leaves your network unless you connect an LLM yourself.
