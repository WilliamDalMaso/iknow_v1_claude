# iknow v1

`iknow_v1` is a fresh rebuild of a book-intelligence engine.

The goal is not to rush into a chatbot or graph UI. The goal is to first prove that a PDF can be transformed into clean, complete, organized, auditable book content. Retrieval, reasoning, and knowledge graph construction only become trustworthy after that foundation is correct.

## North Star

Turn one uploaded book into a clean, organized, evidence-backed learning engine with a knowledge graph.

The first version is intentionally one-book-first. We will not automate broadly until Phase 1 extraction quality is strong enough to trust.

## Current Priority

Phase 1 is the center of the project:

1. Preserve the original PDF and source metadata.
2. Extract every page without silently losing content.
3. Separate raw extraction from cleaned text.
4. Identify document objects such as headings, paragraphs, tables, figures, captions, footnotes, lists, quotes, and unknown objects.
5. Build a candidate reading order with page and object provenance.
6. Record cleanup decisions, omissions, uncertainty, and review needs.
7. Audit cleanliness before retrieval or graph construction.

Phase 1 outputs are candidate-first. Nothing is called canonical until it has passed audit, comparison, and review.

Paragraph merge policy changes also require a tracked gold review set. Heuristic warning counts can
show symptoms, but they are not enough to adopt a new extraction policy.

Current Douglass Phase 1 status: `v2_cross_page_continuation` is the active paragraph merge policy
after formal gold, side-effect review, blocker investigation, and validation gates. Downstream
retrieval, embeddings, reasoning, and graph work remain blocked until canonical paragraph review is
safe. The current post-adoption blocker is `bbox_span_risk`; gold coverage has been expanded for the
former 6 gold-set gap cases. The front-matter/metadata review found 7 valid Preface/Letter
paragraphs and 2 valid Chapter I paragraphs caught by an overbroad early-page warning. Downstream is
still blocked. The visual-review queue found 7 valid canonical paragraphs and 1 true grouping defect
on page 109. The `v3_chained_cross_page_continuation` experiment fixed that defect and improved gold
paragraph precision/recall to 1.000 without object-label regression, but it proposed 9 chained joins
and 8 are outside authoritative gold coverage. It is not active; the next gate is side-effect review,
not adoption.

## Documentation Model

Local durable documentation is HTML-first.

GitHub does not render local HTML as a rich repository landing page, so this `README.md` exists only as a GitHub-facing orientation layer. The canonical local docs remain in `docs/`.

Start here locally:

- [`docs/index.html`](docs/index.html)
- [`docs/governance/0002-source-of-truth.html`](docs/governance/0002-source-of-truth.html)
- [`docs/governance/0003-codex-self-guidance.html`](docs/governance/0003-codex-self-guidance.html)
- [`docs/strategy/0001-phase-1-strategy.html`](docs/strategy/0001-phase-1-strategy.html)

## Local Observability

The repo includes a dependency-free black-and-white localhost dashboard for watching project/process events during development.

Run:

```bash
python3 src/observability_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Append an event:

```bash
python3 src/observe.py codex "Started Phase 1 extraction design" --details '{"phase":"phase_1"}'
```

Runtime observability logs are ignored by git.

## Local API Key

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` locally. `.env` is ignored by git.

Model names are intentionally not defaulted. When a model-assisted step is added, the selected model must be explicit and documented.

## Verification

Install the Phase 1 development/test dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

Run the standard verification command before committing code or artifact-contract changes:

```bash
python3 -m pytest tests
```

If `pytest` is unavailable in a constrained environment, a direct test-function runner may be used only as a temporary fallback. The standard command remains `python3 -m pytest tests`.

## Repository Shape

- `docs/`: HTML project memory, decisions, lessons, strategy, and history.
- `src/`: reusable engine code.
- `tests/`: automated checks for extraction, cleanup, and artifact contracts.
- `data/books/`: local source-book inputs and book setup files.
- `data/runs/`: generated per-book/per-run artifacts.

## Hard Rules

- Do not optimize for a chatbot before the extracted book is clean.
- Do not let retrieval hide extraction failures.
- Do not silently drop PDF content.
- Do not treat LLM output as ground truth without provenance and review.
- Do not widen to many books before one book passes Phase 1 gates.
- Do not spend expensive model calls without recording why the model was chosen.
- Do not adopt paragraph merge policy changes from heuristic warning improvement alone.
- Keep durable local documentation in HTML; use Markdown only where GitHub needs it.
