# iknow v1 — Claude lane

`iknow_v1_claude` is the independent Claude review/proposal lane forked from `iknow_v1` (originally
bootstrapped by Codex). It carries the same evidence-first method and Phase 1 discipline; governance
and self-guidance are now maintained by Claude. The upstream record under `docs/history/` is
preserved verbatim as an immutable audit log.

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

Current Douglass Phase 1 status: `v3_chained_cross_page_continuation_guarded` is the active
paragraph merge policy after formal gold, side-effect review, false-join blocking, and validation
gates. It keeps `cp_000103` fixed, blocks the rejected pages 59-61 false join, preserves 7/7 accepted
chained-join decisions, and improves gold paragraph precision/recall to 1.000. Downstream retrieval,
embeddings, reasoning, and graph work remain blocked. The recalculated canonical layer has 115
promoted paragraphs, 129 review warnings, 72 risky paragraphs, and `safe_for_downstream: false`.

## Documentation Model

Local durable documentation is HTML-first.

GitHub does not render local HTML as a rich repository landing page, so this `README.md` exists only as a GitHub-facing orientation layer. The canonical local docs remain in `docs/`.

Start here locally:

- [`docs/index.html`](docs/index.html)
- [`docs/governance/0002-source-of-truth.html`](docs/governance/0002-source-of-truth.html)
- [`docs/governance/0003-claude-self-guidance.html`](docs/governance/0003-claude-self-guidance.html)
- [`docs/strategy/0001-phase-1-strategy.html`](docs/strategy/0001-phase-1-strategy.html)

## Local Observability

The Claude lane runs a dependency-free localhost dashboard with severity levels, run correlation,
search, faceted filters, and run grouping.

Run (Claude-lane port `8799`):

```bash
python3 src/claude_observability_server.py --host 127.0.0.1 --port 8799
```

Open:

```text
http://127.0.0.1:8799
```

Append a structured event:

```bash
python3 src/observe.py phase "Started Phase 1 extraction design" \
  --level milestone --book douglass_narrative --run phase1_v3 --phase A \
  --details '{"note":"design"}'
```

Events carry `schema, timestamp, level, actor, kind, book_id, run_id, phase, git_commit, message,
details` (`git_commit` is stamped automatically). The legacy/Codex dashboard remains on port `8765`
(`src/observability_server.py`); the Claude lane never reuses 8765 so both can run at once. Runtime
observability logs are ignored by git.

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
