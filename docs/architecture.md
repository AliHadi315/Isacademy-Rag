# Architecture

## Layers

```
UI            app/ui/streamlit_app.py + pages.py        bilingual, no logic
   │
PIPELINE      app/agents/pipeline.py                    one function, 5 steps
   │
AGENTS        query_agent · retrieval_agent · analysis_agent
   │
CAPABILITIES  rag/ (chunk, embed, store, search)
              vision/ (find the image, Gemini reads it)
              analytics/ (semantic PCA, dataset PCA, clustering, plots)
              reports/ (Markdown, HTML, PDF)
   │
PROVIDERS     models/gemini.py       the only AI provider
              rag/embeddings.py      local embedding model
              rag/vector_store.py    ChromaDB
   │
FOUNDATION    config.py · i18n.py · models/schemas.py · utils/
```

Dependencies point downward. A page never imports Chroma; an agent never
imports Streamlit.

## Why a function, not an agent framework

`ask()` in `app/agents/pipeline.py` is 60 lines and runs:

1. query agent → what kind of question is this?
2. retrieval agent → which passages?
3. vision (if visual) → which image, and what does Gemini see in it?
4. semantic PCA → where does the question sit among the documents?
5. analysis agent → Gemini writes the grounded answer.

Every page calls that one function, so the behaviour is identical everywhere
and the whole system can be explained in a minute. LangGraph, CrewAI and
friends would add vocabulary, not capability, at this size.

## Typed contracts

`app/models/schemas.py` defines what crosses each boundary:

| Stage | Model |
|---|---|
| query agent | `QueryPlan` (intent, figure_hint, page_hint, flags) |
| retrieval | `RetrievalResult` (results, threshold, sufficient) |
| vision | `VisualAnswer` (image_path, page, analysis) |
| semantic PCA | `SemanticPCAResult` (points, clusters, closest, explanation) |
| dataset PCA | `PCAResult` |
| whole run | `AnswerResult` |

## The grounding chain

1. Extraction attaches `document_name` + `page_number` to every artefact.
2. Chunking propagates them, and adds `image_path` for figures.
3. Chroma stores them as metadata beside the vector.
4. Retrieval returns them on every `SearchResult`.
5. The Gemini prompt shows passages as `[1] doc | page N | similarity S`.
6. **Citations are assembled in Python** from those results — Gemini never
   writes a page number, so it cannot invent one.
7. The report is built from the same `AnswerResult`.

## Failure model

| Failure | Behaviour |
|---|---|
| No API key | Retrieval, sources and PCA still run; the answer says a key is missing |
| Gemini quota / bad key / bad model | Specific bilingual message, no fabricated answer |
| Corrupt or empty PDF | Reported on the Documents page; other files still index |
| One bad page | Logged and skipped; the rest of the document indexes |
| Nothing above the threshold | An explicit "not enough information" answer |
| Too few vectors for PCA | Map skipped with a message; the question is still answered |
| No extractable figure | The page is rendered instead |
| Dataset unusable for PCA | `performed=False` plus the reason |
| Missing reportlab | Markdown and HTML still render |
| Any page raising | Caught in `streamlit_app.py`, shown with a traceback expander |
