*This project has been created as part of the 42 curriculum by bbeaurai.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about
the [vLLM](https://github.com/vllm-project/vllm) codebase (v0.10.1).

The pipeline ingests the repository (Python code, Markdown, reStructuredText
and plain-text documentation), chunks it with type-specific strategies,
indexes it with BM25, retrieves the most relevant excerpts for a question,
and generates a grounded natural-language answer with the
`Qwen/Qwen3-0.6B` model. Retrieval quality is measured with recall@k
against annotated ground-truth datasets.

## Instructions

Requirements: [uv](https://docs.astral.sh/uv/) (it installs the pinned
Python version and all dependencies itself).

```bash
make install          # uv sync — creates .venv and installs dependencies
make index            # build the BM25 index from data/raw/vllm-0.10.1
make search Q="How to configure OpenAI server?"
make answer Q="How to configure OpenAI server?"
make search_dataset   # retrieve sources for the whole docs dataset
make answer_dataset   # generate LLM answers for the retrieved sources
make evaluate         # recall@k on BOTH docs and code datasets
make evaluate_one TYPE=code   # recall@k on a single dataset (docs|code)
make lint             # flake8 + mypy (subject flags)
make clean            # remove caches ; make fclean also removes index/outputs
```

The unzipped vLLM repository is expected under `data/raw/vllm-0.10.1`
(the indexer tells you where to put it if it is missing).

## System Architecture

```
                 ┌────────────┐   ┌─────────┐   ┌────────────┐
 vLLM repo ────► │  chunker   │──►│  BM25   │──►│  storage   │  (index)
 (.py .md ...)   │ (per type) │   │ (bm25s) │   │ data/      │
                 └────────────┘   └─────────┘   │ processed/ │
                                                └─────┬──────┘
                                                      │
 question ──► tokenize ──► BM25 retrieve ──► top-k sources  (search)
                                                      │
 top-k excerpts ──► context window ──► Qwen3-0.6B ──► answer  (answer)
                                                      │
 retrieved sources + ground truth ──► recall@k  (evaluate)
```

- `src/chunker.py` — file filtering and the two chunking strategies
- `src/indexing.py` — repository ingestion, BM25 index build/load
- `src/retriever.py` — BM25 search and ranking
- `src/llm.py` — prompt construction and answer generation
- `src/evaluation.py` — recall@k computation (5% overlap rule)
- `src/main.py` — Python Fire CLI exposing `index`, `search`,
  `search_dataset`, `answer`, `answer_dataset`, `evaluate`
- All data exchanged between stages is validated by the pydantic models
  of `src/pydantic.py` (the models required by the subject).

## Chunking Strategy

Two strategies, selected by file extension (`.py` vs `.md`/`.rst`/`.txt`):

- **Python (AST-based)**: the file is parsed with `ast`; every top-level
  statement (function, class, imports, module docstring) becomes a block,
  decorators included. Consecutive blocks are merged while they fit in
  `max_chunk_size`, so a chunk keeps whole definitions together. Blocks
  larger than the limit are re-split with the text splitter, and files
  that do not parse fall back to text chunking.
- **Text (recursive)**: `RecursiveCharacterTextSplitter` (LangChain) with
  a 200-character overlap, splitting on paragraph, then line, then word
  boundaries.

The maximum chunk size is **2000 characters**, configurable with
`index --max_chunk_size N`. Too-small chunks fragment definitions and
paragraphs (a chunk no longer carries enough context to score well and
the 5% overlap with a ground-truth source becomes harder to reach);
too-large chunks dilute the BM25 term statistics and waste the LLM
context window with noise.

## Retrieval Method

BM25 (via `bm25s`), with English stopword removal at tokenization time.
For a query, the top `10 × k` candidate chunks are retrieved, then walked
in descending score order keeping **at most 3 chunks per file** until `k`
results are selected. This cap keeps results diverse across files while
still allowing several excerpts of one file to be returned — important
for code questions, whose ground-truth sources often live in a single
module. Each result is returned as
`(file_path, first_character_index, last_character_index)`.

## Performance Analysis

Measured on the 100-question public and private datasets (k = 10):

| Dataset       | Recall@5  | Recall@10 | Threshold |
|---------------|-----------|-----------|-----------|
| docs public   | **0.850** | 0.930     | ≥ 0.80    |
| docs private  | **0.850** | 0.890     | ≥ 0.80    |
| code public   | **0.550** | 0.600     | ≥ 0.50    |
| code private  | **0.520** | 0.590     | ≥ 0.50    |

System performance (required limits in parentheses):

- Indexing: ~4 s for 1952 files / 14 109 chunks (limit: 5 min)
- Retrieval throughput: 100 questions run in well under a second, far
  below the 90 s / 200 questions limit

## Design Decisions

- **BM25 over TF-IDF**: better length normalization and term saturation,
  and `bm25s` gives a fast, persistable index with no service to run.
- **Per-file result cap of 3** instead of one-chunk-per-file
  deduplication: measured +6 points of docs recall@5 and +7 points of
  code recall@5 versus the strict-dedup baseline, because multi-source
  questions are no longer limited to one excerpt per file.
- **AST chunking for Python**: chunk boundaries follow definitions, so a
  retrieved excerpt is a coherent unit of code instead of an arbitrary
  2000-character window.
- **Plain tokenization kept**: identifier splitting (snake_case /
  camelCase) and English stemming were both tested; they raised code
  recall (up to 0.61) but dropped docs recall below the 80% threshold,
  so they were rejected.
- **Answer prompting**: the retrieved excerpts are concatenated into the
  context within a character budget, and the system prompt (with
  `/no_think`) forces short, self-contained, context-only answers.

## Challenges Faced

- **Character-exact chunk offsets**: recall is computed on character
  ranges, so every chunk must carry exact `first/last_character_index`
  values — the AST chunker converts line numbers to character offsets
  and sub-splitting re-offsets indices relative to the parent block.
- **Docs/code trade-off**: almost every tokenization trick that improved
  code retrieval degraded docs retrieval; the fix that helped both was
  relaxing the per-file deduplication, not the tokenization.
- **Small-model faithfulness**: Qwen3-0.6B hallucinates easily; a strict
  system prompt and disabling its thinking mode keep answers grounded in
  the retrieved context.

## Example Usage

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure OpenAI server?" --k 10
uv run python -m src answer "How to configure OpenAI server?" --k 10
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10
```

## Bonus

- **Index caching**: the BM25 index and chunk metadata are persisted
  under `data/processed/` at indexing time and transparently reloaded by
  every later command, so search never re-ingests the repository.

## Resources

- [Github fcaval](https://github.com/fcaval42/RAG_AgainstTheMachine)
- [Chunking : découper vos documents pour le RAG](https://blog.stephane-robert.info/docs/developper/programmation/python/rag-chunking/)
- [RAG : augmenter un LLM avec vos données](https://blog.stephane-robert.info/docs/developper/programmation/python/rag-introduction/)
- [What is recall-at-k?](https://milvus.io/ai-quick-reference/what-is-recallatk)
- [What is recall-at-k?](https://medium.com/@dev.aguillin/abstract-syntax-tree-python-85d39a53e86d)
- [Qu’est-ce que BM25](https://www.luigisbox.fr/glossaire-recherche/bm25/#:~:text=BM25%2C%20ou%20Best%20Match%2025,de%20leur%20score%20de%20pertinence)
- [transformers](https://huggingface.co/docs/transformers/fr/quicktour)
- [Fire](https://github.com/google/python-fire)
- [Fire guide](https://google.github.io/python-fire/guide/)
- [Regex example](https://regex101.com/)

## 🤖 AI Usage

Artificial Intelligence was utilized during the development of this project to enhance efficiency, maintain high code quality, and assist with technical decisions. Specifically, AI tools were used for the following tasks:

* **Project Architecture:** Assisting in the design, layout, and organization of the project's initial structure.
* **Code Standards & Quality:** Refactoring code to ensure strict compliance with **`mypy`** (static type checking) and **`flake8`** (linting) standards.
* **Documentation:** Generating and refining standardized docstrings for modules, classes, and functions to improve codebase readability.
* **Technical Guidance:** Explaining complex concepts and recommending technologies that best aligned with the overall vision of the project.
