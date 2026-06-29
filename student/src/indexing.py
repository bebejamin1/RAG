#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   indexing.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 11:28:05 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/29 14:13:06 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import os
import bm25s
import pickle
import colorama as c
import time

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List

from student.src.chunker import chunker, is_indexable

IGNORED_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv",
                "build", "dist", ".mypy_cache", ".pytest_cache"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INDEX_DIR = f"{_PROJECT_ROOT}/data/processed"
BM25_PATH = f"{_PROJECT_ROOT}/data/processed/bm25_index"
CHUNKS_PATH = f"{_PROJECT_ROOT}/data/processed/chunks/chunks.pkl"


# *****************************************************************************
# *                            LOAD INDEX                                     *
# *                                                                           *

def load_index() -> Tuple[bm25s.BM25, List[Tuple[str, int, int]]]:
    """Load BM25 index and chunk metadata from disk.

    Returns:
        Tuple of (BM25 retriever, list of (file_path, start, end) metadata).

    Raises:
        SystemExit: If the index or metadata files are missing.
    """

    try:
        if (not os.path.exists(BM25_PATH)):
            raise FileNotFoundError(
                "BM25 index not found. Run:\n"
                "  uv run python -m student index"
                                   )

        if (not os.path.exists(CHUNKS_PATH)):
            raise FileNotFoundError(
                "Chunk metadata not found. Run:\n"
                "  uv run python -m student index"
                                   )

        retriever = bm25s.BM25.load(BM25_PATH, load_corpus=False)
        with open(CHUNKS_PATH, "rb") as f:
            chunk_metadata = pickle.load(f)

    except FileNotFoundError as e:
        print(e)
        exit(1)

    return (retriever, chunk_metadata)


# *****************************************************************************
# *                            LOAD FILES                                     *
# *                                                                           *

def load_files(path_dir: str) -> List[Tuple[str, str]]:
    """Recursively load all indexable text files from a directory.

    Skips hidden directories, known tooling dirs, and non-text extensions.

    Args:
        path_dir: Root directory to walk.

    Returns:
        List of (relative_path, content) tuples.
    """
    files = []

    for path in Path(path_dir).rglob("*"):
        if (not path.is_file()):
            continue

        if (any(part.startswith(".") or part in IGNORED_DIRS
                for part in path.parts)):
            continue

        if (not is_indexable(str(path))):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()

            try:
                rel = str(path.resolve().relative_to(_PROJECT_ROOT))
            except ValueError:
                rel = str(path)

            files.append((rel, content))

        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

    return (files)


# *****************************************************************************
# *                              SEARCH                                       *
# *                                                                           *

def search(
    query: str, k: int
          ) -> List[Tuple[str, int, int]]:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        query: The search query string.
        k: Number of results to return.

    Returns:
        List of (file_path, first_character_index, last_character_index)
        tuples, ordered by BM25 relevance score descending.
        Returns an empty list if the query is empty or k is 0.
    """
    if (not query or k <= 0):
        return []

    retriever, chunk_metadata = load_index()

    tokenized_query = bm25s.tokenize(
        [query], stopwords="en", show_progress=False
                                    )

    n = min(k, len(chunk_metadata))
    results, _ = retriever.retrieve(tokenized_query, k=n)

    indices = results[0].tolist()

    return [chunk_metadata[i] for i in indices]


# *****************************************************************************
# *                            INDEX MAIN                                     *
# *                                                                           *

def index_main(path_dir: str, max_chunk_size: int) -> None:
    """Build the BM25 index and persist it to disk.

    Walks path_dir, chunks every indexable file, tokenises the corpus,
    fits a BM25 model, and saves both the index and the chunk metadata.

    Args:
        path_dir: Root of the repository/directory to index.
        max_chunk_size: Maximum characters per chunk.
    """
    try:

        start_time = time.perf_counter()

        os.makedirs(BM25_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

        print(f"  {c.Fore.CYAN}Reading files from: {path_dir}"
              f"{c.Style.RESET_ALL}\n")
        files = load_files(path_dir)
        print(f"  Found {c.Fore.CYAN}{len(files)}{c.Style.RESET_ALL} "
              "indexable files.\n")

        if (not files):
            path_vllm = path_dir[:path_dir.find("/vllm-0.10.1")]
            os.makedirs(path_vllm, exist_ok=True)
            raise ValueError(
                f"No indexable files found in: {path_dir}\n"
                f"Add the unzipped vllm-0.10.1 file to the {path_vllm} folder"
                            )

    except (PermissionError, OSError) as e:
        raise ValueError(f"Failed to read files: {e}")

    all_chunks = []
    for file_path, content in tqdm(files, desc="Chunking files",
                                   colour="cyan"):
        chunks = chunker(file_path, content, max_chunk_size)
        all_chunks.extend(chunks)

    print(f"\n  Created {c.Fore.CYAN}{len(all_chunks)}{c.Style.RESET_ALL}"
          " total chunks.\n")

    print(f"  {c.Fore.CYAN}Tokenising corpus...{c.Style.RESET_ALL}")
    chunks_text = [chunk[3] for chunk in all_chunks]
    tokenized_text = bm25s.tokenize(
        chunks_text, stopwords="en", show_progress=False
                                   )

    valid_indices = [i for i, ids in enumerate(tokenized_text.ids)
                     if len(ids) > 0]

    if not valid_indices:
        tokenized_text = bm25s.tokenize(
            chunks_text,
            stopwords=None,
            token_pattern=r"(?u)\b\w+\b",
            show_progress=False,
                                       )
        valid_indices = [i for i, ids in enumerate(tokenized_text.ids)
                         if len(ids) > 0]
        if len(valid_indices) < len(all_chunks):
            all_chunks = [all_chunks[i] for i in valid_indices]
            chunks_text = [chunk[3] for chunk in all_chunks]
            tokenized_text = bm25s.tokenize(
                chunks_text,
                stopwords=None,
                token_pattern=r"(?u)\b\w+\b",
                show_progress=False,
                                           )
    elif len(valid_indices) < len(all_chunks):
        all_chunks = [all_chunks[i] for i in valid_indices]
        chunks_text = [chunk[3] for chunk in all_chunks]
        tokenized_text = bm25s.tokenize(
            chunks_text, stopwords="en", show_progress=False
                                       )

    print(f"  {c.Fore.CYAN}Building BM25 index...{c.Style.RESET_ALL}")
    retriever = bm25s.BM25()
    retriever.index(tokenized_text, show_progress=False)

    retriever.save(BM25_PATH)
    print("\n" + "  Saved BM25 index  → "
          f"{c.Fore.CYAN}{BM25_PATH}{c.Style.RESET_ALL}")

    chunk_metadata = [
        (fp, start, end) for (fp, start, end, _) in all_chunks
                     ]
    try:
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunk_metadata, f)
        print(f"  Saved chunk meta  → {c.Fore.CYAN}{CHUNKS_PATH}"
              f"{c.Style.RESET_ALL}\n")
    except (pickle.PicklingError, OSError) as e:
        print(f"[ERROR] Could not save chunk metadata: {e}")
        exit(1)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(c.Fore.CYAN + "  Indexing complete!".center(79) + c.Style.RESET_ALL,
          "\n" + f"{execution_time: .2f}s".center(79))
