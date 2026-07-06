#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   indexing.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 11:28:05 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/02 14:03:05 by bbeaurai           ###   ########.fr      #
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

from src.chunker import chunker, is_indexable

IGNORED_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv",
                "build", "dist", ".mypy_cache", ".pytest_cache"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_DIR = f"{_PROJECT_ROOT}/data/processed"
BM25_PATH = f"{_PROJECT_ROOT}/data/processed/bm25_index"
CHUNKS_PATH = f"{_PROJECT_ROOT}/data/processed/chunks/chunks.pkl"


# *****************************************************************************
# *                            LOAD INDEX                                     *
# *                                                                           *

def load_index() -> Tuple[bm25s.BM25, List[Tuple[str, int, int]]]:

    try:
        if (not os.path.exists(BM25_PATH)):
            raise FileNotFoundError(
                "BM25 index not found. Run:\n"
                "  uv run python -m src index"
                                   )

        if (not os.path.exists(CHUNKS_PATH)):
            raise FileNotFoundError(
                "Chunk metadata not found. Run:\n"
                "  uv run python -m src index"
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
# *                            INDEX MAIN                                     *
# *                                                                           *

def index_main(path_dir: str, max_chunk_size: int) -> None:

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
                                   colour="CYAN"):
        chunks = chunker(file_path, content, max_chunk_size)
        all_chunks.extend(chunks)

    print(f"\n  Created {c.Fore.CYAN}{len(all_chunks)}{c.Style.RESET_ALL}"
          " total chunks.\n")

    print(f"  {c.Fore.CYAN}Tokenising corpus...{c.Style.RESET_ALL}")
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

    print(c.Fore.CYAN + " Indexing complete! ".center(79) + c.Style.RESET_ALL,
          "\n" + f"{execution_time: .2f}s".center(79))
