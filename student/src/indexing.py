#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   indexing.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 11:28:05 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/26 16:31:14 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import os
import bm25s
import pickle
import colorama as c

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List

IGNORED_DIRS = {"__pycache__", "node_modules"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INDEX_DIR = f"{_PROJECT_ROOT}/data/processed"
BM25_PATH = f"{_PROJECT_ROOT}/data/processed/bm25_index"
CHUNKS_PATH = f"{_PROJECT_ROOT}/data/processed/chunks/chunks.pkl"

# BM25 = Best match 25

# prend en compte plusieur elements
# la frequence des termes de la requete apparaissent dans le codument
# la lontueur du doc

# chunking conseiller par sections


# *****************************************************************************
# *                            LOAD INDEX                                     *
# *                                                                           *

def load_index() -> Tuple[bm25s.BM25, List[Tuple[str, int, int]]]:

    try:

        if not os.path.exists(BM25_PATH):
            raise FileNotFoundError("BM25 not found" + "\n"
                                    "uv run python -m student index")

        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError("Chunk metadata not found" + "\n"
                                    "uv run python -m student index")

        retriever = bm25s.BM25.load(BM25_PATH, load_corpus=False)

        with open(CHUNKS_PATH, "rb") as f:
            chunk_metadata = pickle.load(f)

    except FileNotFoundError as e:
        print(e)
        exit()

    return retriever, chunk_metadata


# *****************************************************************************
# *                            LOAD FILES                                     *
# *                                                                           *

def load_files(path_dir: str) -> List[Tuple[str, str]]:
    files = []

    for path in Path(path_dir).rglob("*"):

        if not path.is_file():
            continue

        if any(part.startswith(".") or part in IGNORED_DIRS
               for part in path.parts):
            continue

        try:

            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()

            try:
                relatif_path = f"{path.resolve().relative_to(_PROJECT_ROOT)}"
            except ValueError:
                relatif_path = path

            files.append((relatif_path, content))

        except ValueError as e:
            print(e)

    return (files)


# *****************************************************************************
# *                            INDEX MAIN                                     *
# *                                                                           *

def index_main(path_dir: str, max_chunk_size: int) -> None:

    try:

        os.makedirs(BM25_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

        print(f"🥸​ {c.Fore.LIGHTBLUE_EX}Reading all the files" + "\n")
        files = load_files(path_dir)
        for _ in tqdm(range(len(files) * 10000), desc="reading all files"):
            continue
        print("\n" + f"🔍​ {len(files)} files found" + "\n")

    except (PermissionError, IndexError) as e:
        print(e)
        exit()
