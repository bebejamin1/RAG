#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   indexing.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 11:28:05 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/27 21:14:53 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import os
import bm25s
import pickle
import colorama as c
import numpy as np

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

        except Exception as e:
            print(f"Unreadable {path_dir}: {e}")
            exit()

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
        for file in tqdm(load_files(path_dir), desc="reading all files"):

            continue
        print("\n" + f"🔍​ {len(files)} files found" + "\n")

    except (PermissionError, IndexError) as e:
        print(e)
        exit()

    all_chunks: list = []  # noqa









    # all_chunks = []
    # for file_path, content in tqdm(files, desc="Chunking"):
    #     chunks = chunk_choice(file_path, content, max_chunk_size)
    #     all_chunks.extend(chunks)

    # print(f"     {len(all_chunks)}: Total number of chunks created")

    # # BM25 tokenise les textes
    # # on va chercher le 4eme élément du tuple qui correspond au contenu texte
    # # stopwords="en" -> ignore mots fréquents qui servent à r (the, a, is...)
    # chunks_text = [chunk[3] for chunk in all_chunks]
    # print("\n🖥️​ Corpus tokenization...")
    # tokenized_text = bm25s.tokenize(chunks_text, stopwords="en",
    #                                 show_progress=False)

    # # Construction index BM25 sur texte tokenisé et nettoyé
    # print("\n📚 Construction of the BM25 Index...")
    # retriever = bm25s.BM25()
    # retriever.index(tokenized_text, show_progress=False)

    # # sauvegarde de l'index BM25 sur disque (pour pouvoir le recharger sans
    # # recalculer)
    # retriever.save(BM25_PATH)
    # print(f"     💾 BM25 index saved in {BM25_PATH}")

    # # on sauvegarde les métadonnées des chunks séparément (file_path +
    # # position). Pas besoin de stocker
    # chunk_metadata = [(fp, start, end) for (fp, start, end, _) in all_chunks]
    # try:
    #     with open(CHUNKS_PATH, "wb") as f:
    #         pickle.dump(chunk_metadata, f)
    #     print(f"     💾 Metadata chunks saved in {CHUNKS_PATH}")
    # except pickle.PicklingError:
    #     print("Unable to serialize the chunk metadata.")
    #     sys.exit()
    # except FileNotFoundError:
    #     print(f"The destination folder for {CHUNKS_PATH} does "
    #           "not exist.")
    #     sys.exit()

    # print("\n" + "Ingestion complete! The indexes have been saved to "
    #       "the data/processed/directory".center(80, " ") + "\n")
