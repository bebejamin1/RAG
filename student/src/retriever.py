#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   retriever.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/07/02 13:53:01 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/02 14:28:14 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import re
import bm25s

from typing import Union, List, Tuple, Dict

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


# *****************************************************************************
# *                              SEARCH                                       *
# *                                                                           *

def search(query: Union[str, List[str]], retriever: bm25s.BM25,
           chunk_metadata: List[Tuple[str, int, int]], k: int
           ) -> List[Tuple[str, int, int]]:

    if (not query or k <= 0 or not chunk_metadata):
        return []

    tokenized_query = bm25s.tokenize(
        [query], stopwords="en", show_progress=False
                                    )

    n = min(max(k * 10, 50), len(chunk_metadata))
    results, scores = retriever.retrieve(tokenized_query, k=n)

    indices = results[0].tolist()
    chunk_scores = scores[0].tolist()

    best_per_file: Dict[str, Tuple[int, int, float]] = {}
    for idx, score in zip(indices, chunk_scores):
        file_path, start, end = chunk_metadata[idx]
        current = best_per_file.get(file_path)
        if current is None or score > current[2]:
            best_per_file[file_path] = (start, end, score)

    ranked = sorted(
        best_per_file.items(), key=lambda item: item[1][2], reverse=True
                    )[:k]

    return [
        (file_path, start, end)
        for file_path, (start, end, _score) in ranked
           ]
