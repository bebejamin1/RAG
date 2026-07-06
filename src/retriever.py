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

import bm25s

from typing import Union, List, Tuple, Dict


# *****************************************************************************
# *                              SEARCH                                       *
# *                                                                           *

def search(query: Union[str, List[str]], retriever: bm25s.BM25,
           chunk_metadata: List[Tuple[str, int, int]], k: int,
           max_per_file: int = 3
           ) -> List[Tuple[str, int, int]]:
    """Return the top-k chunks for a query, ranked by BM25 score.

    Candidates are walked in descending score order and at most
    max_per_file chunks are kept per file, so results stay diverse
    while questions whose sources live in a single file can still
    match several excerpts of it.
    """

    if (not query or k <= 0 or not chunk_metadata):
        return []

    tokenized_query = bm25s.tokenize(
        [query], stopwords="en", show_progress=False
                                    )

    n = min(max(k * 10, 50), len(chunk_metadata))
    results, _scores = retriever.retrieve(tokenized_query, k=n)

    indices = results[0].tolist()

    picked: List[Tuple[str, int, int]] = []
    per_file: Dict[str, int] = {}
    for idx in indices:
        file_path, start, end = chunk_metadata[idx]

        count = per_file.get(file_path, 0)
        if (count >= max_per_file):
            continue

        per_file[file_path] = count + 1
        picked.append((file_path, start, end))

        if (len(picked) >= k):
            break

    return (picked)
