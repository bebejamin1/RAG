#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   evaluation.py                                        :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/07/06 00:00:00 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/06 00:00:00 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import Dict, List, Tuple

from src.pydantic import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
                                 )

# A correct source counts as "found" if a retrieved source overlaps
# at least 5% of it (subject VI.1.1).
MIN_OVERLAP = 0.05


# *****************************************************************************
# *                          OVERLAP RATIO                                    *
# *                                                                           *

def overlap_ratio(retrieved: MinimalSource, correct: MinimalSource) -> float:

    if (retrieved.file_path != correct.file_path):
        return 0.0

    start = max(retrieved.first_character_index,
                correct.first_character_index)
    end = min(retrieved.last_character_index,
              correct.last_character_index)

    correct_len = (correct.last_character_index
                   - correct.first_character_index)

    if (end <= start or correct_len <= 0):
        return 0.0

    return ((end - start) / correct_len)


# *****************************************************************************
# *                         QUESTION RECALL                                   *
# *              score = number_found / number_of_correct_sources             *

def question_recall(retrieved: List[MinimalSource],
                    correct: List[MinimalSource]) -> float:

    if (not correct):
        return 0.0

    found = sum(
        1 for c in correct
        if any(overlap_ratio(r, c) >= MIN_OVERLAP for r in retrieved)
               )

    return (found / len(correct))


# *****************************************************************************
# *                          GROUND TRUTH                                     *
# *                                                                           *

def ground_truth(dataset: RagDataset) -> Dict[str, List[MinimalSource]]:

    return {
        q.question_id: q.sources
        for q in dataset.rag_questions
        if isinstance(q, AnsweredQuestion)
           }


# *****************************************************************************
# *                           RECALL AT K                                     *
# *                                                                           *

def recall_at_k(results: StudentSearchResults,
                truth: Dict[str, List[MinimalSource]],
                k: int) -> Tuple[float, int]:

    scores = [
        question_recall(res.retrieved_sources[:k], truth[res.question_id])
        for res in results.search_results
        if res.question_id in truth
             ]

    if (not scores):
        return (0.0, 0)

    return (sum(scores) / len(scores), len(scores))
