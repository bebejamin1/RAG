#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   pydantic.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 09:07:23 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/26 11:02:40 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import List
from pydantic import BaseModel


# *****************************************************************************
# *                           MinimalSource                                   *
# *              represents a minimal source of information                   *

class MinimalSource(BaseModel):
    """Represent a minimal source of information within a file.

    Attributes:
        file_path: Absolute or relative path to the source file.
        first_character_index: Index of the first character in the source
        excerpt.
        last_character_index: Index of the last character in the source
        excerpt.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


# *****************************************************************************
# *                        UnansweredQuestion                                 *
# *                 represent an unanswered question                          *

class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet been answered.

    Attributes:
        question_id: Unique identifier for the question.
        question: The raw question text.
    """

    question_id: str
    question: str


# *****************************************************************************
# *                         AnsweredQuestion                                  *
# *                   represent an answered question                          *

class AnsweredQuestion(UnansweredQuestion):
    """Represent a question paired with its answer and source references.

    Inherits:
        UnansweredQuestion: Provides question_id and question fields.

    Attributes:
        sources: List of minimal source excerpts supporting the answer.
        answer: The generated or curated answer text.
    """

    sources: List[MinimalSource]
    answer: str


# *****************************************************************************
# *                            RagDataset                                     *
# *                represents a dataset of RAG questions                      *

class RagDataset(BaseModel):
    """Represent a dataset of RAG evaluation questions.

    Attributes:
        rag_questions: List of questions, each either answered or unanswered.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


# *****************************************************************************
# *                        MinimalSearchResults                               *
# *                    represent the search results                           *

class MinimalSearchResults(BaseModel):
    """Represent the retrieval results for a single question.

    Attributes:
        question_id: Unique identifier of the queried question.
        question: The question text used for retrieval.
        retrieved_sources: Ordered list of source excerpts retrieved.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


# *****************************************************************************
# *                           MinimalAnswer                                   *
# *                    represent the search answer                            *

class MinimalAnswer(MinimalSearchResults):
    """Represent retrieval results augmented with a generated answer.

    Inherits:
        MinimalSearchResults: Provides question_id, question, and
            retrieved_sources fields.

    Attributes:
        answer: The answer generated from the retrieved sources.
    """

    answer: str


# *****************************************************************************
# *                       StudentSearchResults                                *
# *                     represent search results                              *

class StudentSearchResults(BaseModel):
    """Represent the full set of retrieval results produced by a student
    pipeline.

    Attributes:
        search_results: List of per-question retrieval results.
        k: Number of sources retrieved per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


# *****************************************************************************
# *                 StudentSearchResultsAndAnswer                             *
# *             represent search results with answers                         *

class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represent retrieval results with answers produced by a student pipeline.

    Inherits:
        StudentSearchResults: Provides search_results and k fields.

    Attributes:
        search_results: List of per-question retrieval results with answers.
    """

    search_results: List[MinimalAnswer]
