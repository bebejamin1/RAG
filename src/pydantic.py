#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   pydantic.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 09:07:23 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/19 09:21:01 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import List
from pydantic import BaseModel


# *****************************************************************************
# *                           MinimalSource                                   *
# *              represents a minimal source of information                   *

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


# *****************************************************************************
# *                        UnansweredQuestion                                 *
# *                 represent an unanswered question                          *

class UnansweredQuestion(BaseModel):
    question_id: str
    question: str


# *****************************************************************************
# *                         AnsweredQuestion                                  *
# *                   represent an answered question                          *

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


# *****************************************************************************
# *                            RagDataset                                     *
# *                represents a dataset of RAG questions                      *

class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


# *****************************************************************************
# *                        MinimalSearchResults                               *
# *                    represent the search results                           *

class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


# *****************************************************************************
# *                           MinimalAnswer                                   *
# *                    represent the search answer                            *

class MinimalAnswer(MinimalSearchResults):
    answer: str


# *****************************************************************************
# *                       StudentSearchResults                                *
# *                     represent search results                              *

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


# *****************************************************************************
# *                 StudentSearchResultsAndAnswer                             *
# *             represent search results with answers                         *

class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
