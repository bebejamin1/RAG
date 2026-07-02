#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   chunker.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/22 14:51:11 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/02 11:34:48 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple

# import ast
# import re  # REGEX regular expression

# (file_path, first_char_index, last_char_index, text)
Chunk = Tuple[str, int, int, str]

INDEXABLE_EXTENSIONS = {'.py', '.md', '.rst', '.txt'}


# *****************************************************************************
# *                          IS INDEXABLE                                     *
# *                                                                           *

def is_indexable(file_path: str) -> bool:

    if ('.' not in file_path.rsplit('/', 1)[-1]):
        return False

    ext = '.' + file_path.rsplit('.', 1)[-1].lower()

    return (ext in INDEXABLE_EXTENSIONS)


# *****************************************************************************
# *                           CHUNK PYTHON                                    *
# *                                                                           *

def chunk_python(
    file_path: str, content: str, max_size: int
                ) -> List[Chunk]:
    # TODO: AST-based chunking (functions/classes) -> fallback for now
    return (chunk_text(file_path, content, max_size))


# *****************************************************************************
# *                            CHUNK TEXT                                     *
# *                                                                           *

def chunk_text(
    file_path: str, content: str, max_size: int
              ) -> List[Chunk]:

    if (not content):
        return []

    overlap = min(200, max(0, max_size // 10))

    char_split = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        add_start_index=True)

    documents = char_split.create_documents([content])

    return [
        (file_path, doc.metadata["start_index"],
         doc.metadata["start_index"] + len(doc.page_content),
         doc.page_content)
        for doc in documents
           ]


# *****************************************************************************
# *                             CHUNKER                                       *
# *                                                                           *

def chunker(
    file_path: str, content: str, max_size: int
           ) -> List[Chunk]:

    if file_path.endswith('.py'):
        return (chunk_python(file_path, content, max_size))

    return (chunk_text(file_path, content, max_size))
