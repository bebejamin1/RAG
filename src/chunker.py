#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   chunker.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/22 14:51:11 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/02 11:40:04 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import ast

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple

# (file_path, first_char_index, last_char_index, text)
Chunk = Tuple[str, int, int, str]

INDEXABLE_EXTENSIONS = {'.py', '.md', '.rst', '.txt'}


# *****************************************************************************
# *                          IS INDEXABLE                                     *
# *                                                                           *

def is_indexable(file_path: str) -> bool:
    """Check whether a file's extension is one this project chunks.

    Args:
        file_path: Path to the candidate file.

    Returns:
        True if the file extension is in INDEXABLE_EXTENSIONS.
    """

    if ('.' not in file_path.rsplit('/', 1)[-1]):
        return False

    ext = '.' + file_path.rsplit('.', 1)[-1].lower()

    return (ext in INDEXABLE_EXTENSIONS)


# *****************************************************************************
# *                          LINE OFFSETS                                     *
# *                                                                           *

def line_offsets(content: str) -> List[int]:
    """Return the character offset of the start of each line.

    offsets[i] is the index of the first character of line i + 1,
    and offsets[-1] is len(content).
    """

    offsets = [0]

    for line in content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    return (offsets)


# *****************************************************************************
# *                           CHUNK PYTHON                                    *
# *                    AST-based structural chunking                          *

def chunk_python(
    file_path: str, content: str, max_size: int
                ) -> List[Chunk]:
    """Chunk Python source code along its AST structure.

    Each top-level statement (function, class, imports, ...) becomes a
    block, decorators included. Consecutive small blocks are merged up
    to max_size so a chunk keeps whole definitions together. Blocks
    bigger than max_size are re-split with the text splitter. Files
    that do not parse fall back to plain text chunking.
    """

    if (not content):
        return []

    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return (chunk_text(file_path, content, max_size))

    if (not tree.body):
        return (chunk_text(file_path, content, max_size))

    offsets = line_offsets(content)

    blocks: List[Tuple[int, int]] = []
    for node in tree.body:
        decorators = getattr(node, "decorator_list", [])
        first_line = decorators[0].lineno if decorators else node.lineno
        end_line = node.end_lineno or node.lineno
        blocks.append((offsets[first_line - 1], offsets[end_line]))

    merged: List[Tuple[int, int]] = []
    for start, end in blocks:
        if (merged and end - merged[-1][0] <= max_size):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    chunks: List[Chunk] = []
    for start, end in merged:
        text = content[start:end]

        if (len(text) <= max_size):
            chunks.append((file_path, start, end, text))
            continue

        for _, sub_start, sub_end, sub_text in chunk_text(
                file_path, text, max_size):
            chunks.append(
                (file_path, start + sub_start, start + sub_end, sub_text))

    return (chunks)


# *****************************************************************************
# *                            CHUNK TEXT                                     *
# *                                                                           *

def chunk_text(
    file_path: str, content: str, max_size: int
              ) -> List[Chunk]:
    """Chunk plain text or Markdown content with a recursive splitter.

    Args:
        file_path: Path recorded alongside each chunk.
        content: Raw text content to split.
        max_size: Maximum number of characters per chunk.

    Returns:
        A list of (file_path, first_character_index, last_character_index,
        text) chunks covering the content.
    """

    if (not content):
        return []

    char_split = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=200,
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
    """Dispatch a file to the chunking strategy matching its type.

    Args:
        file_path: Path to the file being chunked.
        content: Raw file content.
        max_size: Maximum number of characters per chunk.

    Returns:
        Python-aware chunks for .py files, plain-text chunks otherwise.
    """

    if file_path.endswith('.py'):
        return (chunk_python(file_path, content, max_size))

    return (chunk_text(file_path, content, max_size))
