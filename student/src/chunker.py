#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   chunker.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/22 14:51:11 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/29 00:00:00 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import ast
import re
from typing import List, Tuple

# (file_path, first_char_index, last_char_index, text)
Chunk = Tuple[str, int, int, str]

INDEXABLE_EXTENSIONS = {'.py', '.md', '.rst', '.txt'}


def _get_line_offsets(content: str) -> List[int]:
    """Return the character offset of the start of each line.

    Args:
        content: Full file content as a string.

    Returns:
        List where index i is the char offset of line i+1 (1-indexed lines).
    """
    offsets: List[int] = [0]
    for line in content.split('\n')[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _split_by_size(
    file_path: str,
    content: str,
    start: int,
    end: int,
    max_size: int,
) -> List[Chunk]:
    """Split content[start:end] into chunks of at most max_size characters.

    Args:
        file_path: Source file path.
        content: Full file content.
        start: Start character index (inclusive).
        end: End character index (exclusive).
        max_size: Maximum characters per chunk.

    Returns:
        List of Chunk tuples.
    """
    chunks: List[Chunk] = []
    i = start
    while i < end:
        chunk_end = min(i + max_size, end)
        text = content[i:chunk_end]
        if text.strip():
            chunks.append((file_path, i, chunk_end, text))
        i = chunk_end
    return chunks


def chunk_python(
    file_path: str, content: str, max_size: int
) -> List[Chunk]:
    """Chunk Python source using AST: one chunk per top-level definition.

    Falls back to chunk_text on SyntaxError.

    Args:
        file_path: Path to the Python source file.
        content: Full file content.
        max_size: Maximum characters per chunk.

    Returns:
        List of Chunk tuples.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_text(file_path, content, max_size)

    offsets = _get_line_offsets(content)

    def char_range(node: ast.AST) -> Tuple[int, int]:
        """Convert AST node line positions to character indices."""
        # Include decorators if present
        if hasattr(node, 'decorator_list') and node.decorator_list:  # type: ignore[union-attr]
            first_line = node.decorator_list[0].lineno  # type: ignore[union-attr]
        else:
            first_line = node.lineno  # type: ignore[union-attr]
        start = offsets[first_line - 1]
        end_line = node.end_lineno  # type: ignore[union-attr]
        end_col = node.end_col_offset  # type: ignore[union-attr]
        end = offsets[end_line - 1] + end_col
        return start, end

    top_nodes = [
        n for n in tree.body
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
    ]

    intervals = sorted(
        [char_range(n) for n in top_nodes], key=lambda x: x[0]
    )

    chunks: List[Chunk] = []
    prev = 0

    for start, end in intervals:
        # Gap before this node (imports, module-level code)
        if start > prev:
            gap = content[prev:start]
            if gap.strip():
                chunks.extend(
                    _split_by_size(file_path, content, prev, start, max_size)
                )
        # The node itself
        if end - start <= max_size:
            text = content[start:end]
            if text.strip():
                chunks.append((file_path, start, end, text))
        else:
            chunks.extend(
                _split_by_size(file_path, content, start, end, max_size)
            )
        prev = end

    # Trailing content after the last top-level node
    if prev < len(content):
        trailing = content[prev:]
        if trailing.strip():
            chunks.extend(
                _split_by_size(
                    file_path, content, prev, len(content), max_size
                )
            )

    return chunks if chunks else chunk_text(file_path, content, max_size)


def chunk_text(
    file_path: str, content: str, max_size: int
) -> List[Chunk]:
    """Chunk text/Markdown by headers or paragraph breaks, then by size.

    For .md/.rst files, splits at Markdown headers (# ...).
    For other text files, splits at double newlines (paragraphs).
    Adjacent small sections are merged up to max_size.

    Args:
        file_path: Path to the text file.
        content: Full file content.
        max_size: Maximum characters per chunk.

    Returns:
        List of Chunk tuples.
    """
    if not content.strip():
        return []

    if file_path.endswith('.md') or file_path.endswith('.rst'):
        # Split at the START of each Markdown header line
        pattern = re.compile(r'(?=^#{1,6}\s)', re.MULTILINE)
    else:
        # Split at double (or more) newlines
        pattern = re.compile(r'\n{2,}')

    boundaries = (
        [0]
        + [m.start() for m in pattern.finditer(content)]
        + [len(content)]
    )
    sections = list(zip(boundaries[:-1], boundaries[1:]))

    chunks: List[Chunk] = []
    buf_start: int = 0
    buf: str = ""

    for sec_start, sec_end in sections:
        seg = content[sec_start:sec_end]
        if not seg.strip():
            continue

        if len(buf) + len(seg) <= max_size:
            if not buf:
                buf_start = sec_start
            buf += seg
        else:
            if buf:
                chunks.append(
                    (file_path, buf_start, buf_start + len(buf), buf)
                )
            if len(seg) <= max_size:
                buf_start = sec_start
                buf = seg
            else:
                chunks.extend(
                    _split_by_size(
                        file_path, content, sec_start, sec_end, max_size
                    )
                )
                buf = ""
                buf_start = sec_end

    if buf:
        chunks.append((file_path, buf_start, buf_start + len(buf), buf))

    return chunks


def is_indexable(file_path: str) -> bool:
    """Return True if the file extension is worth indexing.

    Args:
        file_path: Path or filename to check.

    Returns:
        True if the file should be indexed.
    """
    if '.' not in file_path.rsplit('/', 1)[-1]:
        return False
    ext = '.' + file_path.rsplit('.', 1)[-1].lower()
    return ext in INDEXABLE_EXTENSIONS


def chunker(
    file_path: str, content: str, max_size: int
) -> List[Chunk]:
    """Dispatch to the appropriate chunking strategy based on file type.

    Args:
        file_path: Path to the source file (used to detect file type).
        content: Full text content of the file.
        max_size: Maximum number of characters per chunk.

    Returns:
        List of (file_path, first_char_idx, last_char_idx, text) tuples.
    """
    if file_path.endswith('.py'):
        return chunk_python(file_path, content, max_size)
    return chunk_text(file_path, content, max_size)
