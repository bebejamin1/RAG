#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   llm.py                                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/07/01 09:14:14 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/03 15:17:28 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import re

from transformers import pipeline, GenerationConfig
from typing import Dict, List, Optional, Tuple, Any

_cache_pipeline: Optional[Any] = None

# (file_path, first_character_index, last_character_index)
Source = Tuple[str, int, int]

# Total context budget in characters, fixed regardless of k so a larger
# k (more sources) can't grow the prompt past what a small GPU's
# attention buffers can hold (memory grows faster than linearly with
# context length).
_MAX_CONTEXT_CHARS = 8000


# *****************************************************************************
# *                              LOAD LLM                                     *
# *                                                                           *

def load_llm() -> None:
    """Load and cache the Qwen/Qwen3-0.6B text-generation pipeline.

    Subsequent calls are no-ops once the pipeline is cached.
    """

    global _cache_pipeline

    if _cache_pipeline is not None:
        return

    try:
        _cache_pipeline = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
    except Exception:
        print("[ERROR] Failed to load model")
        exit()


# *****************************************************************************
# *                           READ SOURCES                                    *
# *                                                                           *

def read_source(src: Source) -> str:
    """Read the excerpt of a source file covered by a retrieval result.

    Args:
        src: A (file_path, first_character_index, last_character_index)
            tuple identifying the excerpt to read.

    Returns:
        The text between first_character_index and last_character_index,
        or an empty string if the file cannot be read.
    """

    file_path, first_character_index, last_character_index = src

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content[first_character_index:last_character_index]

    except OSError as e:
        print(f"[WARN] Could not read source {file_path}: {e}")
        return ""


# *****************************************************************************
# *                           MAKE MESSAGE                                    *
# *                                                                           *

def make_message(query: str,
                 sources: List[Source]) -> List[Dict[str, str]]:
    """Build the chat messages sent to the LLM for answer generation.

    Args:
        query: The user's question.
        sources: Retrieved sources to ground the answer in, read and
            concatenated into a bounded context window.

    Returns:
        A list of chat messages (system + user) ready for the pipeline.
    """

    context = ""
    limit = _MAX_CONTEXT_CHARS

    for src in sources:
        chunk_text = read_source(src)
        if not chunk_text:
            continue

        if (len(chunk_text) > limit):
            chunk_text = chunk_text[:limit]
            context += "Source: " + chunk_text.strip("\n") + " ..." + "\n"
            break

        context += "Source: " + chunk_text.strip("\n") + " ..." + "\n"
        limit -= len(chunk_text)

    return [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant answering "
                "questions about a codebase and its documentation. "
                "Use ONLY the information given in the context to "
                "answer the question directly, in 2 to 4 sentences. "
                "Cover the key point(s) without listing every detail "
                "from the context. "
                "Never use prior knowledge, and never invent facts, "
                "names, file paths, or details that are not explicitly "
                "present in the context. "
                "If the context does not answer the question, say so "
                "clearly instead of guessing. "
                "The answer must be self-contained and understandable "
                "without seeing the question or the context again. "
                "Do not mention sources, file names, or file paths. "
                "/no_think"
                       ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
           ]


# *****************************************************************************
# *                            GEN ANSWER                                     *
# *                                                                           *

def gen_answer(query: str, sources: List[Source]) -> str:
    """Generate a grounded natural-language answer from retrieved sources.

    Args:
        query: The user's question.
        sources: Retrieved sources to ground the answer in.

    Returns:
        The generated answer text, with any <think> reasoning stripped.

    Raises:
        ValueError: If the model produces an empty answer.
    """

    message = make_message(query, sources)

    gen_conf = GenerationConfig(max_new_tokens=256, do_sample=False)

    result = _cache_pipeline(  # type: ignore[misc]
        message, generation_config=gen_conf)

    try:
        raw = str(result[0]["generated_text"][-1]["content"])
    except (IndexError, KeyError, TypeError) as e:
        print(f"[WARN] Unexpected pipeline output: {e}\n{result}")
        raw = ""

    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if not (answer):
        raise ValueError("No answer could be generated from the provided"
                         " context.")

    return (answer)
