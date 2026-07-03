#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   llm.py                                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/07/01 09:14:14 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/03 11:54:25 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import re

from student.src.pydantic import MinimalSource
from transformers import pipeline, GenerationConfig
from typing import List, Optional, Any

_cache_pipeline: Optional[Any] = None


# *****************************************************************************
# *                              LOAD LLM                                     *
# *                                                                           *

def load_llm() -> None:
    global _cache_pipeline

    if _cache_pipeline is not None:
        return

    try:
        _cache_pipeline = pipeline("text-generation", model="Qwen/Qwen3-0.6B",
                                   clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        exit()


# *****************************************************************************
# *                           READ SOURCES                                    *
# *                                                                           *

def read_source(src: MinimalSource) -> str:

    try:
        with open(src[0], "r", encoding="utf-8",
                  errors="ignore") as f:
            content = f.read()
            return content[src[1]:src[2]]

    except TypeError:
        try:
            with open(src.file_path, "r", encoding="utf-8",
                      errors="ignore") as f:
                content = f.read()
            return content[
                src.first_character_index:src.last_character_index]

        except Exception:
            print("[ERROR]")
            exit()


# *****************************************************************************
# *                           MAKE MESSAGE                                    *
# *                                                                           *

def make_message(query: str, sources: List[MinimalSource]) -> str:
    context = ""
    limit = 2000 * len(sources)

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

def gen_answer(query: str, sources: List[MinimalSource]) -> str:

    message = make_message(query, sources)

    gen_conf = GenerationConfig(max_new_tokens=512, do_sample=False,
                                temperature=1.0, top_p=1.0, top_k=50)

    result = _cache_pipeline(  # type: ignore[misc]
        message, generation_config=gen_conf)

    try:
        raw = result[0]["generated_text"][-1]["content"]
    except Exception:
        print("\n", result, "\n")
        raw = result

    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if not (answer):
        raise ValueError("No answer could be generated from the provided"
                         " context.")

    return (answer)
