#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   llm.py                                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/07/01 09:14:14 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/01 16:06:26 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import colorama as c
import re
import time

from student.src.pydantic import MinimalSource
from transformers import pipeline, GenerationConfig
from typing import List, Optional, Any

_cache_pipeline: Optional[Any] = None


def load_llm() -> None:
    global _cache_pipeline

    if _cache_pipeline is not None:
        return

    print("Loading model: " + c.Fore.MAGENTA + "Qwen/Qwen3-0.6B" + "\n" +
          c.Fore.RESET)

    try:
        _cache_pipeline = pipeline("text-generation", model="Qwen/Qwen3-0.6B",
                                   clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        exit()


def read_source(src: MinimalSource) -> str:
    try:
        with open(src[0], "r", encoding="utf-8",
                  errors="ignore") as f:
            content = f.read()
            return content[src[1]:src[2]]
    except ValueError:
        return ""


def make_message(query: str, sources: List[MinimalSource]) -> str:
    context = ""
    limit = 3000

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
                "You are a helpful assistant. "
                "Answer the question using ONLY the provided context. "
                "Be concise and mention the source file. /no_think"
                       ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}" + "\n\n" + "Question: {query}",
        },
           ]


def gen_answer(query: str, sources: List[MinimalSource]) -> str:

    start_time = time.perf_counter()

    load_llm()

    message = make_message(query, sources)

    gen_conf = GenerationConfig(max_new_tokens=256, do_sample=False,
                                temperature=1.0, top_p=1.0, top_k=50)
    result = _cache_pipeline(  # type: ignore[misc]
        message, generation_config=gen_conf)

    raw = result[0]["generated_text"][-1]["content"]
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if not (answer):
        raise ValueError("No answer could be generated from the provided"
                         " context.")

    print("\n" + c.Fore.MAGENTA + "Sources: " + c.Fore.RESET)

    for i, s in enumerate(sources, 1):
        print(c.Fore.MAGENTA + f"[{i}] " + c.Fore.RESET +
              f"{s[0]}" + "\n" +
              c.Fore.MAGENTA + "chars " + c.Fore.RESET +
              f"{s[1]} → {s[2]}" + "\n")

    print("\n" + c.Fore.MAGENTA + "Question: " + c.Fore.RESET + query)
    print("\n" + c.Fore.MAGENTA + "Answer:" + c.Fore.RESET + "\n" + answer)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print("\n" +
          c.Fore.MAGENTA + " Indexing complete! ".center(79) + c.Fore.RESET +
          "\n" + f"{execution_time: .2f}s".center(79))

    return (answer)
