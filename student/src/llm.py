#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   llm.py                                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/07/01 09:14:14 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/01 09:17:34 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from transformers import pipeline, GenerationConfig


def load_llm() -> None:
    global _cache_pipeline

    # si déjà chargé = ok
    if _cache_pipeline is not None:
        return

    print("\n" + "🚀​ Loading model: Qwen/Qwen3-0.6B" + "\n")

    # pipeline fais 3 étapes = tokenizer -> modèle -> detokenizer
    try:
        _cache_pipeline = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        exit()

    print("\n" + "🌚 Model loaded !" "\n")


def gen_answer() -> None:
    GenerationConfig
