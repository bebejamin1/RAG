#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 13:18:12 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/29 00:00:00 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import colorama as c
import os
import fire

from student.src.indexing import _PROJECT_ROOT, index_main, search as _search
from student.src.pydantic import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
    RagDataset,
)


# *****************************************************************************
# *                               CLASS                                       *
# *                                                                           *

class RagSystem():

    def __init__(self):
        self.ra = c.Style.RESET_ALL
        self.rs = "\033[0m"
        self.r = "\033[31m\033[5m\033[1m"

# ============================ INDEX ==========================================

    def index(self,
              repo_path: str = f"{_PROJECT_ROOT}/data/raw/vllm-0.10.1",
              max_chunk_size: int = 2000) -> None:
        """Build the BM25 index from a source repository.

        Args:
            repo_path: Path to the repository to index.
            max_chunk_size: Maximum characters per chunk (default 2000).
        """
        os.system("clear")
        print("\n" + c.Fore.CYAN + "".center(79, "="))
        print(" INDEXING ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        try:
            index_main(repo_path, max_chunk_size)
        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit(1)

# ============================ SEARCH =========================================

    def search(self, query: str, k: int = 5) -> None:
        """Search the index and print top-k results as JSON.

        Args:
            query: The search query.
            k: Number of results to return (default 5).
        """
        results = _search(query, k)
        sources = [
            MinimalSource(
                file_path=fp,
                first_character_index=start,
                last_character_index=end,
            )
            for fp, start, end in results
        ]
        output = MinimalSearchResults(
            question_id="cli_query",
            question=query,
            retrieved_sources=sources,
        )
        print(json.dumps(output.model_dump(), indent=2))

# ============================ SEARCH_DATASET =================================

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        output_path: str = "",
    ) -> None:
        """Run search for every question in a dataset file and output JSON.

        Args:
            dataset_path: Path to a JSON file with RagDataset format.
            k: Number of results per question (default 5).
            output_path: Optional path to write output JSON (prints if empty).
        """
        try:
            with open(dataset_path, "r") as f:
                raw = json.load(f)
            dataset = RagDataset.model_validate(raw)
        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: Could not load dataset: {e}")
            exit(1)

        all_results = []
        for item in dataset.rag_questions:
            q = UnansweredQuestion.model_validate(item.model_dump())
            results = _search(q.question, k)
            sources = [
                MinimalSource(
                    file_path=fp,
                    first_character_index=start,
                    last_character_index=end,
                )
                for fp, start, end in results
            ]
            all_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=all_results, k=k)
        out_json = json.dumps(output.model_dump(), indent=2)

        if output_path:
            with open(output_path, "w") as f:
                f.write(out_json)
            print(f"Results written to {output_path}")
        else:
            print(out_json)


# *****************************************************************************
# *                                MAIN                                       *
# *                                                                           *

def main():
    fire.Fire(RagSystem)


if __name__ == "__main__":
    main()
