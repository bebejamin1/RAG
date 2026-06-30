#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 13:18:12 by bbeaurai            #+#    #+#            #
#   Updated: 2026/06/30 14:55:05 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import colorama as c
import os
import fire
import time

from student.src.indexing import _PROJECT_ROOT, index_main, search
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

        try:

            os.system("clear")

            if (max_chunk_size > 2000 or max_chunk_size <= 1):
                raise ValueError("The block size must be less than 2000 "
                                 "and greater than 2")

            print("\n" + c.Fore.CYAN + "".center(79, "="))
            print(" INDEXING ".center(79, "="))
            print("".center(79, "=") + self.ra + "\n\n")

            index_main(repo_path, max_chunk_size)

        except (Exception, ValueError) as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

# ============================ SEARCH =========================================

    def search(self, query: str, k: int = 5) -> None:
        """Search the index and print top-k results as JSON.

        Args:
            query: The search query.
            k: Number of results to return (default 5).
        """

        if (k <= 0):
            print("k cannot be less than or equal to 0")
            exit()

        if not query or not query.strip():
            print("\n" + f"{self.r}[ERROR]{self.rs}:"
                  " the request cannot be empty")
            exit()

        start_time = time.perf_counter()

        os.system("clear")
        print("\n" + c.Fore.YELLOW + "".center(79, "="))
        print(" SEARCH ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        try:
            raw = search(query, k)
        except FileNotFoundError as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

        sources = [
            MinimalSource(
                file_path=fp,
                first_character_index=start,
                last_character_index=end,
            )
            for fp, start, end in raw
        ]

        result = MinimalSearchResults(
            question_id="single-query",
            question=query,
            retrieved_sources=sources)

        data = result.model_dump()

        print(c.Fore.YELLOW + "Question ID : " + self.ra +
              data["question_id"])
        print(c.Fore.YELLOW + "Question    : " + self.ra + data["question"])
        print(c.Fore.YELLOW + "Sources     : " + self.ra +
              f"{len(data['retrieved_sources'])} result(s)\n")

        for i, src in enumerate(data["retrieved_sources"], 1):
            print(c.Fore.YELLOW + f"[{i}]" + self.ra,
                  src["file_path"].strip())
            print(c.Fore.YELLOW + "chars" + self.ra,
                  f"{src['first_character_index']} → "
                  f"{src['last_character_index']}\n")

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(c.Fore.YELLOW + " Search complete! ".center(79) + self.ra)
        print(f"{execution_time: .2f}s".center(79) + "\n")

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

        if (k <= 0):
            print("k cannot be less than or equal to 0")
            exit()

        try:
            with open(dataset_path, "r") as f:
                raw = json.load(f)
            dataset = RagDataset.model_validate(raw)
        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: Could not load dataset: {e}")
            exit()

        all_results = []
        for item in dataset.rag_questions:
            q = UnansweredQuestion.model_validate(item.model_dump())
            results = search(q.question, k)

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
