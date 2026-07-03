#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 13:18:12 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/03 14:13:22 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import colorama as c
import json
import os
import fire
import time
from tqdm import tqdm

from student.src.indexing import _PROJECT_ROOT, index_main, load_index
from student.src.retriever import search
from student.src.llm import gen_answer, load_llm
from student.src.pydantic import (
    MinimalSource,
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
    RagDataset,
    StudentSearchResultsAndAnswer
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

        if (k <= 0):
            print("k cannot be less than or equal to 0")
            exit()

        if not query or not query.strip():
            print("\n" + f"{self.r}[ERROR]{self.rs}:"
                  " the request cannot be empty")
            exit()

        try:
            retriever, chunk_metadata = load_index()
        except PermissionError as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

        start_time = time.perf_counter()

        os.system("clear")
        print("\n" + c.Fore.YELLOW + "".center(79, "="))
        print(" SEARCH ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        try:
            raw = search(query, retriever, chunk_metadata, k)
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
            question_str=query,
            retrieved_sources=sources)

        data = result.model_dump()

        print(c.Fore.YELLOW + "Question ID : " + self.ra +
              data["question_id"])
        print(c.Fore.YELLOW + "Question    : " + self.ra +
              data["question_str"])
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

# =============================== ANSWER ======================================

    def answer(self, query: str, k: int = 5) -> None:

        if (not query or not query.strip()):
            print("\n" + f"{self.r}[ERROR]{self.rs}:"
                  "the request cannot be empty")
            exit()

        try:
            retriever, chunk_metadata = load_index()
        except PermissionError as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

        start_time = time.perf_counter()

        print("\n" + c.Fore.MAGENTA + "".center(79, "="))
        print(" ANSWER ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        sources: list[MinimalSource] = search(  # type: ignore[assignment]
            query, retriever, chunk_metadata, k=k)

        print("Loading model: " + c.Fore.MAGENTA + "Qwen/Qwen3-0.6B" + "\n" +
              c.Fore.RESET)
        load_llm()

        answer_text = gen_answer(query, sources)

        minimal_sources = [
            MinimalSource(
                file_path=fp,
                first_character_index=start,
                last_character_index=end,
            )
            for fp, start, end in sources
        ]

        _ = MinimalAnswer(question_id="single-query",
                          question_str=query,
                          retrieved_sources=minimal_sources,
                          answer=answer_text)

        print("\n" + c.Fore.MAGENTA + "Sources: " + c.Fore.RESET)
        for i, s in enumerate(sources, 1):
            print(c.Fore.MAGENTA + f"[{i}] " + c.Fore.RESET +
                  f"{s[0]}" + "\n" +
                  c.Fore.MAGENTA + "chars " + c.Fore.RESET +
                  f"{s[1]} → {s[2]}" + "\n")

        print("\n" + c.Fore.MAGENTA + "Question: " + c.Fore.RESET + query)
        print("\n" + c.Fore.MAGENTA + "Answer:"
              + c.Fore.RESET + "\n" + answer_text)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(c.Fore.MAGENTA + " Search complete! ".center(79) + self.ra)
        print(f"{execution_time: .2f}s".center(79) + "\n")

# ============================ SEARCH_DATASET =================================

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "",
                      ) -> None:

        os.system("clear")

        start_time = time.perf_counter()

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

        try:
            retriever, chunk_metadata = load_index()
        except PermissionError as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

        print("\n" + c.Fore.BLUE + "".center(79, "="))
        print(" SEARCH DATASET ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        all_results = []
        for item in tqdm(dataset.rag_questions, "Search", colour="BLUE"):
            q = UnansweredQuestion.model_validate(item.model_dump())
            results: list[MinimalSource] = search(  # type: ignore[assignment]
                q.question, retriever, chunk_metadata, k=k)

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
                    question_str=q.question,
                    retrieved_sources=sources,
                                    )
                              )

        output = StudentSearchResults(search_results=all_results, k=k)
        out_json = json.dumps(output.model_dump(), indent=2)

        output_path = ""
        if (save_directory):
            os.makedirs(save_directory, exist_ok=True)
            output_path = os.path.join(
                save_directory, os.path.basename(dataset_path))
            with open(output_path, "w") as f:
                f.write(out_json)

        if output_path:
            print("\n\n" + "Saved student_search_results to:")
            print(c.Fore.BLUE + output_path + self.ra + "\n")

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print("\n" + c.Fore.BLUE + " Search dataset complete! ".center(79) +
              self.ra)
        print(f"{execution_time: .2f}s".center(79) + "\n")

# ============================ SEARCH_DATASET =================================

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
                      ) -> None:

        try:
            with open(student_search_results_path, "r") as f:
                raw = json.load(f)
            dataset = StudentSearchResults.model_validate(raw)
        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: Could not load dataset: {e}")
            exit()

        try:
            retriever, chunk_metadata = load_index()
        except PermissionError as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

        load_llm()
        os.system("clear")

        start_time = time.perf_counter()

        print("\n" + c.Fore.GREEN + "".center(79, "="))
        print(" ANSWER DATASET ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        print("Loading model: " + c.Fore.GREEN + "Qwen/Qwen3-0.6B" + "\n" +
              c.Fore.RESET)

        all_answers = []

        for data in tqdm(dataset.search_results,
                         "Generating Responses", colour="GREEN"):

            sources: list[MinimalSource] = search(  # type: ignore[assignment]
                    data.question_str, retriever,
                    chunk_metadata, k=dataset.k)

            a = gen_answer(data.question_str, sources)

            minimal_sources = [
                MinimalSource(
                    file_path=fp,
                    first_character_index=start,
                    last_character_index=end,
                )
                for fp, start, end in sources
            ]

            answer = MinimalAnswer(question_id=data.question_id,
                                   question_str=data.question_str,
                                   retrieved_sources=minimal_sources,
                                   answer=a)
            all_answers.append(answer)

        output = StudentSearchResultsAndAnswer(search_results=all_answers,
                                               k=dataset.k)
        # faire le search dataset
        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(student_search_results_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output.model_dump_json(indent=2))

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print("\n" + c.Fore.GREEN + " Answer dataset complete! ".center(79) +
              self.ra)
        print(f"{execution_time: .2f}s".center(79) + "\n")


# *****************************************************************************
# *                                MAIN                                       *
# *                                                                           *

def main():
    try:
        fire.Fire(RagSystem)
    except KeyboardInterrupt:
        print("pk tu arrete ?")


if __name__ == "__main__":
    main()
