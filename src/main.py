#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   main.py                                              :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: bbeaurai <bbeaurai@student.42lehavre.fr>     +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/06/19 13:18:12 by bbeaurai            #+#    #+#            #
#   Updated: 2026/07/03 15:23:10 by bbeaurai           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import colorama as c
import json
import os
import fire
import time

from typing import List, Tuple
from tqdm import tqdm
from src.indexing import _PROJECT_ROOT, index_main, load_index
from src.retriever import search
from src.llm import gen_answer, load_llm
from src.evaluation import ground_truth, recall_at_k
from src.pydantic import (
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
    """Python Fire CLI exposing the RAG pipeline: index, search, answer,
    and evaluate.
    """

    def __init__(self) -> None:
        self.ra = c.Style.RESET_ALL
        self.rs = "\033[0m"
        self.r = "\033[31m\033[5m\033[1m"

# ============================ INDEX ==========================================

    def index(self,
              repo_path: str = f"{_PROJECT_ROOT}/data/raw/vllm-0.10.1",
              max_chunk_size: int = 2000) -> None:
        """Ingest a repository and persist a BM25 index under
        data/processed/.

        Args:
            repo_path: Root directory of the corpus to ingest.
            max_chunk_size: Maximum number of characters per chunk
                (must be between 2 and 2000).
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

        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

# ============================ SEARCH =========================================

    def search(self, query: str, k: int = 5) -> None:
        """Search the index for a single query and print the top-k sources.

        Args:
            query: The question to search for.
            k: Number of top results to retrieve.
        """

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
            question=query,
            question_str=query,
            retrieved_sources=sources)

        data = result.model_dump()

        print(c.Fore.YELLOW + "Question ID : " + self.ra +
              data["question_id"])
        print(c.Fore.YELLOW + "Question    : " + self.ra +
              data["question"])
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
        """Answer a single query using context retrieved from the index.

        Args:
            query: The question to answer.
            k: Number of top sources to retrieve as context.
        """

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

        sources: List[Tuple[str, int, int]] = search(
            query, retriever, chunk_metadata, k=k)

        print("Loading model: " + c.Fore.MAGENTA + "Qwen/Qwen3-0.6B" + "\n" +
              c.Fore.RESET)
        load_llm()

        try:
            answer_text = gen_answer(query, sources)
        except ValueError as e:
            print(f"{self.r}[ERROR]{self.rs}: {e}")
            exit()

        minimal_sources = [
            MinimalSource(
                file_path=fp,
                first_character_index=start,
                last_character_index=end,
            )
            for fp, start, end in sources
        ]

        _ = MinimalAnswer(question_id="single-query",
                          question=query,
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
        """Run search over a whole dataset and save a StudentSearchResults
        JSON file.

        Args:
            dataset_path: Path to a JSON RagDataset file.
            k: Number of top sources to retrieve per question.
            save_directory: Directory to write the output JSON to; the
                output filename mirrors dataset_path's basename. If
                empty, results are not saved to disk.
        """

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
            results: List[Tuple[str, int, int]] = search(
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
                    question=q.question,
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
        """Generate answers for every question in a search-results file.

        Args:
            student_search_results_path: Path to a JSON
                StudentSearchResults file (produced by search_dataset).
            save_directory: Directory to write the
                StudentSearchResultsAndAnswer JSON output to; the output
                filename mirrors student_search_results_path's basename.
        """

        if not save_directory or not save_directory.strip():
            print(f"{self.r}[ERROR]{self.rs}: save_directory cannot be empty")
            exit()

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

        start_time = time.perf_counter()

        print("\n" + c.Fore.GREEN + "".center(79, "="))
        print(" ANSWER DATASET ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        print("Loading model: " + c.Fore.GREEN + "Qwen/Qwen3-0.6B" + "\n" +
              c.Fore.RESET)

        all_answers = []

        for data in tqdm(dataset.search_results,
                         "Generating Responses", colour="GREEN"):

            sources: List[Tuple[str, int, int]] = search(
                    data.question, retriever,
                    chunk_metadata, k=dataset.k)

            try:
                a = gen_answer(data.question, sources)
            except ValueError as e:
                print(f"{self.r}[WARN]{self.rs}: {e}")
                a = ""

            minimal_sources = [
                MinimalSource(
                    file_path=fp,
                    first_character_index=start,
                    last_character_index=end,
                )
                for fp, start, end in sources
            ]

            answer = MinimalAnswer(question_id=data.question_id,
                                   question=data.question,
                                   question_str=data.question,
                                   retrieved_sources=minimal_sources,
                                   answer=a)
            all_answers.append(answer)

        output = StudentSearchResultsAndAnswer(search_results=all_answers,
                                               k=dataset.k)

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

# =============================== EVALUATE ====================================

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
                ) -> None:
        """Report recall@k for a search-results file against ground truth.

        This command is for local iteration only; the official recall@k
        used during the defense is computed by the moulinette.

        Args:
            student_search_results_path: Path to a JSON
                StudentSearchResults file to evaluate.
            dataset_path: Path to the ground-truth JSON RagDataset file.
            k: Maximum k to report recall@k for (1, 3, 5, 10, and k
                itself are reported when applicable).
        """

        if (k <= 0):
            print("k cannot be less than or equal to 0")
            exit()

        try:
            with open(student_search_results_path, "r") as f:
                results = StudentSearchResults.model_validate(json.load(f))
        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: "
                  f"Could not load student results: {e}")
            exit()

        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except Exception as e:
            print(f"{self.r}[ERROR]{self.rs}: Could not load dataset: {e}")
            exit()

        start_time = time.perf_counter()

        print("\n" + c.Fore.LIGHTMAGENTA_EX + "".center(79, "="))
        print(" EVALUATE ".center(79, "="))
        print("".center(79, "=") + self.ra + "\n\n")

        truth = ground_truth(dataset)

        with_sources = sum(1 for s in truth.values() if s)
        with_student = sum(
            1 for r in results.search_results if r.retrieved_sources)

        print("Student data is valid: True")
        print(f"Total number of questions: {len(dataset.rag_questions)}")
        print(f"Total number of questions with sources: {with_sources}")
        print("Total number of questions with student sources: "
              f"{with_student}\n")

        ks = [n for n in (1, 3, 5, 10) if n <= k]
        if (k not in ks):
            ks.append(k)

        recalls = []
        evaluated = 0
        for n in tqdm(ks, "Evaluating", colour="MAGENTA"):
            score, evaluated = recall_at_k(results, truth, n)
            recalls.append((n, score))

        print("\n\n" + c.Fore.LIGHTMAGENTA_EX + "Evaluation Results" +
              self.ra)
        print("".center(40, "="))
        print(f"Questions evaluated: {evaluated}")
        for n, score in recalls:
            print(f"Recall@{n}: {score:.3f}")

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print("\n" + c.Fore.LIGHTMAGENTA_EX +
              " Evaluation complete! ".center(79) + self.ra)
        print(f"{execution_time: .2f}s".center(79) + "\n")


# *****************************************************************************
# *                                MAIN                                       *
# *                                                                           *

def main() -> None:
    """Entry point: expose RagSystem as a Python Fire CLI."""
    try:
        fire.Fire(RagSystem)
    except KeyboardInterrupt:
        print("STOOOOOOOOOOOOOOOOOOOOOOOOP")


if __name__ == "__main__":
    main()
