# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: bbeaurai <bbeaurai@student.42lehavre.fr    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/06/26 11:22:22 by bbeaurai          #+#    #+#              #
#    Updated: 2026/07/03 15:34:08 by bbeaurai         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


UV		= uv run python -m src

# TYPE=docs (default) or TYPE=code — e.g. `make evaluate TYPE=code`
TYPE		?= docs

DATASET		= data/datasets/UnansweredQuestions/dataset_$(TYPE)_public.json

GT_DATASET	= data/datasets/AnsweredQuestions/dataset_$(TYPE)_public.json
SEARCH_OUT	= data/output/search_results/UnansweredQuestions
ANSWER_OUT	= data/output/search_results_and_answer/UnansweredQuestions
REPO		= data/raw/vllm-0.10.1
K		= 10

RED		= \033[0;31m
GREEN		= \033[0;32m
YELLOW		= \033[0;33m
BLUE		= \033[0;34m
PINK		= \033[35m
NC		= \033[0m

# Lets `make search "some question"` / `make answer "some question"` work
# without Q=: everything after the target name is joined back into one
# string. This only covers the query text — "-k"/"--k" can never be
# passed this way (make's own option parser swallows them as an
# abbreviation of --keep-going before the Makefile is even read), so
# overriding k still requires K=10.
RAW_QUERY	= $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

ifneq ($(filter search answer,$(firstword $(MAKECMDGOALS))),)
.DEFAULT:
	@:
endif

check-venv :
	clear
	@test -d .venv || (echo "" && \
		echo "$(RED)ERREUR : venv not found.$(NC)" && \
		echo "$(YELLOW)first throw :$(NC) make install" && \
		echo "" && exit 1)

check-index :
	@test -d data/processed/bm25_index || (echo "" && \
		echo "$(YELLOW)Index not found, indexing...$(NC)" && \
		$(MAKE) index)

# ── Targets obligatoires (sujet) ────────────────────────────────────────── #

install :
	clear
	uv sync

run : check-venv
	@$(UV)

debug : check-venv
	@uv run python -m pdb -m src

all : help

help :
	clear
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║              RAG — commandes Make            ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "  $(GREEN)make $(NC)                   Install dependencies"
	@echo "  $(GREEN)make index$(NC)              Index the vLLM (BM25) repository"
	@echo "  $(GREEN)make search \"...\" [K=10]$(NC)  Test the retriever on a question"
	@echo "  $(GREEN)make answer \"...\" [K=10]$(NC)  Complete RAG chain for a given question"
	@echo "  $(GREEN)make search_dataset$(NC)     Search the entire dataset"
	@echo "  $(GREEN)make answer_dataset$(NC)     Generates LLM responses for the dataset"
	@echo "  $(GREEN)make evaluate$(NC)           Recall@k on docs AND code"
	@echo "  $(GREEN)make evaluate_one TYPE=code$(NC)  Recall@k on one dataset"
	@echo ""
	@echo "  $(YELLOW)make lint$(NC)               Flake8 + mypy"
	@echo "  $(YELLOW)make clean$(NC)              Clears Python caches"
	@echo "  $(YELLOW)make clean_index$(NC)        Deletes the BM25 index"
	@echo "  $(YELLOW)make clean_output$(NC)       Deletes the output files"
	@echo "  $(YELLOW)make fclean$(NC)             Clean everything"
	@echo ""

# ── Pipeline RAG ────────────────────────────────────────────────────────── #

index : check-venv
	@$(UV) index --repo_path=$(REPO)

# Usage: make search "What is vLLM ?" [K=10]  — or  make search Q="..." K=10
# "-k"/"--k" cannot be passed positionally (see RAW_QUERY note above);
# use K=10 to override the number of results.
search : check-venv check-index
	@QUERY="$(strip $(if $(Q),$(Q),$(RAW_QUERY)))"; \
	$(UV) search --query="$$QUERY" --k=$(K)

# Usage: make answer "What is vLLM ?" [K=10]  — or  make answer Q="..." K=10
answer : check-venv check-index
	@echo ""
	@QUERY="$(strip $(if $(Q),$(Q),$(RAW_QUERY)))"; \
	$(UV) answer --query="$$QUERY" --k=$(K)

SEARCH_RESULT	= $(SEARCH_OUT)/$(notdir $(DATASET))

search_dataset : check-venv check-index
	@echo ""
	@$(UV) search_dataset --dataset_path=$(DATASET) --k=$(K) \
		--save_directory=$(SEARCH_OUT)

$(SEARCH_RESULT) :  # if
	@$(MAKE) search_dataset

answer_dataset : check-venv check-index $(SEARCH_RESULT)
	@echo ""
	@$(UV) answer_dataset \
		--student_search_results_path=$(SEARCH_RESULT) \
		--save_directory=$(ANSWER_OUT)

# Shows both recalls required by the evaluation scale:
# docs Recall@5 >= 80% and code Recall@5 >= 50%
evaluate : check-venv check-index
	@$(MAKE) --no-print-directory prepare_search TYPE=docs
	@$(MAKE) --no-print-directory prepare_search TYPE=code
	@$(MAKE) --no-print-directory evaluate_one TYPE=docs
	@$(MAKE) --no-print-directory evaluate_one TYPE=code

prepare_search : $(SEARCH_RESULT)

# make evaluate_one TYPE=docs|code — evaluate a single dataset
evaluate_one : check-index $(SEARCH_RESULT)
	@echo ""
	@echo "$(PINK)EVALUATION RECALL@K — $(TYPE)$(NC)"
	@$(UV) evaluate \
		--student_search_results_path=$(SEARCH_RESULT) \
		--dataset_path=$(GT_DATASET) \
		--k=$(K)

# ── Qualité du code ──────────────────────────────────────────────────────── #

lint : check-venv
	@echo ""
	@echo "$(RED)TESTING FLAKE8 / MYPY...$(NC)"
	@uv run flake8 .
	@uv run mypy . --cache-dir .mypy_cache \
		--warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@echo ""

lint-strict : check-venv
	@echo ""
	@echo "$(RED)TESTING FLAKE8 / MYPY --strict...$(NC)"
	@uv run flake8 .
	@uv run mypy . --cache-dir .mypy_cache --strict
	@echo ""

# ── Nettoyage ────────────────────────────────────────────────────────────── #

clean :
	@echo ""
	@echo "$(RED)CLEANING CACHES...$(NC)"
	@find . -name "__pycache__" -exec rm -rf {} \+
	@find . -name ".mypy_cache" -exec rm -rf {} \+
	@find . -name ".vscode"     -exec rm -rf {} \+
	@find . -name "*.pyc"       -exec rm -f  {} \+
	@echo "$(GREEN)DELETE [OK]$(NC)"

clean_index :
	@echo ""
	@echo "$(RED)DELETION INDEX BM25...$(NC)"
	@rm -rf data/processed/bm25_index data/processed/chunks
	@echo "$(GREEN)INDEX DELETE$(NC)"

clean_output :
	@echo ""
	@echo "$(RED)DELETION OUTPUTS...$(NC)"
	@rm -rf data/output
	@echo "$(GREEN)OUTPUTS DELETE$(NC)"

fclean : clean clean_index clean_output

.PHONY: all help install run debug index search answer search_dataset \
        answer_dataset evaluate evaluate_one prepare_search lint lint-strict \
        clean clean_index clean_output fclean check-venv check-index
