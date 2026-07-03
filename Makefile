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


UV		= uv run python -m student.src

RAW_SEARCH_QUERY	= $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
RAW_ANSWER_QUERY	= $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))


DATASET		= datasets_public/public/UnansweredQuestions/dataset_docs_public.json

GT_DATASET	= datasets_public/public/AnsweredQuestions/dataset_docs_public.json
SEARCH_OUT	= data/output/search_results
ANSWER_OUT	= data/output/search_results_and_answer
REPO		= data/raw/vllm-0.10.1
K		= 10

RED		= \033[0;31m
GREEN		= \033[0;32m
YELLOW		= \033[0;33m
BLUE		= \033[0;34m
PINK		= \033[35m
NC		= \033[0m

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
	uv venv .venv --python 3.10
	uv sync

run : check-venv
	@$(UV)

debug : check-venv
	@uv run python -m pdb -m student.src

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
	@echo "  $(GREEN)make search Q=\"...\"$(NC)   Test the retriever on a question"
	@echo "  $(GREEN)make answer Q=\"...\"$(NC)   Complete RAG chain for a given question"
	@echo "  $(GREEN)make search_dataset$(NC)     Search the entire dataset"
	@echo "  $(GREEN)make answer_dataset$(NC)     Generates LLM responses for the dataset"
	@echo "  $(GREEN)make evaluate$(NC)           Calculate Recall@k"
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

# make search Q="What is vLLM ?"
search : check-venv check-index
	@QUERY="$(strip $(if $(Q),$(Q),$(RAW_SEARCH_QUERY)))"; \
	$(UV) search --query="$$QUERY" --k=$(K)

# make answer Q="What is vLLM ?"
answer : check-venv check-index
	@QUERY="$(strip $(if $(Q),$(Q),$(RAW_ANSWER_QUERY)))"; \
	echo ""; \
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

evaluate : check-venv
	@echo ""
	@echo "$(PINK)EVALUATION RECALL@K...$(NC)"
	@$(UV) evaluate \
		--student_answer_path=$(SEARCH_OUT)/$(notdir $(DATASET)) \
		--dataset_path=$(GT_DATASET) \
		--k=$(K)

# ── Qualité du code ──────────────────────────────────────────────────────── #

lint : check-venv
	@echo ""
	@echo "$(RED)TESTING FLAKE8 / MYPY...$(NC)"
	@uv run flake8 student/
	@uv run mypy student/ --cache-dir .mypy_cache \
		--warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
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
        answer_dataset evaluate lint clean clean_index \
        clean_output fclean check-venv check-index
