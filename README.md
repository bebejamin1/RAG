# RAG against the machine

*This project has been created as part of the 42 curriculum by bbeaurai.*

---

## Checklist de progression

### Setup & Structure du projet
- [X] Initialiser le projet avec `uv` (`uv init`)
- [X] Créer `pyproject.toml` et générer `uv.lock`
- [X] Créer le dossier `src/` avec le module principal (`src/__main__.py` ou `src/__init__.py`)
- [X] Ajouter un `Makefile` avec les règles : `install`, `run`, `debug`, `clean`, `lint`
- [X] Vérifier que `.gitignore` exclut `__pycache__`, `.mypy_cache`, données générées, poids de modèles
- [X] Python 3.10+ uniquement
- [X] `uv run python -m src` doit fonctionner

### Dépendances recommandées à installer
- [X] `pydantic` — validation des modèles
- [X] `fire` — CLI
- [X] `tqdm` — barres de progression
- [X] `bm25s` ou `scikit-learn` — retrieval (BM25 ou TF-IDF)
- [X] `transformers` — LLM (Qwen3-0.6B)
- [X] `flake8` + `mypy` — qualité de code

### Modèles Pydantic (7 obligatoires)
- [ ] `MinimalSource` — `file_path`, `first_character_index`, `last_character_index`
- [ ] `UnansweredQuestion` — `question_id` (uuid), `question`
- [ ] `AnsweredQuestion(UnansweredQuestion)` — `sources`, `answer`
- [ ] `RagDataset` — `rag_questions: List[AnsweredQuestion | UnansweredQuestion]`
- [ ] `MinimalSearchResults` — `question_id`, `question`, `retrieved_sources`
- [ ] `MinimalAnswer(MinimalSearchResults)` — `answer`
- [ ] `StudentSearchResults` — `search_results`, `k`
- [ ] `StudentSearchResultsAndAnswer(StudentSearchResults)` — `search_results: List[MinimalAnswer]`

### Ingestion / Indexing	
- [ ] Lire tous les fichiers utiles du repo `vllm-0.10.1/` (`.py`, `.md`, `.rst`...)
- [ ] Implémenter le chunking **Python** (par fonction/classe via AST)
- [ ] Implémenter le chunking **Markdown/texte** (par section/paragraphe)
- [ ] Taille max des chunks : **2000 caractères**, configurable via `--max_chunk_size`
- [ ] Chaque chunk stocke : `file_path`, `first_character_index`, `last_character_index`
- [ ] Construire l'index BM25 ou TF-IDF
- [ ] Sauvegarder l'index dans `data/processed/` pour réutilisation
- [ ] Temps d'indexing < **5 minutes**
- [ ] CLI : `uv run python -m src index --max_chunk_size 2000`

### Retrieval System
- [ ] Implémenter **BM25** ou **TF-IDF** (au moins un des deux)
- [ ] Retourner les top-k résultats avec `file_path`, `first_character_index`, `last_character_index`
- [ ] CLI : `uv run python -m src search "query" --k 10`
- [ ] CLI : `uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir>`
- [ ] Output JSON valide conforme à `StudentSearchResults`
- [ ] Cold start latency < **60 secondes**
- [ ] Throughput : 1000 questions en < **90 secondes** (après cold start)

### Answer Generation
- [ ] Utiliser `Qwen/Qwen3-0.6B` comme modèle **par défaut**
- [ ] Passer les chunks récupérés en contexte au LLM (dans les limites de tokens)
- [ ] Générer des réponses : self-contained, source-grounded, faithful, relevant
- [ ] CLI : `uv run python -m src answer "query" --k 10`
- [ ] CLI : `uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>`
- [ ] Output JSON valide conforme à `StudentSearchResultsAndAnswer`

### Evaluation System
- [ ] Implémenter la métrique **Recall@k**
- [ ] Overlap de 5% minimum entre source récupérée et source correcte = "found"
- [ ] Score par question : `nb_found / total_correct_sources`
- [ ] CLI : `uv run python -m src evaluate --student_answer_path <path> --dataset_path <path> --k 10`

### Performances cibles (obligatoires)
- [ ] **Docs Recall@5 >= 80%** (test sur dataset docs privé, 100 questions)
- [ ] **Code Recall@5 >= 50%** (test sur dataset code privé, 100 questions)

### Gestion des cas limites (edge cases)
- [ ] Query vide : `search "" --k 10` → pas de crash
- [ ] Query absurde : `search "asdfghjkl" --k 10` → pas de crash
- [ ] k=0 : `answer "..." --k 0` → pas de crash
- [ ] Chemin dataset inexistant : `search_dataset --dataset_path /nonexistent.json` → pas de crash
- [ ] Aucune exception Python non gérée (pas de traceback visible)

### Qualité du code
- [ ] Toutes les classes utilisent **pydantic**
- [ ] Code conforme **flake8** (pas d'erreurs)
- [ ] **Type hints** sur toutes les fonctions (paramètres + retours)
- [ ] **mypy** passe sans erreurs (`--disallow-untyped-defs --check-untyped-defs`)
- [ ] **Docstrings** sur les classes et fonctions (style Google ou NumPy)
- [ ] Exceptions gérées avec `try-except` partout
- [ ] Context managers pour les ressources (fichiers, connexions)
- [ ] **tqdm** pour toutes les opérations longues
- [ ] Aucun import de packages `moulinette` : `grep -rn "moulinette" src/ --include="*.py"` → rien

### README (obligatoire, 5/6 sections minimum)
- [ ] Première ligne italique avec login 42
- [ ] Section **Description** (but du projet, overview)
- [ ] Section **Instructions** (installation, exécution)
- [ ] Section **System Architecture** (pipeline RAG, composants)
- [ ] Section **Chunking Strategy** (approche de segmentation)
- [ ] Section **Retrieval Method** (algorithme + ranking)
- [ ] Section **Performance Analysis** (recall@k obtenu)
- [ ] Section **Design Decisions** (choix d'implémentation + trade-offs)
- [ ] Section **Challenges** (difficultés rencontrées + solutions)
- [ ] Section **Example Usage** (exemples de commandes)
- [ ] Section **Resources** (références + utilisation de l'IA)

---

## Bonus (optionnel, max 5 points)

- [ ] **+1pt** — Query expansion (synonymes, query rewriting)
- [ ] **+1pt** — Semantic embeddings pour le retrieval
- [ ] **+1pt** — Caching (index cache, query cache)
- [ ] **+2pts** — Hybrid retrieval (BM25 + embeddings)
- [ ] **+2pts** — Inférence LLM via **vLLM** (local serving)

---

## Ordre suggéré pour commencer

1. Setup `uv` + structure `src/` + CLI basique avec `fire`
2. Modèles Pydantic (copier-coller du sujet, adapter)
3. Ingestion + chunking basique (taille fixe pour commencer)
4. BM25 simple → tester Recall@5 sur le dataset public
5. Améliorer le chunking (Python AST + Markdown sections)
6. Intégrer Qwen3-0.6B pour la génération
7. Edge cases + flake8/mypy
8. README complet
9. Bonus si le temps le permet


pytest et unittest pour faire des tests
utiliser des venv

pendant la creation dune ia il faut lentrainer
pour developper
- la comprehension du langage
- le raisonnement et lanalyse structurelle
pour cela il faut fournir une enorme quantite de donnees

apres ca le model se souvient de ce quil a appris mais il ne connais que les donnees quon lui a fournies
et donc pour quil connaissent plus de chose faut le reentrainer

Lentrainement est une technique et le RAG en est une autre
le RAG donne acces a une source dinfo externe de notre choix plutot que de fournir des donnees

RAG ==== Génération Augmentée par Récupération | Retrieving Augmented Generation

1er etape
Indexation:
les donnees doivents etre indexees. etape structure et organise linfo afin de la rendre cherchable

2eme etape
Recuperation:
pour entrainer le model il doit interroger la base de donnees pour recuperer les extraits les plus utiles.
Le model doit comprendre la question, une fois cela fait il met en correspondance la requete avec la base indexee pour choisir le meilleurs resultats, les informations les plus pertinents cela implique ||| l'encodage de la requête, la recherche par similarité et le classement (ranking).

3eme etape
Augmentation:
Une fois que lia a trouver linformation elle peut la combiner a ce quelle sait deja.
Mais on se basera autant que possible sur les donnes recuperee que sur la connaissance interne du modele
car les deux peut conduire a des reponses obsoletes ou hallucinees. on peut recuperer nettoyer et les filtrer pour retirer les extraits non pertinents
afin deviter le bruit pour inserer dans la ||| fenêtre de contexte

4eme etape
Generation:
le llm lit la fenetre de contexte

Vous pouvez utiliser les bibliothèques que vous voulez ; nous recommandons vivement les paquets
transformers, dspy, fire, tqdm, langchain, bm25s, chromadb.

• Vous devez utiliser uv comme gestionnaire de projet et de paquets.

• Votre système doit fournir une interface en ligne de commande (CLI) en utilisant Python Fire.

• Des barres de progression doivent être implémentées pour les opérations longues à l'aide de tqdm.

• Évaluer la qualité de votre système de récupération à l'aide des métriques recall@k

CLI        ??????????

chunk max 2000char
