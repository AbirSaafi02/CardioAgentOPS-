# 🫀 CardioAgent-Ops

> **Plateforme micro-services conteneurisée, MLOps, fine-tuning LLM, RAG et multi-agents IA pour l'analyse cardio-acoustique**

---

## 📖 Présentation du projet

**CardioAgent-Ops** est un projet académique réalisé à l'École Nationale Supérieure d'Ingénieurs de Tunis (**ENSIT**, section 3GInfo GLID & NTS), dans le cadre des modules **Machine Learning** et **Nouvelles Architectures (DevOps)**.

Le projet vise à explorer le passage de l'expérimentation à l'industrialisation d'un système intelligent :

- **De l'audio à l'inférence :** préparation de signaux cardiaques, évaluation de modèles et inférence via ONNX.
- **Architecture découplée :** séparation des services audio, API, orchestration d'agents, RAG, LLM et MLOps.
- **Traçabilité :** suivi des entraînements et des inférences avec MLflow.
- **Automatisation :** conteneurisation Docker et mise en place prévue d'un pipeline CI/CD/CT.

Le dépôt contient déjà le moteur audio FastAPI/ONNX, l'API Gateway et la configuration du serveur MLflow. Les autres composants et certaines intégrations décrivent l'architecture cible et restent des étapes de développement du projet.

---

## 🏗️ Architecture cible — 6 micro-services

L'architecture cible sépare les responsabilités en six services, connectés sur un réseau Docker privé. Le flux ci-dessous représente la conception visée ; l'intégration complète entre tous les services est en cours.

```text
                         ┌──────────────────────────┐
                         │ Service 6: mlflow-server │
                         │ Tracking & Model Registry│
                         └────────────▲─────────────┘
                           (métriques, inférences,
                            expériences, guardrails)
                                      │
[Client / fichier WAV]               │
          │                           │
          ▼ POST /analyze_pcg         │
┌─────────────────────────┐           │
│ Service 5: api-gateway  │           │
│ FastAPI / Uvicorn       │           │
└────────────┬────────────┘           │
             │                        │
             ▼                        │
┌─────────────────────────┐           │
│ Service 3: audio-engine │───────────┘
│ Librosa / ONNX Runtime  │
└────────────┬────────────┘
             │ probabilités et caractéristiques
             ▼
┌────────────────────────────┐
│ Service 4: orchestrator    │───────────────┐
│ LangGraph multi-agents     │               │
└──────────┬────────────────┘               │
           │ contexte RAG                     │ prompt / génération
           ▼                                  ▼
┌─────────────────────┐             ┌──────────────────────┐
│ Service 2: vector-db│             │ Service 1: llm-server│
│ Documents ESC / RAG │             │ Ollama ou vLLM       │
└─────────────────────┘             └──────────────────────┘
```

### Rôle des services

1. **`llm-server` — service LLM :** exécution locale d'un modèle de langage spécialisé en cardiologie, avec quantification et format de déploiement adaptés (par exemple GGUF).
2. **`vector-db` — base vectorielle :** indexation et recherche sémantique de ressources cliniques, notamment des recommandations ESC.
3. **`audio-engine` — traitement audio :** prétraitement des fichiers, extraction de représentations audio et inférence du modèle ONNX.
4. **`agent-orchestrator` — orchestration IA :** coordination visée de trois rôles (*Analyste*, *Cardiologue RAG*, *Auditeur*) et validation structurée des réponses.
5. **`api-gateway` — API REST :** point d'entrée client, validation des fichiers WAV et routage des requêtes vers les services internes.
6. **`mlflow-server` — MLOps :** centralisation des expériences, métriques, artefacts et traces d'inférence.

---

## 🛠️ Stack technique

| Domaine | Technologies ciblées |
| :--- | :--- |
| **Conteneurisation & orchestration** | Docker, Docker Compose, réseau privé de services |
| **Langage & API** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **Traitement audio** | Librosa, Pandas, NumPy, SciPy |
| **ML / DL & inférence** | Scikit-learn, PyTorch, Torchvision, ONNX Runtime |
| **Fine-tuning & serving LLM** | Unsloth, PEFT, TRL, BitsAndBytes, Hugging Face, Ollama ou vLLM |
| **RAG & multi-agents** | LangGraph, LangChain, Qdrant ou ChromaDB, Sentence-Transformers |
| **MLOps & qualité** | MLflow, GitHub Actions, Pytest, Pytest-Cov, HTTPX |

---

## 🚀 Pipeline et jalons techniques

Les éléments ci-dessous décrivent les objectifs et axes de développement du projet ; ils ne signifient pas que chaque étape est déjà entièrement intégrée.

1. **Data engineering audio :** fusion, nettoyage et équilibrage des jeux de données **PASCAL Heart Sound Challenge** et **PhysioNet/CinC Challenge**, avec rééchantillonnage à 2 000 Hz, normalisation temporelle (fenêtres de 2 ou 5 secondes) et extraction de caractéristiques (MFCC, ZCR et spectrogrammes Mel).
2. **Benchmarking audio & MLOps :** comparer cinq approches, suivre les expériences sous MLflow, sélectionner un modèle en privilégiant le recall sur la classe pathologique, puis exporter le modèle retenu en ONNX. Les pistes de comparaison comprennent :
   - Decision Tree sur des caractéristiques MFCC ;
   - SVM avec noyau RBF ;
   - CNN 2D sur des spectrogrammes Mel ;
   - CNN pré-entraîné en transfer learning (ResNet18/50 ou VGG16) ;
   - architecture hybride Transformer-CNN (AST ou Conformer).
3. **Fine-tuning LLM :** étudier l'adaptation 4-bit d'un modèle open source (`Qwen-2.5-7B-Instruct` ou `Llama-3.1-8B-Instruct`) avec QLoRA sur un corpus médical, suivre les courbes de perte sous MLflow et viser un export local au format GGUF.
4. **Pipeline RAG :** découper les ressources cliniques, générer des embeddings et indexer les documents de référence dans le service vectoriel.
5. **Multi-agents & guardrails :** construire un workflow d'analyse, de recherche clinique et d'audit, avec un schéma de sortie structuré et validé.
6. **Continuous Testing & CI/CD :** ajouter les tests unitaires audio, les tests d'intégration API et la validation des schémas LLM, puis automatiser leur exécution et la couverture avec GitHub Actions, Pytest et Pytest-Cov.

### Moteur audio actuellement présent

Le service audio existant charge `audio-engine/models/cardio_model.onnx`. Ses paramètres sont décrits dans `audio-engine/models/cardio_model_meta.json` :

- fréquence d'échantillonnage : **2 000 Hz** ;
- durée d'une fenêtre : **5 secondes** ;
- entrée : spectrogramme log-Mel avec **40 bandes** ;
- classes : `normal` et `pathologique` ;
- exécution : ONNX Runtime sur CPU.

L'API audio expose `/health` et `/predict`. L'API Gateway expose `/health` et `/analyze_pcg`. La Gateway valide le fichier et transmet l'audio au moteur ; l'intégration du résultat avec l'orchestrateur multi-agents fait partie de la suite du projet. Les inférences peuvent être tracées dans MLflow.

> **Note :** ce projet est un prototype académique et ne constitue pas un dispositif de diagnostic médical.

---

## 📁 Structure du dépôt

```text
CardioAgent-Ops/
├── .github/
│   └── workflows/
│       └── mlops-ci-cd.yml
├── agent-orchestrator/
│   ├── graph.py
│   ├── prompts.py
│   ├── schemas.py
│   └── server.py
├── api-gateway/
│   ├── main.py
│   ├── schemas.py
│   └── Dockerfile
├── audio-engine/
│   ├── app/                  # API, prétraitement et inférence
│   ├── models/               # modèle ONNX et métadonnées
│   └── Dockerfile
├── docker/
│   └── docker-compose.yml
├── llm-server/
│   ├── finetuning/
│   ├── models/
│   └── Modelfile
├── mlflow-server/
│   └── Dockerfile
├── Notebook/
│   └── CardioAgentOps_01_audio_mlops_benchmark (1).ipynb
├── tests/
│   ├── test_agents.py
│   ├── test_api.py
│   └── test_audio.py
└── vector-db/
    ├── data/
    │   └── esc/
    └── rag/
        ├── chunking.py
        ├── embedding.py
        └── index_esc.py

---

## ⚙️ Environnement et déploiement

Le projet cible une exécution conteneurisée avec Docker et Docker Compose. Les services communiquent sur un réseau interne ; les ports publiés, volumes de données, dépendances et ressources matérielles seront définis dans la configuration de déploiement.

Le service LLM local pourra bénéficier d'un accès GPU selon le modèle choisi et les ressources disponibles. Le serveur MLflow utilise un stockage persistant pour les données de suivi et les artefacts.

Cette section présente les principes d'environnement et de déploiement, sans commandes de lancement.

---

## 🧪 Tests et intégration continue

La stratégie de validation prévue comprend :

- tests unitaires du prétraitement et de l'inférence audio ;
- tests d'intégration de l'API Gateway et de ses réponses ;
- tests des schémas et du workflow multi-agents ;
- exécution automatisée des contrôles avec GitHub Actions et suivi de la couverture.

---

## 🎓 Cadre académique

- **Établissement :** Université de Tunis — École Nationale Supérieure d'Ingénieurs de Tunis (ENSIT)
- **Filière :** 3GInfo — GLID & NTS
- **Modules :** Machine Learning & Nouvelles Architectures (DevOps)
- **Encadrant :** Dr. Mehrez Boulares
