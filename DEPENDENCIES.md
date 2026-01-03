# Module Dependencies Visualization

This document provides a comprehensive visualization of module dependencies and interactions within the Onboarding Assistant project.

---

## Table of Contents

1. [Dependency Graph](#dependency-graph)
2. [Layer Architecture](#layer-architecture)
3. [Module-Level Dependencies](#module-level-dependencies)
4. [External Dependencies](#external-dependencies)
5. [Data Flow](#data-flow)
6. [Integration Points](#integration-points)

---

## Dependency Graph

### High-Level Component Dependencies

```
┌────────────────────────────────────────────────────────────────────┐
│                          Entry Point                                │
│                       app/main.py (FastAPI)                         │
└─────────────────────────┬──────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────────┐ ┌──────────┐ ┌─────────────────┐
│   API Routes    │ │  Gradio  │ │   Config        │
│ onboarding_     │ │   UI     │ │  (pydantic-     │
│    routes.py    │ │gradio_   │ │   settings)     │
└────────┬────────┘ │  app.py  │ └────────┬────────┘
         │          └─────┬────┘          │
         │                │               │
         └────────────────┼───────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │  RepositoryOnboardingService       │
         │     (services/onboarding_service)   │
         └────────┬──────────────┬────────────┘
                  │              │
         ┌────────┼──────────────┼────────┐
         │        │              │        │
         ▼        ▼              ▼        ▼
    ┌────────┐ ┌─────┐  ┌──────────┐  ┌────────┐
    │  State │ │ RAG │  │  GitHub  │  │  File  │
    │Machine │ │Engine│  │  Utils   │  │ Filter │
    └────────┘ └──┬──┘  └──────────┘  └────────┘
                  │
         ┌────────┼────────┐
         │        │        │
         ▼        ▼        ▼
    ┌────────┐ ┌────┐  ┌────────┐
    │ Llama  │ │Chroma│ │Gemini │
    │ Index  │ │  DB  │ │  API  │
    └────────┘ └────┘  └────────┘
         │        │        │
         └────────┼────────┘
                  ▼
         ┌─────────────────┐
         │  File System    │
         │  (data/)        │
         └─────────────────┘
```

---

## Layer Architecture

### 1. Presentation Layer

```
┌──────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                         │
├─────────────────────────┬────────────────────────────────────┤
│                         │                                    │
│  ┌───────────────────┐  │  ┌──────────────────────────────┐  │
│  │   Gradio Web UI   │  │  │      FastAPI REST API        │  │
│  │ (app/ui/gradio_   │  │  │  (app/api/onboarding_routes) │  │
│  │       app.py)     │  │  │                              │  │
│  │                   │  │  │  Endpoints:                  │  │
│  │ Features:         │  │  │  - POST /api/v1/onboarding   │  │
│  │ - Add repository  │  │  │  - GET /api/v1/onboarding/id │  │
│  │ - View list       │  │  │  - GET /api/v1/onboarding    │  │
│  │ - Q&A interface   │  │  │  - GET /api/health           │  │
│  └─────────┬─────────┘  │  └──────────────┬───────────────┘  │
│            │            │                 │                  │
│            └────────────┼─────────────────┘                  │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
                  Depends on schemas
                          │
              ┌───────────▼───────────┐
              │ API Schemas           │
              │ (Pydantic Models)     │
              │ - OnboardingRequest   │
              │ - OnboardingResponse  │
              │ - OnboardingStatus... │
              └───────────────────────┘
```

**Dependencies:**
- [app/ui/gradio_app.py](app/ui/gradio_app.py)
  - → [app/services/onboarding_service.py](app/services/onboarding_service.py)
  - → [app/services/rag_engine.py](app/services/rag_engine.py)
  - → [app/core/models.py](app/core/models.py)
  - → [app/config.py](app/config.py)

- [app/api/onboarding_routes.py](app/api/onboarding_routes.py)
  - → [app/api/onboarding_schemas.py](app/api/onboarding_schemas.py)
  - → [app/services/onboarding_service.py](app/services/onboarding_service.py)
  - → [app/core/exceptions.py](app/core/exceptions.py)

---

### 2. Service Layer

```
┌────────────────────────────────────────────────────────────────┐
│                       SERVICE LAYER                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │     RepositoryOnboardingService (Main Orchestrator)      │  │
│  │                                                          │  │
│  │  Key Methods:                                            │  │
│  │  • create_job() ────────────────────┐                    │  │
│  │  • start_job() ─────────────────┐   │                    │  │
│  │  • get_job_status()             │   │                    │  │
│  │  • list_jobs()                  │   │                    │  │
│  │  • _process_job_async() ◄───────┘   │                    │  │
│  │  • _handle_cloning()                │                    │  │
│  │  • _handle_parsing()                │                    │  │
│  │  • _handle_generating_overview()    │                    │  │
│  └──────────┬──────────────────┬───────┼────────────────────┘  │
│             │                  │       │                       │
│             │                  │       │                       │
│     ┌───────▼──────┐  ┌────────▼───────▼──────┐               │
│     │ StateMachine │  │      RAGEngine         │               │
│     │              │  │                        │               │
│     │ • transition │  │ • load_repository()    │               │
│     │ • validate   │  │ • create_index()       │               │
│     │ • track      │  │ • search()             │               │
│     │   metrics    │  │ • collect_metadata()   │               │
│     └──────────────┘  └────────┬───────────────┘               │
│                                │                               │
└────────────────────────────────┼───────────────────────────────┘
                                 │
                   ┌─────────────┼─────────────┐
                   │             │             │
                   ▼             ▼             ▼
           ┌──────────┐  ┌────────────┐  ┌─────────┐
           │LlamaIndex│  │  ChromaDB  │  │ Gemini  │
           │Framework │  │VectorStore │  │   LLM   │
           └──────────┘  └────────────┘  └─────────┘
```

**Dependencies:**

- [app/services/onboarding_service.py](app/services/onboarding_service.py)
  - → [app/services/rag_engine.py](app/services/rag_engine.py)
  - → [app/services/state_machine.py](app/services/state_machine.py)
  - → [app/core/models.py](app/core/models.py)
  - → [app/core/exceptions.py](app/core/exceptions.py)
  - → [app/utils/github_utils.py](app/utils/github_utils.py)
  - → [app/utils/file_filter.py](app/utils/file_filter.py)
  - → [app/config.py](app/config.py)

- [app/services/rag_engine.py](app/services/rag_engine.py)
  - → `llama_index` (LlamaIndex framework)
  - → `llama_index.llms.gemini` (Gemini LLM)
  - → `llama_index.embeddings.gemini` (Gemini embeddings)
  - → `llama_index.vector_stores.chroma` (ChromaDB)
  - → `llama_index.readers.github` (GitHub reader)
  - → [app/core/exceptions.py](app/core/exceptions.py)
  - → [app/config.py](app/config.py)

- [app/services/state_machine.py](app/services/state_machine.py)
  - → [app/core/models.py](app/core/models.py)
  - → [app/core/exceptions.py](app/core/exceptions.py)

---

### 3. Core Layer

```
┌─────────────────────────────────────────────────────────┐
│                      CORE LAYER                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │              Core Models                           │ │
│  │  (app/core/models.py)                              │ │
│  │                                                    │ │
│  │  Data Classes:                                     │ │
│  │  • OnboardingJob ─────────────┐                   │ │
│  │      - job_id: UUID           │                   │ │
│  │      - repo_url: str          │                   │ │
│  │      - current_state: State   │                   │ │
│  │      - state_history: List    │                   │ │
│  │      - collection_name: str   │                   │ │
│  │      - total_files: int       │                   │ │
│  │      - languages: List[str]   │                   │ │
│  │      - overview: str | None   │                   │ │
│  │                               │                   │ │
│  │  • OnboardingState (Enum) ◄───┘                   │ │
│  │      CREATED                                       │ │
│  │      CLONING                                       │ │
│  │      PARSING                                       │ │
│  │      GENERATING_OVERVIEW                           │ │
│  │      COMPLETED                                     │ │
│  │      FAILED                                        │ │
│  │                                                    │ │
│  │  • StateTransition                                 │ │
│  │  • StateMetrics                                    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │           Custom Exceptions                        │ │
│  │  (app/core/exceptions.py)                          │ │
│  │                                                    │ │
│  │  • OnboardingAssistantError (Base)                │ │
│  │      ├─ RepositoryDownloadError                   │ │
│  │      ├─ OnboardingError                           │ │
│  │      ├─ StateTransitionError                      │ │
│  │      ├─ ParsingError                              │ │
│  │      ├─ EmbeddingError                            │ │
│  │      └─ VectorStoreError                          │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Dependencies:**

- [app/core/models.py](app/core/models.py)
  - → `dataclasses` (Python standard library)
  - → `enum` (Python standard library)
  - → `datetime` (Python standard library)

- [app/core/exceptions.py](app/core/exceptions.py)
  - → No internal dependencies (base exceptions)

---

### 4. Utilities Layer

```
┌──────────────────────────────────────────────────────────┐
│                   UTILITIES LAYER                         │
│                                                           │
│  ┌───────────────────────┐  ┌────────────────────────┐   │
│  │   GitHub Utils        │  │    File Filter         │   │
│  │                       │  │                        │   │
│  │ • parse_github_url()  │  │ • filter_repository_   │   │
│  │ • validate_github_    │  │     files()            │   │
│  │     url()             │  │ • should_process_      │   │
│  │ • get_repo_identifier│  │     file()             │   │
│  │ • normalize_github_   │  │ • get_language_from_   │   │
│  │     url()             │  │     extension()        │   │
│  │                       │  │ • count_files_by_      │   │
│  │ Used by:              │  │     language()         │   │
│  │ - onboarding_service  │  │                        │   │
│  │ - rag_engine          │  │ Used by:               │   │
│  │                       │  │ - onboarding_service   │   │
│  └───────────────────────┘  │ - rag_engine           │   │
│                              └────────────────────────┘   │
│                                                           │
│  ┌───────────────────────────────────────────────────┐   │
│  │          Configuration (Pydantic Settings)        │   │
│  │                                                   │   │
│  │  Settings Class with validation:                 │   │
│  │  • gemini_api_key: SecretStr (required)          │   │
│  │  • github_token: Optional[SecretStr]             │   │
│  │  • data_dir, repos_dir, indexes_dir, cache_dir   │   │
│  │  • gemini_model, max_tokens, temperature         │   │
│  │  • embedding_model                               │   │
│  │  • chroma_mode, chroma_host, chroma_port         │   │
│  │  • top_k_results, chunk_size, chunk_overlap      │   │
│  │  • max_file_size_mb, supported_extensions        │   │
│  │  • server_host, server_port                      │   │
│  │                                                   │   │
│  │  Loaded from: .env file or environment variables │   │
│  │  Used by: ALL modules                            │   │
│  └───────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

**Dependencies:**

- [app/utils/github_utils.py](app/utils/github_utils.py)
  - → `re` (Python standard library)
  - → `urllib.parse` (Python standard library)

- [app/utils/file_filter.py](app/utils/file_filter.py)
  - → `pathlib` (Python standard library)
  - → [app/config.py](app/config.py)

- [app/config.py](app/config.py)
  - → `pydantic` (v2.5.3)
  - → `pydantic_settings` (v2.1.0)
  - → `python-dotenv` (v1.0.0)

---

## Module-Level Dependencies

### Detailed Dependency Matrix

| Module | Depends On | Used By |
|--------|-----------|---------|
| **[app/main.py](app/main.py)** | api/*, ui/*, config | - (entry point) |
| **[app/config.py](app/config.py)** | pydantic, pydantic_settings | ALL modules |
| **[app/core/models.py](app/core/models.py)** | dataclasses, enum | services/*, api/*, ui/* |
| **[app/core/exceptions.py](app/core/exceptions.py)** | - | services/*, api/*, rag_engine |
| **[app/api/onboarding_routes.py](app/api/onboarding_routes.py)** | FastAPI, schemas, services, exceptions | main.py |
| **[app/api/onboarding_schemas.py](app/api/onboarding_schemas.py)** | pydantic, core/models | api/onboarding_routes |
| **[app/services/onboarding_service.py](app/services/onboarding_service.py)** | rag_engine, state_machine, core/*, utils/*, config | api/*, ui/* |
| **[app/services/rag_engine.py](app/services/rag_engine.py)** | llama_index, chromadb, google-generativeai, config, exceptions | onboarding_service, ui/gradio_app |
| **[app/services/state_machine.py](app/services/state_machine.py)** | core/models, core/exceptions | onboarding_service |
| **[app/ui/gradio_app.py](app/ui/gradio_app.py)** | gradio, services/*, core/models, config | main.py |
| **[app/utils/github_utils.py](app/utils/github_utils.py)** | re, urllib | services/*, rag_engine |
| **[app/utils/file_filter.py](app/utils/file_filter.py)** | pathlib, config | services/*, rag_engine |

---

## External Dependencies

### Python Package Dependencies

```
FastAPI Ecosystem
├── fastapi==0.109.0
└── uvicorn==0.27.0

LlamaIndex Framework
├── llama-index>=0.10.35
├── llama-index-llms-gemini>=0.1.3
├── llama-index-embeddings-gemini>=0.1.6
├── llama-index-vector-stores-chroma>=0.1.6
├── llama-index-readers-github>=0.1.9
└── llama-index-retrievers-bm25==0.1.3

Vector Database
├── chromadb==0.4.22

Google AI
├── google-generativeai>=0.4.1,<0.6.0

UI Framework
├── gradio==4.12.0
└── huggingface_hub>=0.19.0,<0.21.0

Data Validation
├── pydantic==2.5.3
└── pydantic-settings==2.1.0

Code Processing
├── tree-sitter==0.20.4
├── tree-sitter-languages==1.10.2
├── pygments==2.17.2
└── markdown2==2.4.12

Environment & Testing
├── python-dotenv==1.0.0
├── pytest==7.4.3
└── pytest-asyncio==0.23.0
```

### External Service Dependencies

```
┌─────────────────────────────────────────┐
│      External Services                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌────────────────────────────────┐    │
│  │  Google Gemini API             │    │
│  │  - LLM (gemini-2.5-flash)      │    │
│  │  - Embeddings (embedding-001)  │    │
│  │  Required: GEMINI_API_KEY      │    │
│  └────────────────────────────────┘    │
│              ▲                          │
│              │                          │
│  ┌───────────┴──────────────────────┐  │
│  │  GitHub API (Optional)           │  │
│  │  - Repository cloning            │  │
│  │  - Private repo access           │  │
│  │  Optional: GITHUB_TOKEN          │  │
│  └──────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

---

## Data Flow

### Onboarding Workflow Data Flow

```
1. USER INPUT
   │
   ├─► [Gradio UI] or [REST API]
   │   (GitHub URL + options)
   │
   ▼
2. REQUEST VALIDATION
   │
   ├─► onboarding_schemas.py
   │   (Pydantic validation)
   │
   ▼
3. JOB CREATION
   │
   ├─► RepositoryOnboardingService.create_job()
   │   • Generate UUID
   │   • Create collection name
   │   • Initialize state (CREATED)
   │   • Save to cache
   │
   ▼
4. ASYNC PROCESSING START
   │
   ├─► RepositoryOnboardingService._process_job_async()
   │   (Background task)
   │
   ▼
5. STATE: CLONING
   │
   ├─► _handle_cloning()
   │   ├─► RAGEngine.load_repository()
   │   │   ├─► GithubRepositoryReader (LlamaIndex)
   │   │   │   ├─► GitHub API
   │   │   │   └─► Download files
   │   │   ├─► file_filter.py (filter files)
   │   │   └─► Save to /data/repos/{owner-repo}/
   │   │
   │   └─► StateMachine.transition(CLONING → PARSING)
   │
   ▼
6. STATE: PARSING
   │
   ├─► _handle_parsing()
   │   ├─► RAGEngine.collect_overview_metadata()
   │   │   ├─► Extract README
   │   │   ├─► Detect languages
   │   │   └─► Count files
   │   │
   │   ├─► RAGEngine.create_index()
   │   │   ├─► GeminiEmbedding.get_embeddings()
   │   │   │   └─► Google Gemini API
   │   │   ├─► ChromaDB.add_documents()
   │   │   │   └─► Save to /data/indexes/
   │   │   └─► VectorStoreIndex.from_documents()
   │   │
   │   └─► StateMachine.transition(PARSING → GENERATING_OVERVIEW)
   │
   ▼
7. STATE: GENERATING_OVERVIEW
   │
   ├─► _handle_generating_overview()
   │   ├─► _analyze_collected_files()
   │   │   └─► Extract metadata (frameworks, entry points)
   │   │
   │   ├─► _query_rag_for_insights()
   │   │   ├─► RAGEngine.search() (3 queries)
   │   │   │   ├─► ChromaDB.query() (vector search)
   │   │   │   └─► Retrieve top-k results
   │   │   │
   │   │   └─► Collect architectural insights
   │   │
   │   ├─► _synthesize_overview()
   │   │   ├─► Build context (metadata + RAG results)
   │   │   ├─► Gemini LLM.complete()
   │   │   │   └─► Google Gemini API
   │   │   └─► Format response
   │   │
   │   └─► StateMachine.transition(GENERATING_OVERVIEW → COMPLETED)
   │
   ▼
8. STATE: COMPLETED
   │
   ├─► Save final job state
   │   ├─► Update cache (/data/cache/onboarding_jobs.json)
   │   └─► Store project_overview
   │
   ▼
9. USER RETRIEVAL
   │
   ├─► [GET /api/v1/onboarding/{job_id}]
   │   └─► Return job details + overview
   │
   └─► [Gradio UI]
       └─► Display overview in interface
```

### Q&A Data Flow

```
1. USER QUESTION
   │
   ├─► [Gradio Chat Interface]
   │   (User selects repo + asks question)
   │
   ▼
2. RAG QUERY
   │
   ├─► RAGEngine.search(query, collection_name)
   │   │
   │   ├─► Query Preparation
   │   │   └─► Text normalization
   │   │
   │   ├─► Vector Search
   │   │   ├─► GeminiEmbedding.get_query_embedding()
   │   │   │   └─► Google Gemini API
   │   │   ├─► ChromaDB.query(embedding)
   │   │   │   └─► Retrieve top-k similar chunks
   │   │   └─► Rank results by similarity
   │   │
   │   ├─► Context Building
   │   │   └─► Concatenate retrieved chunks
   │   │
   │   └─► LLM Generation
   │       ├─► Build prompt (context + question)
   │       ├─► Gemini LLM.complete()
   │       │   └─► Google Gemini API
   │       └─► Return answer with sources
   │
   ▼
3. RESPONSE DISPLAY
   │
   └─► [Gradio Chat Interface]
       ├─► Show answer
       └─► Show source file references
```

---

## Integration Points

### 1. LlamaIndex Integration

**Location**: [app/services/rag_engine.py](app/services/rag_engine.py)

```python
Components Used:
├── Settings (LlamaIndex global config)
│   ├── llm: Gemini LLM
│   ├── embed_model: GeminiEmbedding
│   └── text_splitter: SentenceSplitter
│
├── GithubRepositoryReader
│   └── Loads documents from GitHub
│
├── VectorStoreIndex
│   ├── ChromaVectorStore (persistent)
│   └── SimpleDocumentStore (pickle persistence)
│
└── QueryEngine
    └── Handles RAG queries
```

### 2. ChromaDB Integration

**Location**: [app/services/rag_engine.py](app/services/rag_engine.py)

```python
ChromaDB Configuration:
├── Mode: Embedded (local) or Server (remote)
├── Persist Directory: /app/data/indexes/
├── Collections: One per repository
│   └── Naming: {CHROMA_COLLECTION_PREFIX}{repo_identifier}_{hash}
│
Operations:
├── Collection Creation
│   └── Stores vectors + metadata
├── Document Insertion
│   └── Batch processing with embeddings
├── Vector Search
│   └── Cosine similarity search
└── Collection Deletion
    └── Cleanup on force_reclone
```

### 3. Google Gemini API Integration

**Location**: [app/services/rag_engine.py](app/services/rag_engine.py)

```python
Gemini LLM:
├── Model: models/gemini-2.5-flash (configurable)
├── Parameters:
│   ├── max_tokens: 8192 (configurable)
│   ├── temperature: 0.7 (configurable)
│   └── api_key: from GEMINI_API_KEY
│
├── Usage:
│   ├── Project overview generation
│   ├── RAG query responses
│   └── Code analysis
│
Gemini Embeddings:
├── Model: models/embedding-001
├── Dimension: 768 (Gemini default)
├── Usage:
│   ├── Document embedding (indexing)
│   └── Query embedding (search)
```

### 4. FastAPI Integration

**Location**: [app/main.py](app/main.py), [app/api/onboarding_routes.py](app/api/onboarding_routes.py)

```python
FastAPI App:
├── CORS Middleware (allow all origins)
├── API Router (/api/v1)
│   └── onboarding routes
├── Gradio Mount (/)
│   └── Gradio Blocks interface
└── Startup Events
    └── Create data directories
```

### 5. Gradio Integration

**Location**: [app/ui/gradio_app.py](app/ui/gradio_app.py)

```python
Gradio Blocks:
├── Tab 1: Add Repository
│   ├── Input: GitHub URL
│   ├── Button: Start indexing
│   └── Output: Status + Progress
│
├── Tab 2: Repository List
│   ├── Display: All indexed repos
│   └── Refresh button
│
└── Tab 3: Q&A
    ├── Dropdown: Select repository
    ├── Chatbot: Q&A interface
    └── Submit: Ask question

Mounted at: FastAPI root path (/)
```

---

## Circular Dependency Prevention

The project maintains a strict layered architecture with **no circular dependencies**:

1. **Presentation Layer** depends on → Service Layer
2. **Service Layer** depends on → Core Layer + Utilities
3. **Core Layer** depends on → Standard Library only
4. **Utilities** depends on → Configuration only

**Dependency Direction**: Always flows downward, never upward or circular.

---

## Summary

### Module Count by Layer

| Layer | Modules | Key Responsibilities |
|-------|---------|---------------------|
| **Entry Point** | 1 | FastAPI app initialization |
| **Presentation** | 3 | API routes, schemas, Gradio UI |
| **Service** | 3 | Orchestration, RAG, state management |
| **Core** | 2 | Data models, exceptions |
| **Utilities** | 3 | Config, GitHub utils, file filtering |
| **Total** | 12 | - |

### External Dependency Count

- **Python Packages**: 20+ packages
- **External Services**: 2 (Google Gemini API, GitHub API)
- **Databases**: 1 (ChromaDB - embedded)

---

## Dependency Update Strategy

### When to Update Dependencies

1. **Security patches**: Update immediately
2. **Bug fixes**: Update within sprint
3. **Minor versions**: Review quarterly
4. **Major versions**: Plan migration carefully

### Critical Dependencies to Monitor

- **LlamaIndex**: Core RAG functionality
- **ChromaDB**: Vector storage backend
- **FastAPI**: API framework
- **Pydantic**: Data validation (v2.x required)
- **google-generativeai**: Gemini API client

### Version Pinning Strategy

- **Exact pins** (`==`): Core packages (FastAPI, ChromaDB, Pydantic)
- **Minor pins** (`>=x.y.z,<x.y+1.0`): LlamaIndex ecosystem
- **Flexible** (`>=x.y.z`): Utilities and testing

---

**For detailed architecture and design decisions, see**:
- [README.md](README.md) - Complete project documentation
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) - Trade-offs and rationale
- [architecture.md](architecture.md) - Detailed technical architecture
