# Onboarding Assistant

AI-powered repository onboarding system with Retrieval-Augmented Generation (RAG) for interactive code understanding and automated documentation generation.

## Overview

**Onboarding Assistant** helps developers quickly understand and navigate new codebases by:
- Automatically cloning and analyzing GitHub repositories
- Creating semantic vector indexes for intelligent code search
- Generating comprehensive project overviews using AI
- Providing an interactive Q&A interface for code exploration
- Supporting multiple programming languages and frameworks

### Key Features

- **Automated Repository Analysis**: Clone and process GitHub repos automatically
- **Semantic Code Search**: Vector-based search using ChromaDB and Gemini embeddings
- **AI-Powered Insights**: Generate project overviews using Google Gemini LLM
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, C++, C#, Ruby, PHP, and more
- **Interactive UI**: Web interface built with Gradio for easy interaction
- **REST API**: Complete FastAPI backend for programmatic access
- **Docker Ready**: Containerized deployment with Docker and Docker Compose
- **State Management**: Robust state machine for tracking onboarding progress

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                          │
├─────────────────────────┬───────────────────────────────────────┤
│   Gradio Web UI (/)     │     FastAPI REST API (/api/v1)        │
└──────────┬──────────────┴────────────────┬──────────────────────┘
           │                               │
           └───────────┬───────────────────┘
                       │
           ┌───────────▼────────────────────────────┐
           │   RepositoryOnboardingService          │
           │  (Main Orchestration Layer)            │
           └───────────┬────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ State        │ │   RAG    │ │  GitHub Utils  │
│ Machine      │ │  Engine  │ │  File Filter   │
└──────────────┘ └──────────┘ └────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌─────────┐ ┌───────────────┐
│  LlamaIndex  │ │ChromaDB │ │ Gemini LLM    │
│   (RAG)      │ │(Vectors)│ │ & Embeddings  │
└──────────────┘ └─────────┘ └───────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│           Persistent Storage                  │
│  - Repos: /app/data/repos                    │
│  - Indexes: /app/data/indexes                │
│  - Cache: /app/data/cache                    │
└──────────────────────────────────────────────┘
```

### System Components

#### 1. **API Layer** ([app/api/](app/api/))
- **[onboarding_routes.py](app/api/onboarding_routes.py)**: REST endpoints for job management
  - `POST /api/v1/onboarding` - Start repository onboarding
  - `GET /api/v1/onboarding/{job_id}` - Get job status and details
  - `GET /api/v1/onboarding/{job_id}/overview` - Retrieve project overview
  - `GET /api/v1/onboarding` - List all onboarding jobs
- **[onboarding_schemas.py](app/api/onboarding_schemas.py)**: Pydantic request/response schemas

#### 2. **Service Layer** ([app/services/](app/services/))
- **[onboarding_service.py](app/services/onboarding_service.py)**: Main orchestration service
  - Job lifecycle management
  - Async background processing
  - State transitions and metrics tracking
- **[rag_engine.py](app/services/rag_engine.py)**: RAG implementation using LlamaIndex
  - Repository document loading via GitHub API
  - Vector index creation with ChromaDB
  - Semantic search and retrieval
- **[state_machine.py](app/services/state_machine.py)**: State management
  - Enforces valid state transitions
  - Tracks metrics and history

#### 3. **UI Layer** ([app/ui/](app/ui/))
- **[gradio_app.py](app/ui/gradio_app.py)**: Gradio web interface
  - Repository input and onboarding initiation
  - Real-time progress tracking
  - Interactive Q&A with RAG system

#### 4. **Core Models** ([app/core/](app/core/))
- **[models.py](app/core/models.py)**: Data models
  - `OnboardingJob`: Job entity with state and metrics
  - `OnboardingState`: State enum (CREATED → CLONING → PARSING → GENERATING_OVERVIEW → COMPLETED/FAILED)
  - `StateTransition`, `StateMetrics`: Tracking objects
- **[exceptions.py](app/core/exceptions.py)**: Custom exceptions

#### 5. **Utilities** ([app/utils/](app/utils/))
- **[github_utils.py](app/utils/github_utils.py)**: GitHub URL parsing and validation
- **[file_filter.py](app/utils/file_filter.py)**: File filtering and language detection

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI 0.109.0 | REST API backend |
| **Web Server** | Uvicorn 0.27.0 | ASGI server |
| **Web UI** | Gradio 4.12.0 | Interactive interface |
| **RAG Framework** | LlamaIndex 0.10.35+ | RAG orchestration |
| **LLM** | Google Gemini 2.5 Flash | Text generation & analysis |
| **Embeddings** | Gemini Embedding-001 | Semantic embeddings |
| **Vector DB** | ChromaDB 0.4.22 | Vector storage & retrieval |
| **Code Parsing** | tree-sitter 0.20.4 | Language parsing |
| **Validation** | Pydantic 2.5.3 | Data validation |
| **Testing** | pytest 7.4.3 | Unit & integration tests |

### Workflow States

```
CREATED
   ↓
CLONING (Load repository from GitHub)
   ↓
PARSING (Index files and create vectors)
   ↓
GENERATING_OVERVIEW (AI analysis)
   ↓
COMPLETED ✓

(Any state) → FAILED ✗ (on error)
```

---

## Installation & Setup

### Prerequisites

- **Python 3.11+** (recommended: 3.11)
- **Git** (for repository cloning)
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Google Gemini API Key** (required) - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Option 1: Local Development Setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/onboarding_assistant.git
cd onboarding_assistant
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Step 4: Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

**Required Configuration:**
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

**Optional Configuration:**
- `GITHUB_TOKEN`: GitHub personal access token (for private repos or higher rate limits)
- `GEMINI_MODEL`: Model to use (default: `models/gemini-2.5-flash`)
- `CHUNK_SIZE`: Code chunk size for indexing (default: 1000)
- `TOP_K_RESULTS`: Number of search results (default: 5)

See [.env.example](.env.example) for all available options.

#### Step 5: Create Data Directories

```bash
mkdir -p data/repos data/indexes data/cache
```

#### Step 6: Run the Application

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Access the application:
- **Web UI**: http://localhost:7860
- **API Docs**: http://localhost:7860/docs
- **Health Check**: http://localhost:7860/api/health

### Option 2: Docker Deployment

#### Step 1: Configure Environment

```bash
# Create .env file with your configuration
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

#### Step 2: Build and Run with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Step 3: Access the Application

- **Web UI**: http://localhost:7860
- **API Docs**: http://localhost:7860/docs

#### Docker Commands

```bash
# Rebuild after code changes
docker-compose up -d --build

# Check service status
docker-compose ps

# View resource usage
docker stats onboarding-assistant

# Access container shell
docker exec -it onboarding-assistant /bin/bash

# Remove all data (reset)
docker-compose down -v
```

---

## Usage Examples

### 1. Web UI Usage

#### Add Repository
1. Navigate to http://localhost:7860
2. Go to **"Добавить репозиторий"** tab
3. Enter GitHub URL (e.g., `https://github.com/facebook/react`)
4. Click **"Запустить индексацию"**
5. Monitor progress percentage and status

#### View Repository List
1. Go to **"Список репозиториев"** tab
2. View all indexed repositories
3. See status and metadata for each

#### Interactive Q&A
1. Go to **"Вопросы и ответы"** tab
2. Select indexed repository
3. Ask questions like:
   - "What is the project structure?"
   - "How does authentication work?"
   - "Where are the API endpoints defined?"
4. Receive AI-powered answers based on code context

### 2. REST API Usage

#### Start Onboarding

```bash
curl -X POST "http://localhost:7860/api/v1/onboarding" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/facebook/react",
    "force_reclone": false
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created",
  "repo_url": "https://github.com/facebook/react",
  "collection_name": "onboarding_react_abc123def",
  "message": "Repository onboarding started in background",
  "progress_percent": 0
}
```

#### Check Job Status

```bash
curl "http://localhost:7860/api/v1/onboarding/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "parsing",
  "repo_url": "https://github.com/facebook/react",
  "collection_name": "onboarding_react_abc123def",
  "progress_percent": 65,
  "total_files": 1234,
  "total_chunks": 5678,
  "languages_detected": ["javascript", "typescript"],
  "created_at": "2026-01-03T10:00:00Z",
  "updated_at": "2026-01-03T10:05:30Z",
  "error": null,
  "project_overview": null
}
```

#### Get Project Overview

```bash
curl "http://localhost:7860/api/v1/onboarding/550e8400-e29b-41d4-a716-446655440000/overview"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "repo_url": "https://github.com/facebook/react",
  "project_overview": "## Обзор проекта React\n\nReact - это библиотека JavaScript для создания пользовательских интерфейсов...",
  "status": "completed"
}
```

#### List All Jobs

```bash
curl "http://localhost:7860/api/v1/onboarding"
```

### 3. Python SDK Usage

```python
import requests

BASE_URL = "http://localhost:7860/api/v1"

# Start onboarding
response = requests.post(
    f"{BASE_URL}/onboarding",
    json={
        "repo_url": "https://github.com/fastapi/fastapi",
        "force_reclone": False
    }
)
job = response.json()
job_id = job["job_id"]

# Poll for completion
import time
while True:
    status = requests.get(f"{BASE_URL}/onboarding/{job_id}").json()
    print(f"Status: {status['status']} ({status['progress_percent']}%)")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(5)

# Get overview
if status["status"] == "completed":
    overview = requests.get(f"{BASE_URL}/onboarding/{job_id}/overview").json()
    print(overview["project_overview"])
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `GITHUB_TOKEN` | ✅ Yes | - | GitHub personal access token |
| `GEMINI_MODEL` | ❌ No | `models/gemini-2.5-flash` | Gemini model for generation |
| `EMBEDDING_MODEL` | ❌ No | `models/embedding-001` | Gemini embedding model |
| `MAX_TOKENS` | ❌ No | `8192` | Max tokens for LLM responses |
| `TEMPERATURE` | ❌ No | `0.7` | LLM temperature (0-1) |
| `CHUNK_SIZE` | ❌ No | `1000` | Code chunk size for indexing |
| `CHUNK_OVERLAP` | ❌ No | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | ❌ No | `5` | Number of retrieval results |
| `MAX_FILE_SIZE_MB` | ❌ No | `5` | Max file size to process |
| `SERVER_HOST` | ❌ No | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | ❌ No | `7860` | Server port |
| `CHROMA_MODE` | ❌ No | `embedded` | ChromaDB mode (`embedded` or `server`) |

### Supported File Extensions

Default supported extensions:
```
.py, .js, .ts, .tsx, .jsx, .java, .go, .rs, .cpp, .c, .h, .hpp,
.cs, .rb, .php, .md, .txt, .json, .yaml, .yml
```

Configure via `SUPPORTED_EXTENSIONS` environment variable.

---

## API Reference

### Endpoints

#### `POST /api/v1/onboarding`
Start repository onboarding process.

**Request Body:**
```json
{
  "repo_url": "string (required)",
  "force_reclone": "boolean (optional, default: false)"
}
```

**Response:** `202 Accepted`
```json
{
  "job_id": "uuid",
  "status": "created",
  "repo_url": "string",
  "collection_name": "string",
  "message": "string",
  "progress_percent": 0
}
```

#### `GET /api/v1/onboarding/{job_id}`
Get job status and details.

**Response:** `200 OK`
```json
{
  "job_id": "uuid",
  "status": "string",
  "repo_url": "string",
  "collection_name": "string",
  "progress_percent": "number",
  "total_files": "number",
  "total_chunks": "number",
  "languages_detected": ["string"],
  "created_at": "datetime",
  "updated_at": "datetime",
  "error": "string | null",
  "project_overview": "string | null"
}
```

#### `GET /api/v1/onboarding/{job_id}/overview`
Get project overview (only available when status is "completed").

**Response:** `200 OK`
```json
{
  "job_id": "uuid",
  "repo_url": "string",
  "project_overview": "string",
  "status": "string"
}
```

#### `GET /api/v1/onboarding`
List all onboarding jobs.

**Response:** `200 OK`
```json
{
  "jobs": [
    {
      "job_id": "uuid",
      "status": "string",
      "repo_url": "string",
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ],
  "total": "number"
}
```

#### `GET /api/health`
Health check endpoint.

**Response:** `200 OK`
```json
{
  "status": "healthy"
}
```

---

## Development

### Project Structure

```
onboarding_assistant/
├── app/
│   ├── main.py                  # FastAPI + Gradio entry point
│   ├── config.py                # Configuration management
│   ├── api/                     # REST API endpoints
│   ├── core/                    # Data models & exceptions
│   ├── services/                # Business logic
│   ├── ui/                      # Gradio interface
│   └── utils/                   # Utilities
├── data/                        # Persistent data (gitignored)
│   ├── repos/                   # Cloned repositories
│   ├── indexes/                 # Vector indexes
│   └── cache/                   # Job cache
├── tests/                       # Unit & integration tests
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── Dockerfile                   # Docker build
├── docker-compose.yml           # Docker orchestration
└── README.md                    # This file
```

### Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_onboarding_service.py
```

### Code Style

The project uses standard Python conventions:
- PEP 8 style guide
- Type hints for function signatures
- Docstrings for public functions
- Async/await for I/O operations

### Adding New Features

1. **New API Endpoint**: Add route in [app/api/onboarding_routes.py](app/api/onboarding_routes.py)
2. **New Service Logic**: Extend [app/services/onboarding_service.py](app/services/onboarding_service.py)
3. **New State**: Add to `OnboardingState` enum in [app/core/models.py](app/core/models.py)
4. **New UI Component**: Modify [app/ui/gradio_app.py](app/ui/gradio_app.py)

---

## Troubleshooting

### Common Issues

#### Issue: "GEMINI_API_KEY is required"
**Solution**: Set `GEMINI_API_KEY` in your `.env` file or environment variables.

#### Issue: Docker container fails to start
**Solution**: Check logs with `docker-compose logs` and verify environment variables.

#### Issue: Repository cloning fails
**Solution**:
- Check if repo is public or provide `GITHUB_TOKEN` for private repos
- Verify GitHub URL format (should be `https://github.com/owner/repo`)

#### Issue: Out of memory during indexing
**Solution**:
- Reduce `CHUNK_SIZE` in configuration
- Increase Docker memory limits in `docker-compose.yml`
- Filter out large binary files

#### Issue: ChromaDB collection already exists
**Solution**: Set `force_reclone: true` when starting onboarding to recreate index.

### Logs

**Docker:**
```bash
docker-compose logs -f onboarding-assistant
```

**Local:**
Logs are printed to stdout. Use `--log-level debug` for verbose output:
```bash
uvicorn app.main:app --log-level debug
```

---

## Performance Considerations

### Optimization Tips

1. **Chunk Size**: Adjust `CHUNK_SIZE` based on file types
   - Smaller chunks (500-1000): Better for precise code search
   - Larger chunks (2000-3000): Better for contextual understanding

2. **Vector Store**: Use `CHROMA_MODE=server` for production with multiple instances

3. **Concurrent Jobs**: System supports multiple concurrent onboarding jobs

4. **File Filtering**: Configure `SUPPORTED_EXTENSIONS` to exclude irrelevant files

### Resource Requirements

| Deployment | CPU | Memory | Storage |
|------------|-----|--------|---------|
| Minimal | 1 core | 2 GB | 5 GB |
| Recommended | 2 cores | 4 GB | 20 GB |
| Production | 4+ cores | 8+ GB | 50+ GB |

---

## Security Considerations

1. **API Keys**: Never commit `.env` file to version control
2. **GitHub Token**: Use tokens with minimal required permissions
3. **Docker**: Application runs as non-root user (`appuser`)
4. **File Size Limits**: Configured via `MAX_FILE_SIZE_MB` to prevent abuse
5. **CORS**: Configure allowed origins in production

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Support & Documentation

- **Documentation**: See [architecture.md](architecture.md) for detailed architecture
- **Dependencies**: See [DEPENDENCIES.md](DEPENDENCIES.md) for module dependency graph
- **Design Decisions**: See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for trade-offs and rationale
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join GitHub Discussions for questions

---

## Acknowledgments

- **LlamaIndex**: RAG framework
- **Google Gemini**: LLM and embeddings
- **ChromaDB**: Vector database
- **FastAPI**: Web framework
- **Gradio**: UI framework

---

**Built with ❤️ for developer productivity**
