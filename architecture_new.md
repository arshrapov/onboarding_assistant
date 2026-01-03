# Onboarding Assistant - Architecture Documentation (Updated)

## 5. OPERATIONAL WORKFLOWS

### 5.1 Repository Onboarding Workflow

**Complete execution flow when a user submits a repository**:

#### 1. Job Creation (State: `CREATED`)
- User submits repository URL via Gradio UI or REST API
- System validates GitHub URL format
- Creates `OnboardingJob` with unique UUID
- Initializes state machine in `CREATED` state
- Persists job to `/app/data/jobs/{job_id}.json`
- Starts background thread for processing
- Returns job ID and initial status to user

#### 2. Repository Cloning (State: `CLONING`)
- State machine transitions to `CLONING`
- Extract owner and repo name from URL
- Initialize LlamaIndex `GithubRepositoryReader`
- Configure GitHub token if available (for rate limits)
- Download repository content via GitHub API
- Filter files based on supported extensions
- Progress: 0% → 25%

#### 3. Document Parsing & Indexing (State: `PARSING`)
- State machine transitions to `PARSING`
- Load documents using `GithubRepositoryReader`
- Apply file filters (ignore patterns, size limits)
- Split documents into chunks (chunk_size=3000, overlap=400)
- Generate embeddings using Gemini Embedding-001
- Store vectors in ChromaDB collection
- Track metrics: file_count, chunk_count, languages
- Progress: 25% → 75%

#### 4. Overview Generation (State: `GENERATING_OVERVIEW`)
- State machine transitions to `GENERATING_OVERVIEW`
- Initialize query engine with indexed data
- Execute structured analysis queries:
  - Project purpose and description
  - Technology stack identification
  - Key components and modules
  - Architecture patterns
- Generate comprehensive overview using Gemini LLM
- Store overview in job metadata
- Progress: 75% → 95%

#### 5. Completion (State: `COMPLETED`)
- State machine transitions to `COMPLETED`
- Calculate final metrics (duration, file count, etc.)
- Update job with completion timestamp
- Persist final state to disk
- Progress: 100%
- User can now query the repository

#### Error Handling (State: `FAILED`)
- On any exception, transition to `FAILED` state
- Capture error message and stack trace
- Store error details in job metadata
- Clean up partial resources
- Log error for debugging
- User notified of failure with error message

### 5.2 Q&A Conversation Workflow

**Multi-turn conversation handling**:

1. **User Submits Question**
   - User selects repository from dropdown (Gradio) or provides job_id (API)
   - Enters question in natural language
   - System validates repository is indexed (state = COMPLETED)

2. **Context Retrieval**
   - Load conversation memory for this repository
   - Query ChromaDB for semantically similar code chunks
   - Retrieve top K (default: 10) most relevant chunks
   - Include metadata (file paths, languages, etc.)

3. **Response Generation**
   - Construct context from retrieved chunks
   - Include conversation history from memory
   - Send to Gemini LLM with system prompt
   - Generate context-aware response
   - Update conversation memory

4. **Response Delivery**
   - Format response with markdown
   - Include source file references
   - Display in Gradio chat interface or return via API
   - Maintain conversation state for follow-up questions

### 5.3 Docker Deployment Workflow

**Production deployment steps**:

1. **Build Image**
   ```bash
   docker build -t onboarding-assistant .
   ```
   - Multi-stage build for optimized size
   - Pre-download NLTK data in builder stage
   - Copy only runtime dependencies to final image

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with GEMINI_API_KEY
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```
   - Starts FastAPI + Gradio container
   - Mounts data volume for persistence
   - Exposes port 7860
   - Runs health checks every 30s

4. **Monitor and Manage**
   ```bash
   # View logs
   docker-compose logs -f

   # Check health
   curl http://localhost:7860/api/health

   # Restart service
   docker-compose restart

   # Scale (future)
   docker-compose up -d --scale onboarding-assistant=3
   ```

---

## 6. TECHNOLOGY STACK DETAILS

### 6.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.109.0 | Web framework for REST API |
| `uvicorn` | 0.27.0 | ASGI server |
| `gradio` | 4.12.0 | Web UI framework |
| `llama-index` | 0.10.35+ | RAG orchestration |
| `llama-index-vector-stores-chroma` | 0.1.6 | ChromaDB integration |
| `llama-index-llms-gemini` | 0.1.8 | Gemini LLM integration |
| `llama-index-embeddings-gemini` | 0.1.6 | Gemini embeddings |
| `llama-index-readers-github` | 0.1.9 | GitHub repository loader |
| `chromadb` | 0.4.22 | Vector database |
| `google-generativeai` | 0.3.2 | Google Gemini API client |
| `pydantic` | 2.5.3 | Data validation |
| `pydantic-settings` | 2.1.0 | Settings management |
| `nltk` | 3.8.1 | Natural language processing |

### 6.2 Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | 7.4.3 | Testing framework |
| `pytest-asyncio` | 0.21.1 | Async test support |
| `black` | 23.12.1 | Code formatting |
| `ruff` | 0.1.9 | Linting |

### 6.3 Configuration Parameters

**Environment Variables**:

```python
# Required
GEMINI_API_KEY: str              # Google Gemini API key

# Optional - Server
SERVER_HOST: str = "0.0.0.0"     # Server bind address
SERVER_PORT: int = 7860          # Server port

# Optional - LLM
GEMINI_MODEL: str = "models/gemini-2.5-flash"
EMBEDDING_MODEL: str = "models/embedding-001"
MAX_TOKENS: int = 8192           # Max response tokens
TEMPERATURE: float = 0.7         # LLM temperature (0-1)

# Optional - RAG
CHUNK_SIZE: int = 3000           # Code chunk size
CHUNK_OVERLAP: int = 400         # Chunk overlap
TOP_K_RESULTS: int = 10          # Retrieval count

# Optional - File Processing
MAX_FILE_SIZE_MB: int = 5        # Max file size
SUPPORTED_EXTENSIONS: str = ".py,.js,.ts,..."  # Comma-separated

# Optional - Storage
DATA_DIR: str = "/app/data"      # Base data directory
REPOS_DIR: str = "/app/data/repos"
INDEXES_DIR: str = "/app/data/indexes"
CACHE_DIR: str = "/app/data/cache"

# Optional - ChromaDB
CHROMA_MODE: str = "embedded"    # "embedded" or "server"
CHROMA_COLLECTION_PREFIX: str = "onboarding_"

# Optional - GitHub
GITHUB_TOKEN: str = None         # For private repos / higher rate limits
```

---

## 7. API SPECIFICATION

### 7.1 REST API Endpoints

#### POST /api/v1/onboarding
**Start repository onboarding**

Request:
```json
{
  "repo_url": "https://github.com/owner/repo",
  "force_reclone": false
}
```

Response (202 Accepted):
```json
{
  "job_id": "uuid",
  "status": "created",
  "repo_url": "string",
  "collection_name": "string",
  "message": "Repository onboarding started in background",
  "progress_percent": 0
}
```

#### GET /api/v1/onboarding/{job_id}
**Get job status and details**

Response (200 OK):
```json
{
  "job_id": "uuid",
  "status": "parsing",
  "repo_url": "string",
  "collection_name": "string",
  "progress_percent": 65,
  "total_files": 1234,
  "total_chunks": 5678,
  "languages_detected": ["python", "javascript"],
  "created_at": "2026-01-03T10:00:00Z",
  "updated_at": "2026-01-03T10:05:30Z",
  "error": null,
  "project_overview": null
}
```

#### GET /api/v1/onboarding/{job_id}/overview
**Get project overview (when completed)**

Response (200 OK):
```json
{
  "job_id": "uuid",
  "repo_url": "string",
  "project_overview": "## Project Overview\n\n...",
  "status": "completed"
}
```

#### GET /api/v1/onboarding
**List all jobs**

Response (200 OK):
```json
{
  "jobs": [
    {
      "job_id": "uuid",
      "status": "completed",
      "repo_url": "string",
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ],
  "total": 5
}
```

#### GET /api/health
**Health check**

Response (200 OK):
```json
{
  "status": "healthy"
}
```

---

## 8. DEPLOYMENT ARCHITECTURE

### 8.1 Container Architecture

```
Docker Host
├── onboarding-assistant container
│   ├── FastAPI (port 7860)
│   ├── Gradio UI (mounted at /)
│   ├── Python 3.11 runtime
│   ├── ChromaDB (embedded mode)
│   └── App processes
│       ├── Main thread (FastAPI/Gradio)
│       └── Background threads (job processing)
│
└── Volumes
    └── onboarding-data:/app/data
        ├── repos/      # Repository content
        ├── indexes/    # ChromaDB vectors
        ├── cache/      # Temporary files
        └── jobs/       # Job state JSON
```

### 8.2 Resource Requirements

**Minimum** (for testing):
- CPU: 1 core
- RAM: 2 GB
- Storage: 5 GB

**Recommended** (for production):
- CPU: 2-4 cores
- RAM: 4-8 GB
- Storage: 20-50 GB

**Scaling Considerations**:
- Stateless API layer (can run multiple instances)
- Shared volume for job state (requires NFS or similar)
- External ChromaDB server for distributed setup
- Load balancer for multiple instances

### 8.3 Security Considerations

1. **Container Security**:
   - Runs as non-root user (`appuser`)
   - Minimal base image (Python 3.11-slim)
   - No unnecessary packages installed

2. **API Security**:
   - CORS configured for trusted origins
   - Rate limiting (optional, via middleware)
   - Input validation with Pydantic

3. **Data Security**:
   - API keys via environment variables (not hardcoded)
   - Volume permissions restricted to appuser
   - No sensitive data in logs

4. **GitHub Token**:
   - Optional (only for private repos)
   - Stored as environment variable
   - Minimal required permissions (read-only)

---

## 9. MONITORING AND OBSERVABILITY

### 9.1 Health Checks

**Docker Health Check**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1
```

**Application Health**:
- `/api/health` endpoint returns service status
- Checks FastAPI is responding
- Can be extended to check ChromaDB connectivity

### 9.2 Logging

**Log Levels**:
- `INFO`: Job lifecycle events, API requests
- `WARNING`: Recoverable errors, rate limits
- `ERROR`: Failures, exceptions

**Log Format** (JSON structured):
```json
{
  "timestamp": "2026-01-03T10:00:00Z",
  "level": "INFO",
  "message": "Job created",
  "job_id": "uuid",
  "repo_url": "string"
}
```

**Docker Logging**:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 9.3 Metrics

**Job Metrics** (tracked per job):
- Duration (total, per state)
- File count
- Chunk count
- Languages detected
- Error count

**System Metrics** (can be added):
- API request latency
- ChromaDB query performance
- LLM API call duration
- Memory usage
- Disk usage

---

## 10. FUTURE ENHANCEMENTS

### 10.1 Planned Features

1. **Incremental Indexing**:
   - Detect repository changes
   - Re-index only modified files
   - Reduce processing time for updates

2. **Multi-repository Search**:
   - Search across multiple indexed repos
   - Cross-repository code similarity
   - Dependency graph analysis

3. **Advanced Analytics**:
   - Code quality metrics
   - Complexity analysis
   - Architecture visualization

4. **Collaboration Features**:
   - Shared repository indexes
   - Team annotations
   - Knowledge base building

5. **Performance Optimizations**:
   - Caching layer (Redis)
   - Async job queue (Celery)
   - Distributed ChromaDB

### 10.2 Scalability Roadmap

1. **Phase 1** (Current):
   - Single instance deployment
   - Embedded ChromaDB
   - Threading for background jobs

2. **Phase 2** (Near-term):
   - External ChromaDB server
   - Job queue (Celery + Redis)
   - Multiple worker instances

3. **Phase 3** (Long-term):
   - Kubernetes deployment
   - Horizontal pod autoscaling
   - Distributed vector store (Milvus/Qdrant)
   - Multi-region deployment

---

## 11. CONCLUSION

### System Strengths

- **Modularity**: Clean separation of concerns enables easy component replacement
- **Scalability**: Stateless API design allows horizontal scaling
- **Maintainability**: Type hints, clear structure, comprehensive documentation
- **Reliability**: State machine ensures consistent job processing
- **Flexibility**: Supports both web UI and programmatic API access
- **Simplicity**: Minimal external dependencies, easy deployment

### Current Limitations

- Single-instance deployment (no distributed job queue)
- Embedded ChromaDB (not optimal for multi-instance)
- GitHub-only support (no GitLab, Bitbucket, etc.)
- English-only documentation in code
- No authentication/authorization system

### Production Readiness

The system is ready for:
- Individual developer use
- Small team deployments (< 10 users)
- Internal corporate tools
- MVP/prototype deployments

For enterprise scale, consider:
- Adding job queue (Celery)
- External vector database
- Authentication system
- Rate limiting
- Monitoring/alerting
- High availability setup