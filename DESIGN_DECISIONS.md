# Design Decisions and Trade-offs

This document explains the key architectural decisions, design patterns, and trade-offs made during the development of the Onboarding Assistant project.

---

## Table of Contents

1. [Technology Selection](#technology-selection)
2. [Architecture Patterns](#architecture-patterns)
3. [Data Management](#data-management)
4. [API Design](#api-design)
5. [Security & Privacy](#security--privacy)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling](#error-handling)
8. [Testing Strategy](#testing-strategy)
9. [Future Considerations](#future-considerations)

---

## Technology Selection

### 1. LlamaIndex as RAG Framework

**Decision**: Use LlamaIndex as the primary RAG orchestration framework.

**Rationale**:
- **Mature Ecosystem**: Extensive integration support (ChromaDB, Gemini, GitHub)
- **High-Level Abstractions**: Simplifies complex RAG workflows
- **Active Development**: Frequent updates and bug fixes
- **Community Support**: Large community and documentation

**Alternatives Considered**:
- **LangChain**: More general-purpose, but heavier and less focused on RAG
- **Custom RAG**: More control, but significant development overhead
- **Haystack**: Great for search, but less flexible for code understanding

**Trade-offs**:
- ✅ **Pros**: Rapid development, proven patterns, easy maintenance
- ❌ **Cons**: Dependency on external framework, limited low-level control
- ⚖️ **Verdict**: Speed and reliability outweigh control limitations

---

### 2. Google Gemini for LLM and Embeddings

**Decision**: Use Google Gemini 2.5 Flash for text generation and Gemini Embedding-001 for embeddings.

**Rationale**:
- **Cost-Effective**: Gemini Flash is significantly cheaper than GPT-4
- **Speed**: Fast inference for interactive Q&A
- **Quality**: Excellent code understanding capabilities
- **Unified API**: Same provider for both LLM and embeddings
- **Free Tier**: Generous free tier for development

**Alternatives Considered**:
- **OpenAI GPT-4**: Higher quality but much more expensive
- **Claude**: Excellent for code but higher cost
- **Local LLMs (Llama 2/3)**: Free but requires GPU infrastructure
- **Azure OpenAI**: Enterprise features but complex setup

**Trade-offs**:
- ✅ **Pros**: Cost-effective, fast, good quality, easy setup
- ❌ **Cons**: Google dependency, potential API changes, data privacy concerns
- ⚖️ **Verdict**: Best balance of cost, quality, and ease of use

**Mitigation**: Configuration allows easy switching to other providers via LlamaIndex.

---

### 3. ChromaDB as Vector Database

**Decision**: Use ChromaDB for vector storage and retrieval.

**Rationale**:
- **Embedded Mode**: No separate database server required
- **Simple API**: Easy integration with LlamaIndex
- **Persistent Storage**: File-based persistence without complex setup
- **Performance**: Fast for small-to-medium datasets (<100k vectors)
- **Open Source**: No licensing costs

**Alternatives Considered**:
- **Pinecone**: Excellent performance but requires paid account
- **Weaviate**: Enterprise-grade but complex deployment
- **FAISS**: Fast but requires custom persistence logic
- **Qdrant**: Good alternative but less mature LlamaIndex integration

**Trade-offs**:
- ✅ **Pros**: Simple deployment, no infrastructure, good for MVP
- ❌ **Cons**: Limited scalability (>1M vectors), single-node only
- ⚖️ **Verdict**: Ideal for current scale, easy migration path if needed

**Scalability Plan**: Can switch to `CHROMA_MODE=server` or migrate to Pinecone/Weaviate for production scale.

---

### 4. FastAPI + Gradio Dual Interface

**Decision**: Combine FastAPI (REST API) and Gradio (web UI) in a single application.

**Rationale**:
- **Flexibility**: Support both programmatic and interactive usage
- **Gradio Simplicity**: Quick UI development without frontend code
- **FastAPI Performance**: High-performance async API
- **Single Deployment**: One container for both interfaces

**Alternatives Considered**:
- **React + FastAPI**: More customizable UI but requires frontend development
- **Streamlit**: Similar to Gradio but less flexible for API integration
- **Gradio Only**: Limited to interactive usage
- **FastAPI Only**: No built-in UI

**Trade-offs**:
- ✅ **Pros**: Fast development, dual interface, single deployment
- ❌ **Cons**: Limited UI customization, Gradio dependency
- ⚖️ **Verdict**: Perfect for MVP and internal tools, can add React later if needed

---

### 5. Python 3.11

**Decision**: Target Python 3.11 as the minimum version.

**Rationale**:
- **Performance**: 10-25% faster than Python 3.10 (PEP 659)
- **Type Hints**: Improved typing features (Self, TypeVarTuple)
- **Error Messages**: Better tracebacks
- **Stability**: Mature version (released Oct 2022)

**Trade-offs**:
- ✅ **Pros**: Performance gains, modern features, stable
- ❌ **Cons**: Excludes users on older Python versions
- ⚖️ **Verdict**: Python 3.11 is widespread enough (2+ years old)

---

## Architecture Patterns

### 1. Layered Architecture

**Decision**: Implement strict layered architecture (Presentation → Service → Core → Utilities).

**Rationale**:
- **Separation of Concerns**: Clear module responsibilities
- **Testability**: Easy to mock and test individual layers
- **Maintainability**: Changes in one layer don't affect others
- **Scalability**: Easy to add new features within existing layers

**Pattern**:
```
Presentation (API/UI) → Services (Business Logic) → Core (Models) → Utilities
```

**Trade-offs**:
- ✅ **Pros**: Clean architecture, easy testing, clear dependencies
- ❌ **Cons**: More boilerplate, learning curve for contributors
- ⚖️ **Verdict**: Essential for long-term maintainability

---

### 2. State Machine for Job Management

**Decision**: Implement explicit state machine for onboarding workflow.

**Rationale**:
- **Clarity**: Clear workflow states and transitions
- **Error Recovery**: Easy to identify where failures occur
- **Metrics**: Track time spent in each state
- **Debugging**: State history provides audit trail
- **Retry Logic**: Can resume from specific states

**States**:
```
CREATED → CLONING → PARSING → GENERATING_OVERVIEW → COMPLETED
                                                   → FAILED
```

**Trade-offs**:
- ✅ **Pros**: Robust error handling, clear workflow, easy debugging
- ❌ **Cons**: More complex than simple flags, state explosion risk
- ⚖️ **Verdict**: Necessary for reliable background processing

---

### 3. Async Background Processing

**Decision**: Use Python asyncio with background tasks for repository processing.

**Rationale**:
- **Non-Blocking**: API remains responsive during long operations
- **Scalability**: Can handle multiple concurrent jobs
- **Progress Tracking**: Users can poll for status updates
- **Resource Efficiency**: Better than threading for I/O-bound tasks

**Implementation**:
```python
asyncio.create_task(_process_job_async(job_id))
loop.run_in_executor(None, blocking_operation)
```

**Alternatives Considered**:
- **Celery**: More powerful but requires Redis/RabbitMQ infrastructure
- **Huey**: Lighter than Celery but still requires separate process
- **Threading**: GIL limitations, less efficient
- **Synchronous**: Blocks API, poor user experience

**Trade-offs**:
- ✅ **Pros**: Simple deployment, no external dependencies, good performance
- ❌ **Cons**: Jobs lost on server restart, limited to single process
- ⚖️ **Verdict**: Sufficient for MVP, can add Celery later for production

**Mitigation**: Job state persisted to disk, can implement job recovery on startup.

---

### 4. Repository-Based Vector Collections

**Decision**: Create one ChromaDB collection per repository.

**Rationale**:
- **Isolation**: No cross-contamination between repositories
- **Deletion**: Easy to remove specific repository indexes
- **Scalability**: Can shard collections across databases if needed
- **Clarity**: Clear mapping between repos and collections

**Alternatives Considered**:
- **Single Collection**: All repos in one collection with metadata filtering
- **User-Based Collections**: Collections per user
- **Hybrid**: Shared collection for public repos, separate for private

**Trade-offs**:
- ✅ **Pros**: Simple, isolated, easy cleanup, clear boundaries
- ❌ **Cons**: More collections to manage, potential overhead
- ⚖️ **Verdict**: Simplicity and isolation worth the overhead

---

### 5. File Filtering Strategy

**Decision**: Implement explicit allow-list of file extensions with size limits.

**Rationale**:
- **Performance**: Avoid processing binaries and large files
- **Quality**: Focus on relevant code and documentation
- **Cost Control**: Reduce embedding API calls
- **Configurability**: Users can adjust for their needs

**Filters Applied**:
- Extension whitelist (`.py`, `.js`, `.md`, etc.)
- Max file size (default: 5MB)
- Directory blacklist (`node_modules`, `.git`, `__pycache__`)
- Binary file detection

**Trade-offs**:
- ✅ **Pros**: Fast indexing, lower costs, better results
- ❌ **Cons**: Might miss relevant files, requires configuration
- ⚖️ **Verdict**: Necessary for practical operation, easy to extend

---

## Data Management

### 1. File-Based Job Persistence

**Decision**: Store job metadata in JSON file (`onboarding_jobs.json`).

**Rationale**:
- **Simplicity**: No database setup required
- **Portability**: Easy to backup and inspect
- **Atomicity**: Python's `json.dump` is atomic on most filesystems
- **Low Overhead**: Minimal dependencies

**Alternatives Considered**:
- **SQLite**: More robust but adds dependency
- **PostgreSQL**: Overkill for current scale
- **Redis**: Requires separate service
- **In-Memory Only**: Jobs lost on restart

**Trade-offs**:
- ✅ **Pros**: Simple, no infrastructure, easy debugging
- ❌ **Cons**: Limited concurrency, no transactions, potential corruption
- ⚖️ **Verdict**: Good for MVP, migrate to SQLite if concurrency becomes issue

**Mitigation**: Use file locking (future enhancement) or switch to SQLite.

---

### 2. Pickle for Document Storage

**Decision**: Use Python pickle for storing LlamaIndex documents.

**Rationale**:
- **LlamaIndex Standard**: LlamaIndex uses pickle for docstore persistence
- **Complete Serialization**: Preserves all document metadata
- **Performance**: Fast serialization/deserialization

**Alternatives Considered**:
- **JSON**: Human-readable but loses some object types
- **MessagePack**: Faster but less compatible
- **Database**: More complex setup

**Trade-offs**:
- ✅ **Pros**: Full compatibility with LlamaIndex, simple, fast
- ❌ **Cons**: Security risk (untrusted data), not human-readable, Python-only
- ⚖️ **Verdict**: Safe for trusted internal data, standard LlamaIndex approach

**Security Note**: Never deserialize pickles from untrusted sources.

---

### 3. Directory Structure

**Decision**: Organize data in separate directories by type.

**Structure**:
```
data/
├── repos/         # Repository documents (pickles)
├── indexes/       # ChromaDB vector stores
└── cache/         # Job metadata (JSON)
```

**Rationale**:
- **Clarity**: Easy to understand and navigate
- **Backup**: Can backup/restore specific types
- **Cleanup**: Easy to clear specific data types
- **Volumes**: Easy to mount in Docker

**Trade-offs**:
- ✅ **Pros**: Organized, easy to manage, clear ownership
- ❌ **Cons**: More directories to create and manage
- ⚖️ **Verdict**: Organizational benefits worth the complexity

---

## API Design

### 1. REST API Pattern

**Decision**: Implement RESTful API following standard conventions.

**Endpoints**:
- `POST /api/v1/onboarding` - Create resource (start job)
- `GET /api/v1/onboarding/{id}` - Retrieve resource (job status)
- `GET /api/v1/onboarding` - List resources (all jobs)
- `GET /api/v1/onboarding/{id}/overview` - Sub-resource access

**Rationale**:
- **Standard**: Well-understood by developers
- **HTTP Semantics**: Proper use of methods and status codes
- **Versioning**: `/api/v1` allows future changes
- **Tooling**: Works with standard HTTP clients

**Trade-offs**:
- ✅ **Pros**: Standard, well-documented, tool-friendly
- ❌ **Cons**: Less flexible than GraphQL for complex queries
- ⚖️ **Verdict**: REST is perfect for current use case

---

### 2. Async Job Pattern

**Decision**: Return immediately (202 Accepted) and allow polling for status.

**Flow**:
1. `POST /api/v1/onboarding` → `202 Accepted` + `job_id`
2. Client polls `GET /api/v1/onboarding/{job_id}` for status
3. When `status == "completed"`, retrieve results

**Rationale**:
- **Responsiveness**: API doesn't block during long operations
- **User Experience**: Users see progress updates
- **Scalability**: Server can handle multiple concurrent jobs
- **Timeout Avoidance**: No HTTP timeout issues

**Alternatives Considered**:
- **Synchronous**: Blocks until complete (timeouts, poor UX)
- **WebSockets**: Real-time updates but more complex
- **Server-Sent Events (SSE)**: Good for streaming but limited browser support
- **Webhooks**: Requires callback URL from client

**Trade-offs**:
- ✅ **Pros**: Simple, scalable, no timeout issues
- ❌ **Cons**: Requires polling, delayed results
- ⚖️ **Verdict**: Standard pattern for long-running operations

**Enhancement**: Could add WebSocket support for real-time updates in future.

---

### 3. Pydantic for Validation

**Decision**: Use Pydantic v2 for all request/response validation.

**Rationale**:
- **Type Safety**: Compile-time and runtime type checking
- **Validation**: Automatic validation of inputs
- **Documentation**: Auto-generated OpenAPI schemas
- **FastAPI Integration**: Native FastAPI support
- **Error Messages**: Clear validation error responses

**Trade-offs**:
- ✅ **Pros**: Type safety, auto validation, great DX, documentation
- ❌ **Cons**: Learning curve, schema definition overhead
- ⚖️ **Verdict**: Essential for production-grade API

---

## Security & Privacy

### 1. API Key Management

**Decision**: Use environment variables for API keys, never hardcode.

**Implementation**:
- `GEMINI_API_KEY`: Required, loaded via pydantic-settings
- `GITHUB_TOKEN`: Optional, for private repos
- `SecretStr` type in config to prevent accidental logging

**Alternatives Considered**:
- **Config Files**: Risk of committing to git
- **Secret Management Services**: Overkill for current scale
- **Encrypted Config**: Complex key management

**Trade-offs**:
- ✅ **Pros**: Industry standard, Docker-friendly, simple
- ❌ **Cons**: Visible in process list, requires `.env` management
- ⚖️ **Verdict**: Standard approach for containerized apps

**Best Practices**:
- ✅ `.env` in `.gitignore`
- ✅ Provide `.env.example` template
- ✅ Use `SecretStr` to prevent accidental logging
- ✅ Validate required keys on startup

---

### 2. Non-Root Docker User

**Decision**: Run Docker container as non-root user (`appuser`, UID 1000).

**Rationale**:
- **Security**: Limit damage from container escape
- **Best Practice**: Docker security recommendation
- **Compliance**: Many security policies require non-root

**Implementation**:
```dockerfile
RUN useradd -m -u 1000 -s /bin/bash appuser
USER appuser
```

**Trade-offs**:
- ✅ **Pros**: Enhanced security, compliance
- ❌ **Cons**: Potential permission issues with volumes
- ⚖️ **Verdict**: Security benefit outweighs minor inconvenience

---

### 3. File Size Limits

**Decision**: Implement configurable file size limit (default: 5MB).

**Rationale**:
- **DoS Prevention**: Prevent processing huge files
- **Cost Control**: Limit embedding API costs
- **Performance**: Avoid memory issues

**Trade-offs**:
- ✅ **Pros**: Prevents abuse, controls costs
- ❌ **Cons**: Might skip legitimate large files
- ⚖️ **Verdict**: Necessary protection, configurable if needed

---

### 4. No Authentication (Current)

**Decision**: No authentication in current version.

**Rationale**:
- **Use Case**: Internal tool, trusted environment
- **Simplicity**: Faster MVP development
- **Deployment**: Expected to run behind corporate firewall/VPN

**Security Implications**:
- ⚠️ Anyone with access can query any indexed repository
- ⚠️ No rate limiting on API calls
- ⚠️ No user-based access control

**Mitigation for Production**:
- Add API key authentication
- Implement rate limiting
- Add user-based repository access control
- Use OAuth2 for enterprise deployment

**Trade-offs**:
- ✅ **Pros**: Simple, fast development
- ❌ **Cons**: Not suitable for public deployment
- ⚖️ **Verdict**: Acceptable for internal tools, must add auth for public use

---

## Performance Optimization

### 1. Chunking Strategy

**Decision**: Use configurable chunk size (default: 1000 chars) with overlap (default: 200 chars).

**Rationale**:
- **Context Preservation**: Overlap prevents splitting related code
- **Retrieval Quality**: Smaller chunks = more precise matches
- **Embedding Cost**: Smaller chunks = more API calls
- **Memory**: Larger chunks = more memory per query

**Tuning Guidance**:
- **Small chunks (500-1000)**: Better for precise code search
- **Medium chunks (1000-2000)**: Balanced
- **Large chunks (2000-3000)**: Better for conceptual questions

**Trade-offs**:
- ✅ **Pros**: Configurable, preserves context, good defaults
- ❌ **Cons**: Requires tuning, cost increases with smaller chunks
- ⚖️ **Verdict**: Good default, allows optimization per use case

---

### 2. Top-K Retrieval

**Decision**: Retrieve top 5 results by default (configurable).

**Rationale**:
- **Quality**: Top results usually most relevant
- **Speed**: Fewer results = faster processing
- **Context Limits**: LLM context window constraints
- **Cost**: Fewer results = lower API costs

**Trade-offs**:
- ✅ **Pros**: Fast, cost-effective, usually sufficient
- ❌ **Cons**: Might miss relevant context
- ⚖️ **Verdict**: Good default, increase if needed for complex queries

---

### 3. Async I/O with Thread Pool

**Decision**: Use `loop.run_in_executor()` for blocking operations.

**Rationale**:
- **Non-Blocking**: Repository cloning and indexing don't block event loop
- **Concurrency**: Can process multiple jobs simultaneously
- **Resource Control**: ThreadPoolExecutor limits concurrent threads

**Implementation**:
```python
await loop.run_in_executor(None, rag_engine.load_repository, repo_url)
```

**Trade-offs**:
- ✅ **Pros**: Good concurrency, simple implementation
- ❌ **Cons**: Thread overhead, GIL limitations for CPU-bound tasks
- ⚖️ **Verdict**: Ideal for I/O-bound operations (API calls, file I/O)

---

### 4. Caching Strategy

**Decision**: Cache repository documents and indexes on disk.

**Caching Layers**:
1. **Repository Documents**: Pickled documents in `/data/repos/`
2. **Vector Indexes**: ChromaDB persistence in `/data/indexes/`
3. **Job Metadata**: JSON cache in `/data/cache/`

**Invalidation**:
- Manual: `force_reclone: true` parameter
- No automatic invalidation (repositories treated as immutable snapshots)

**Trade-offs**:
- ✅ **Pros**: Fast re-indexing, lower API costs, offline querying
- ❌ **Cons**: Stale data if repository updates, disk space usage
- ⚖️ **Verdict**: Appropriate for onboarding use case (point-in-time snapshots)

**Enhancement**: Could add TTL-based cache invalidation or webhook-based updates.

---

## Error Handling

### 1. Custom Exception Hierarchy

**Decision**: Define custom exception types for different error categories.

**Hierarchy**:
```python
OnboardingAssistantError (base)
├── RepositoryDownloadError
├── OnboardingError
├── StateTransitionError
├── ParsingError
├── EmbeddingError
└── VectorStoreError
```

**Rationale**:
- **Granular Handling**: Different recovery strategies per error type
- **Debugging**: Clear error source
- **API Responses**: Appropriate HTTP status codes per error type
- **Logging**: Categorized error logging

**Trade-offs**:
- ✅ **Pros**: Clear error taxonomy, better debugging
- ❌ **Cons**: More boilerplate code
- ⚖️ **Verdict**: Essential for robust error handling

---

### 2. State Machine Error Recovery

**Decision**: Failed jobs transition to FAILED state with retry metadata.

**Recovery Strategy**:
- Track attempt count in state metrics
- Allow manual retry from appropriate state
- Preserve error details in job metadata
- No automatic retry (to prevent infinite loops)

**Trade-offs**:
- ✅ **Pros**: Explicit error handling, no infinite loops
- ❌ **Cons**: Requires manual intervention
- ⚖️ **Verdict**: Safer than automatic retry for long-running jobs

**Enhancement**: Could add exponential backoff retry for transient errors.

---

### 3. Graceful Degradation

**Decision**: Continue processing even if some files fail.

**Example**: If one file fails to parse, skip it and continue with others.

**Trade-offs**:
- ✅ **Pros**: Partial results better than complete failure
- ❌ **Cons**: Might miss important files
- ⚖️ **Verdict**: Better user experience, log errors for investigation

---

## Testing Strategy

### 1. Pytest for Testing

**Decision**: Use pytest with pytest-asyncio for async support.

**Rationale**:
- **Industry Standard**: Most popular Python testing framework
- **Rich Ecosystem**: Extensive plugin support
- **Async Support**: pytest-asyncio for testing async code
- **Fixtures**: Easy dependency injection and test setup

**Trade-offs**:
- ✅ **Pros**: Powerful, well-documented, community support
- ❌ **Cons**: None significant
- ⚖️ **Verdict**: Clear choice for Python testing

---

### 2. Test Coverage (Current Gap)

**Current State**: Minimal test coverage (TODO).

**Recommended Strategy**:
1. **Unit Tests**: Core logic (state machine, utilities)
2. **Integration Tests**: Service layer with mocked external APIs
3. **E2E Tests**: Full workflow with test repositories
4. **API Tests**: FastAPI endpoint testing

**Priority**:
1. State machine transitions (critical path)
2. GitHub URL parsing (edge cases)
3. File filtering (various file types)
4. API endpoints (contract testing)

**Trade-offs**:
- ✅ **Pros**: Test coverage ensures reliability
- ❌ **Cons**: Development time, test maintenance
- ⚖️ **Verdict**: Critical for production deployment

---

## Future Considerations

### 1. Scalability Improvements

**Current Limitations**:
- Single-process async (limited by CPU cores)
- File-based job storage (concurrency issues)
- ChromaDB embedded mode (single-node only)

**Scaling Path**:
1. **Phase 1 (Current)**: Single-process async (good for <10 concurrent jobs)
2. **Phase 2**: Add Celery + Redis for distributed task queue
3. **Phase 3**: Migrate to ChromaDB server mode or Pinecone
4. **Phase 4**: Add horizontal scaling with load balancer

**When to Upgrade**:
- Phase 2: >10 concurrent jobs or job persistence requirements
- Phase 3: >100k vectors or multi-instance deployment
- Phase 4: >1000 requests/min

---

### 2. Authentication & Authorization

**Future Options**:
1. **API Key Auth**: Simple, good for service-to-service
2. **OAuth2/OIDC**: Enterprise SSO integration
3. **JWT**: Stateless auth for distributed systems
4. **Role-Based Access Control (RBAC)**: User permissions

**Implementation Priority**:
1. API key auth (simplest, 80% use case)
2. OAuth2 (enterprise deployment)
3. RBAC (multi-tenant deployment)

---

### 3. Multi-Tenant Support

**Current State**: All users share same data directory.

**Multi-Tenancy Options**:
1. **Collection-Based**: Tenant ID in collection name
2. **Directory-Based**: Separate data directory per tenant
3. **Database-Based**: Separate database per tenant

**Trade-offs**:
- **Collection-Based**: Simple, shared infrastructure, limited isolation
- **Directory-Based**: Good isolation, more complex management
- **Database-Based**: Full isolation, highest resource cost

**Recommendation**: Start with collection-based for cost efficiency.

---

### 4. Alternative LLM Support

**Current**: Tightly coupled to Google Gemini.

**Extensibility**:
- LlamaIndex abstractions support multiple providers
- Add configuration for provider selection
- Implement adapter pattern for provider-specific logic

**Potential Providers**:
- OpenAI GPT-4 (higher quality, higher cost)
- Anthropic Claude (excellent code understanding)
- Local models (Llama 3, CodeLlama)
- Azure OpenAI (enterprise compliance)

**Implementation**: Provider factory pattern + configuration.

---

### 5. Real-Time Updates

**Current**: Polling-based status updates.

**Alternatives**:
1. **WebSockets**: Bidirectional real-time communication
2. **Server-Sent Events (SSE)**: Server-to-client streaming
3. **Long Polling**: Better than short polling, worse than WebSockets

**When to Implement**: If polling overhead becomes issue.

---

### 6. Code Navigation Features

**Potential Enhancements**:
- **Jump to Definition**: Link code references in answers
- **Call Graph Visualization**: Show function dependencies
- **File Tree View**: Browse repository structure in UI
- **Syntax Highlighting**: Code blocks with language detection
- **Diff Viewer**: Compare versions

**Implementation**: Requires tree-sitter integration + frontend development.

---

### 7. Advanced RAG Techniques

**Current**: Simple vector similarity search.

**Enhancements**:
1. **Hybrid Search**: Combine vector search + BM25 keyword search
2. **Reranking**: Use cross-encoder for result reranking
3. **Query Decomposition**: Break complex queries into sub-queries
4. **Contextual Compression**: Remove irrelevant context before LLM
5. **Citation Tracking**: Precise source attribution

**Priority**: Hybrid search has best ROI for quality improvement.

---

## Summary of Key Trade-offs

| Decision | Pros | Cons | Verdict |
|----------|------|------|---------|
| **LlamaIndex** | Fast development, proven | Framework dependency | ✅ Use |
| **Gemini LLM** | Cost-effective, good quality | Vendor lock-in | ✅ Use |
| **ChromaDB** | Simple, no infrastructure | Limited scalability | ✅ MVP, plan migration |
| **Async Jobs** | Non-blocking, scalable | Jobs lost on restart | ✅ Use, add persistence |
| **File-Based Storage** | Simple, portable | Limited concurrency | ✅ MVP, migrate to DB |
| **No Auth** | Fast development | Not production-ready | ⚠️ Add before public |
| **Polling Updates** | Simple, standard | Network overhead | ✅ Use, consider WebSockets |
| **Repository Snapshots** | Fast, cacheable | Stale data | ✅ Fits use case |

---

## Lessons Learned

### What Worked Well

1. ✅ **LlamaIndex Abstractions**: Saved weeks of development time
2. ✅ **State Machine Pattern**: Made debugging and error handling much easier
3. ✅ **Layered Architecture**: Easy to add features and modify components
4. ✅ **Gradio for MVP**: Instant UI with zero frontend code
5. ✅ **Docker Deployment**: Consistent environment, easy deployment

### What Could Be Improved

1. ⚠️ **Test Coverage**: Should have written tests from the start
2. ⚠️ **Job Persistence**: File-based storage has limitations
3. ⚠️ **Error Messages**: Could be more user-friendly
4. ⚠️ **Configuration Validation**: Better startup checks needed
5. ⚠️ **Documentation**: Should generate API docs automatically

### If Starting Over

1. 🔄 **Use SQLite from Day 1**: Minimal overhead, better than JSON files
2. 🔄 **Write Tests First**: TDD for critical paths
3. 🔄 **Add Observability Early**: Structured logging + metrics from start
4. 🔄 **Design for Multi-Tenancy**: Easier to add upfront than retrofit
5. ✅ **Keep**: Everything else was good decision

---

## Conclusion

The Onboarding Assistant was designed with **pragmatic trade-offs** prioritizing:

1. **Speed to Market**: MVP features over premature optimization
2. **Simplicity**: Minimal infrastructure requirements
3. **Flexibility**: Easy to extend and modify
4. **Cost-Effectiveness**: Optimize for low operational costs

The architecture provides a **solid foundation** that can scale from personal use to enterprise deployment with targeted enhancements.

**Next Steps for Production**:
1. Add authentication and rate limiting
2. Implement comprehensive test suite
3. Migrate to SQLite for job storage
4. Add monitoring and observability
5. Implement hybrid search for better quality

---

**For related documentation, see**:
- [README.md](README.md) - Complete project documentation
- [DEPENDENCIES.md](DEPENDENCIES.md) - Module dependency visualization
- [architecture.md](architecture.md) - Detailed technical architecture
