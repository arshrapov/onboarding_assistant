# Planned Improvements

This file outlines the next steps for improving the project quality, performance, and user experience.

## 1. Improving Chunking

* Review the current chunking strategy and identify weaknesses (e.g., oversized chunks, loss of context, poor boundary detection).
* Experiment with alternative chunking approaches (semantic chunking, syntax-aware chunking for code, adaptive chunk sizes).
* Tune chunk size and overlap to balance retrieval accuracy and performance.
* Add benchmarks to measure the impact of chunking changes on retrieval quality and response relevance.

## 2. Using Alternative Models for Embeddings

* Evaluate different embedding models to improve semantic search quality.
* Compare open-source and hosted models in terms of quality, latency, and cost.
* Make the embedding model configurable to allow easy experimentation and future upgrades.
* Rebuild or migrate existing embeddings if necessary and validate improvements with test queries.

## 3. Fixing and Improving the UI

* Address existing UI bugs and visual inconsistencies.
* Improve layout, responsiveness, and overall usability.
* Enhance feedback to users (loading states, error messages, empty states).
* Refactor UI components where needed to improve maintainability and readability of the codebase.

## 4. Separating Backend and Frontend

* Clearly separate backend and frontend into independent services or projects.
* Define a clean API contract (REST or GraphQL) between backend and frontend.
* Ensure the backend can run and be deployed independently of the frontend.
* Simplify frontend development by decoupling it from backend implementation details.
* Improve scalability, maintainability, and team collaboration by enforcing this separation.
