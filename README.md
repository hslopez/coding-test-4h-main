Multimodal Document Chat System – Technical Coding Assessment

1. Overview

This project implements a multimodal document intelligence system that allows users to upload PDF documents, extract text, images, and tables, and engage in context-aware, multi-turn chat enriched with relevant visual and tabular content.

2. Key Capabilities

• PDF processing using Docling
• Vector search with PostgreSQL + pgvector
• Multimodal chat (text, images, tables)
• Multi-turn conversational memory
• Fully Dockerized environment

3. Architecture Summary

Frontend (Next.js) communicates with a FastAPI backend. The backend processes documents via Docling, stores embeddings in PostgreSQL with pgvector, and performs retrieval-augmented generation (RAG) to produce multimodal answers.

4. Technology Stack

Backend:
- FastAPI
- SQLAlchemy
- PostgreSQL + pgvector
- Redis
- Docling
- OpenAI API / Local LLM (Ollama)

Frontend:
- Next.js 14
- TailwindCSS
- shadcn/ui

Infrastructure:
- Docker
- Docker Compose

5. Getting Started

Prerequisites:
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+

Steps:
1. Clone repository
2. Copy .env.example to .env
3. Run docker compose up --build

6. Evaluation Focus

• Code quality and modularity
• Document processing accuracy
• Vector search relevance
• Multimodal RAG chat quality
• Documentation clarity

7. Submission

Submit the GitHub repository containing full source code, Docker configuration, documentation, and sample PDF.
