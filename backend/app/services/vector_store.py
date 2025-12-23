from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.document import DocumentChunk  # adjust if your model name differs
from sqlalchemy import Float
from typing import List

class VectorStore:
    def __init__(self, db: Session):
        self.db = db

        # ✅ REQUIRED: create the OpenAI client
        # settings.OPENAI_API_KEY must exist and be loaded (.env)
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # optional: configure model name in settings
        self.embedding_model = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")

    async def similarity_search(self, query: str, document_id: Optional[int] = None, k: int = 5):
        q_emb = await self.generate_embedding(query)

        distance_expr = DocumentChunk.embedding.op("<->")(q_emb).cast(Float).label("distance")

        q = self.db.query(DocumentChunk, distance_expr)

        if document_id is not None:
            q = q.filter(DocumentChunk.document_id == document_id)

        rows = q.order_by(distance_expr.asc()).limit(k).all()

        results = []
        for chunk, distance in rows:
            results.append({
                "content": chunk.content,
                "page_number": getattr(chunk, "page_number", None),
                "metadata": getattr(chunk, "meta", {}) or {},
                "score": float(distance) if distance is not None else 0.0
            })

        return results

    async def generate_embedding(self, text: str):
        # ✅ use async client
        resp = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    async def store_chunk(
        self,
        content: str,
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        embedding = await self.generate_embedding(content)

        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            page_number=page_number,
            chunk_index=chunk_index,
            embedding=embedding,
            meta=metadata or {},  # or metadata=... depending on your model column
        )

        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk
