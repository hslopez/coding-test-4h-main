
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation, Message
from app.models.document import DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from app.core.config import settings
import time
from openai import OpenAI

class ChatEngine:
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm = None  # TODO: Initialize LLM (OpenAI, Ollama, etc.)
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        # TODO: Implement message processing
        # 
        # Example LLM usage with OpenAI:
        # from openai import OpenAI
        # client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # 
        # response = client.chat.completions.create(
        #     model=settings.OPENAI_MODEL,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # 
        # Example with LangChain:
        # from langchain_openai import ChatOpenAI
        # from langchain.prompts import ChatPromptTemplate
        # 
        # llm = ChatOpenAI(model=settings.OPENAI_MODEL)
        # prompt = ChatPromptTemplate.from_messages([...])
        # chain = prompt | llm
        # response = chain.invoke({...})
        
        # raise NotImplementedError("Message processing not implemented yet")
        start = time.time()

        history = await self._load_conversation_history(conversation_id)
        context = await self._search_context(message, document_id)
        media = await self._find_related_media(context)

        answer = await self._generate_response(message, context, history, media)
        sources = self._format_sources(context, media)

        # self.db.add(Message(
        #    conversation_id=conversation_id,
        #    role="assistant",
        #    content=answer,
        #    sources=sources
        #))
        #self.db.commit()

        return {
            "answer": answer,
            "sources": sources,
            "processing_time": round(time.time() - start, 2)
        }
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        # raise NotImplementedError("History loading not implemented yet")
        msgs = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {"role": m.role, "content": m.content}
            for m in reversed(msgs)
        ]
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        #raise NotImplementedError("Context search not implemented yet")
        return await self.vector_store.similarity_search(query, document_id, k=k)


    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        #raise NotImplementedError("Related media finding not implemented yet")
        image_ids = set()
        table_ids = set()

        for c in context_chunks:
            image_ids.update(c["metadata"].get("related_images", []))
            table_ids.update(c["metadata"].get("related_tables", []))

        images = self.db.query(DocumentImage).filter(DocumentImage.id.in_(image_ids)).all()
        tables = self.db.query(DocumentTable).filter(DocumentTable.id.in_(table_ids)).all()

        return {
            "images": [
                {
                    "url": img.file_path.replace(settings.UPLOAD_DIR, "/uploads"),
                    "caption": img.caption,
                    "page": img.page_number
                }
                for img in images
            ],
            "tables": [
                {
                    "url": tbl.image_path.replace(settings.UPLOAD_DIR, "/uploads"),
                    "caption": tbl.caption,
                    "page": tbl.page_number,
                    "data": tbl.data
                }
                for tbl in tables
            ]
        }
    
    async def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        #raise NotImplementedError("Response generation not implemented yet")
        system = (
            "You are a document-grounded assistant. "
            "Answer strictly using provided context. "
            "If the answer is not in the document, say so."
        )

        context_text = "\n\n".join(
            f"(Page {c['page_number']}) {c['content']}"
            for c in context
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion:\n{message}"
        })

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages
        )

        return response.choices[0].message.content.strip()
    
    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        sources = []
        
        # Add text sources
        for chunk in context[:3]:  # Top 3 text chunks
            sources.append({
                "type": "text",
                "content": chunk["content"],
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0.0)
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image["url"],
                "caption": image.get("caption"),
                "page": image.get("page")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table["url"],
                "caption": table.get("caption"),
                "page": table.get("page"),
                "data": table.get("data")
            })
        
        return sources
