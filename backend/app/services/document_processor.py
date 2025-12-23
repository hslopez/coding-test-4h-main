"""
Document processing service using Docling (when available) with a pypdf fallback.

In Docker, Docling may fail to initialize due to missing HF model artifacts
(e.g., beehive layout model). In that case we fall back to pypdf text extraction
so the pipeline still works.
"""
import io
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore


class DocumentProcessor:
    logger = logging.getLogger(__name__)

    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)

    # ---------- Fallback text extraction ----------
    @staticmethod
    def _extract_text_pypdf(file_path: str) -> Tuple[str, int]:
        """Return (text, total_pages) extracted via pypdf."""
        reader = PdfReader(file_path)
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip(), len(reader.pages)

    # ---------- Docling extraction ----------
    def _try_extract_with_docling(self, file_path: str) -> Tuple[Optional[Any], Optional[str], Optional[int], Optional[Exception]]:
        """
        Try to convert with Docling.
        Returns (result, markdown_text, total_pages, error).
        """
        try:
            # Import locally so Docker can still run if docling isn't usable
            from docling.document_converter import DocumentConverter  # type: ignore

            converter = DocumentConverter()
            result = converter.convert(file_path)

            # Prefer markdown if available
            text: Optional[str] = None
            if hasattr(result, "document") and hasattr(result.document, "export_to_markdown"):
                text = result.document.export_to_markdown()

            total_pages: Optional[int] = None
            if hasattr(result, "pages") and result.pages is not None:
                try:
                    total_pages = len(result.pages)
                except Exception:
                    total_pages = None

            return result, text, total_pages, None
        except Exception as e:
            return None, None, None, e

    # ---------- Main pipeline ----------
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        start = time.time()

        try:
            document = self.db.query(Document).get(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")

            document.processing_status = "processing"
            self.db.commit()

            # Try Docling first
            result, md_text, total_pages, docling_err = self._try_extract_with_docling(file_path)

            text_chunks: List[Dict[str, Any]] = []
            images: List[DocumentImage] = []
            tables: List[DocumentTable] = []

            if result is not None and docling_err is None:
                # ---------- TEXT (Docling) ----------
                # Best-effort: per-page text if supported, otherwise use markdown as one chunk
                if hasattr(result, "document") and hasattr(result, "pages") and result.pages is not None:
                    doc = result.document
                    total_pages = total_pages or 0

                    for page_number in range(1, total_pages + 1):
                        try:
                            page_doc = doc.filter(page_nrs={page_number})
                            page_text = (page_doc.export_to_text() or "").strip()
                        except Exception:
                            page_text = ""

                        if page_text:
                            text_chunks.extend(self._chunk_text(page_text, document_id, page_number))

                # Fallback: if per-page export failed, use markdown/text once
                if not text_chunks:
                    base_text = (md_text or "").strip()
                    if not base_text and hasattr(result, "document") and hasattr(result.document, "export_to_text"):
                        base_text = (result.document.export_to_text() or "").strip()

                    if base_text:
                        # If we don't know pages, store as page 1
                        text_chunks.extend(self._chunk_text(base_text, document_id, 1))

                # ---------- IMAGES (Docling) ----------
                for img_obj in (getattr(result, "images", None) or []):
                    try:
                        images.append(
                            await self._save_image(
                                img_obj.image,
                                document_id,
                                getattr(img_obj, "page_number", 1),
                                getattr(img_obj, "metadata", None) or {},
                            )
                        )
                    except Exception as e:
                        self.logger.warning("Failed saving image: %s", e)

                # ---------- TABLES (Docling) ----------
                for tbl_obj in (getattr(result, "tables", None) or []):
                    try:
                        tables.append(
                            await self._save_table(
                                tbl_obj,
                                document_id,
                                getattr(tbl_obj, "page_number", 1),
                                getattr(tbl_obj, "metadata", None) or {},
                            )
                        )
                    except Exception as e:
                        self.logger.warning("Failed saving table: %s", e)

            else:
                # ---------- FALLBACK (pypdf text-only) ----------
                self.logger.warning("Docling failed in this environment; using pypdf fallback. Error=%r", docling_err)

                text, total_pages_fallback = self._extract_text_pypdf(file_path)
                total_pages = total_pages or total_pages_fallback

                if text:
                    # Store all text as page 1 (or you can split by page later if needed)
                    text_chunks = self._chunk_text(text, document_id, 1)

            # ---------- STORE CHUNKS (embeddings + metadata) ----------
            for idx, chunk in enumerate(text_chunks):
                await self.vector_store.store_chunk(
                    content=chunk["content"],
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    chunk_index=idx,
                    metadata={
                        "related_images": [img.id for img in images if img.page_number == chunk["page_number"]],
                        "related_tables": [tbl.id for tbl in tables if tbl.page_number == chunk["page_number"]],
                    },
                )

            document.processing_status = "completed"
            document.text_chunks_count = len(text_chunks)
            document.images_count = len(images)
            document.tables_count = len(tables)
            document.total_pages = int(total_pages or 0)
            self.db.commit()

            return {
                "status": "success",
                "text_chunks": len(text_chunks),
                "images": len(images),
                "tables": len(tables),
                "processing_time": round(time.time() - start, 2),
            }

        except Exception as e:
            print("🔥 process_document failed:", repr(e))
            import traceback
            print(traceback.format_exc())
            await self._update_document_status(document_id, "error", str(e))
            raise

    # ---------- Helpers ----------
    def _chunk_text(self, text: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
        size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        chunks: List[Dict[str, Any]] = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append({"content": text[start:end], "page_number": page_number})
            start += max(1, size - overlap)

        return chunks

    async def _save_image(
        self,
        image_data: bytes,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any],
    ) -> DocumentImage:
        image = Image.open(io.BytesIO(image_data))

        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(settings.UPLOAD_DIR, "images", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        image.save(path)

        record = DocumentImage(
            document_id=document_id,
            file_path=path,
            page_number=page_number,
            width=image.width,
            height=image.height,
            caption=metadata.get("caption"),
            meta=metadata,  # your model uses "meta"
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    async def _save_table(
        self,
        table_data: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any],
    ) -> DocumentTable:
        img = table_data.to_image()

        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(settings.UPLOAD_DIR, "tables", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        img.save(path)

        data = getattr(table_data, "data", None)
        rows = len(data) if data else 0
        cols = len(data[0]) if data and len(data) > 0 else 0

        record = DocumentTable(
            document_id=document_id,
            image_path=path,
            data=data,
            page_number=page_number,
            rows=rows,
            columns=cols,
            caption=metadata.get("caption"),
            meta=metadata,  # your model uses "meta"
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    async def _update_document_status(self, document_id: int, status: str, error_message: str = None):
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            if error_message:
                document.error_message = error_message
            self.db.commit()
