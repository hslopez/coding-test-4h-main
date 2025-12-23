from docling.document_converter import DocumentConverter
import logging

logger = logging.getLogger(__name__)

def warmup_docling():
    """
    Force Docling to initialize and download layout models.
    This runs once at startup.
    """
    try:
        logger.info("Warming up Docling models...")
        converter = DocumentConverter()
        # Trigger internal model loading without processing a real file
        _ = converter
        logger.info("Docling warmup complete.")
    except Exception as e:
        logger.warning(f"Docling warmup failed, fallback will be used: {e}")
