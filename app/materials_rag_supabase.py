"""
RAG (Retrieval-Augmented Generation) for Training Materials using Supabase.

Uses Supabase Storage for file storage and pgvector for embeddings.
Each user has their own materials and embeddings (per-user isolation).
"""

import hashlib
import io
import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Chunking settings (same as local RAG)
CHUNK_SIZE = 1000  # ~250 tokens per chunk
CHUNK_OVERLAP = 200  # Overlap for context continuity
MIN_CHUNK_SIZE = 100  # Skip very small chunks


class SupabaseMaterialsRAG:
    """
    RAG system using Supabase for storage and pgvector for embeddings.
    
    Each user has isolated materials and embeddings.
    """

    def __init__(self, supabase_client, user_id: str):
        """
        Initialize RAG for a specific user.
        
        Args:
            supabase_client: Supabase client (service or authenticated)
            user_id: User's UUID
        """
        self.supabase = supabase_client
        self.user_id = user_id
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Use same model as local RAG
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded embedding model: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return None
        return self._embedding_model

    def _compute_materials_hash(self) -> str:
        """Compute hash of user's materials to detect changes."""
        # Get materials metadata from database
        result = self.supabase.table("user_materials").select(
            "filename,file_size,created_at"
        ).eq("user_id", self.user_id).order("filename").execute()
        
        if not result.data:
            return ""
        
        hasher = hashlib.md5()
        for m in result.data:
            hasher.update(m["filename"].encode())
            hasher.update(str(m["file_size"]).encode())
            hasher.update(str(m["created_at"]).encode())
        
        return hasher.hexdigest()

    def needs_reindex(self) -> bool:
        """Check if materials need to be re-indexed."""
        # Check if any materials exist without embeddings
        result = self.supabase.table("user_materials").select(
            "id"
        ).eq("user_id", self.user_id).is_("indexed_at", "null").execute()
        
        return len(result.data or []) > 0

    def index_materials(self, force: bool = False) -> dict:
        """
        Index user's materials into pgvector.

        Args:
            force: Re-index even if materials haven't changed

        Returns:
            dict with indexing stats
        """
        # Get embedding model
        model = self._get_embedding_model()
        if model is None:
            return {"error": "Embedding model not available"}

        # Get materials to index
        if force:
            # Get all materials
            mat_result = self.supabase.table("user_materials").select(
                "*"
            ).eq("user_id", self.user_id).execute()
            
            # Clear existing embeddings
            self.supabase.table("embeddings").delete().eq(
                "user_id", self.user_id
            ).execute()
        else:
            # Get only non-indexed materials
            mat_result = self.supabase.table("user_materials").select(
                "*"
            ).eq("user_id", self.user_id).is_("indexed_at", "null").execute()

        materials = mat_result.data or []
        
        if not materials:
            if force:
                return {"status": "no_materials", "message": "No materials to index"}
            return {"status": "up_to_date", "message": "All materials already indexed"}

        # Process each material
        files_indexed = []
        total_chunks = 0
        
        from app.db.supabase_client import SupabaseStorage
        storage = SupabaseStorage(self.supabase, "materials")

        for material in materials:
            try:
                logger.debug(f"Indexing: {material['filename']}")
                
                # Download file
                file_data = storage.download_file(material["storage_path"])
                
                # Extract text
                ext = material["filename"].rsplit(".", 1)[-1].lower() if "." in material["filename"] else ""
                
                if ext == "pdf":
                    text = self._extract_pdf_text(file_data)
                else:
                    text = self._extract_text(file_data)
                
                if not text:
                    logger.warning(f"No text extracted from {material['filename']}")
                    continue
                
                # Chunk the text
                chunks = self._chunk_text(text, material["filename"])
                
                if not chunks:
                    continue
                
                # Create embeddings
                chunk_texts = [c["text"] for c in chunks]
                embeddings = model.encode(chunk_texts, show_progress_bar=False)
                
                # Delete existing embeddings for this material
                self.supabase.table("embeddings").delete().eq(
                    "material_id", material["id"]
                ).execute()
                
                # Insert new embeddings
                rows = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    rows.append({
                        "user_id": self.user_id,
                        "material_id": material["id"],
                        "chunk_index": i,
                        "chunk_text": chunk["text"],
                        "section": chunk.get("section", ""),
                        "embedding": embedding.tolist(),
                    })
                
                # Batch insert
                batch_size = 50
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    self.supabase.table("embeddings").insert(batch).execute()
                
                # Update material record
                self.supabase.table("user_materials").update({
                    "indexed_at": datetime.utcnow().isoformat(),
                    "chunk_count": len(chunks),
                }).eq("id", material["id"]).execute()
                
                files_indexed.append({
                    "name": material["filename"],
                    "chunks": len(chunks),
                })
                total_chunks += len(chunks)
                
                logger.debug(f"Created {len(chunks)} chunks from {material['filename']}")
                
            except Exception as e:
                logger.error(f"Failed to index {material['filename']}: {e}")

        logger.info(f"Indexed {total_chunks} chunks from {len(files_indexed)} files")

        return {
            "status": "indexed",
            "chunks": total_chunks,
            "files": files_indexed,
        }

    def _extract_pdf_text(self, file_data: bytes) -> Optional[str]:
        """Extract text from PDF bytes."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=file_data, filetype="pdf")
            text_parts = []

            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.warning(f"Failed to extract PDF: {e}")
            return None

    def _extract_text(self, file_data: bytes) -> Optional[str]:
        """Extract text from text file bytes."""
        try:
            return file_data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_data.decode("latin-1")
            except Exception:
                return None
        except Exception as e:
            logger.warning(f"Failed to extract text: {e}")
            return None

    def _chunk_text(self, text: str, source_name: str) -> list[dict]:
        """Split text into overlapping chunks."""
        chunks = []

        # Clean text
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Split on sections/headers
        sections = re.split(r"\n(?=[A-Z][A-Z\s]{5,}(?:\n|:))", text)

        current_chunk = ""
        current_section = ""

        for section in sections:
            lines = section.strip().split("\n")
            if lines and len(lines[0]) < 100 and lines[0].isupper():
                current_section = lines[0][:50]

            paragraphs = section.split("\n\n")

            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < MIN_CHUNK_SIZE:
                    continue

                if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                    chunks.append({"text": current_chunk.strip(), "section": current_section})
                    overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

        if current_chunk and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append({"text": current_chunk.strip(), "section": current_section})

        return chunks

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Retrieve relevant chunks using pgvector similarity search.

        Args:
            query: The search query
            n_results: Number of chunks to return

        Returns:
            List of dicts with 'text', 'source', 'score'
        """
        model = self._get_embedding_model()
        if model is None:
            return []

        # Check if we have any embeddings
        count_result = self.supabase.table("embeddings").select(
            "id", count="exact"
        ).eq("user_id", self.user_id).limit(1).execute()
        
        if not count_result.count:
            return []

        # Create query embedding
        query_embedding = model.encode([query])[0].tolist()

        # Use pgvector similarity search via RPC function
        try:
            result = self.supabase.rpc(
                "match_embeddings",
                {
                    "query_embedding": query_embedding,
                    "match_count": n_results,
                    "match_user_id": self.user_id,
                }
            ).execute()

            if not result.data:
                return []

            chunks = []
            for row in result.data:
                chunks.append({
                    "text": row["chunk_text"],
                    "section": row.get("section", ""),
                    "source": "materials",  # Could be enhanced with material filename
                    "score": round(row["similarity"], 3),
                })

            return chunks

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            
            # Fallback to basic text search
            return self._fallback_search(query, n_results)

    def _fallback_search(self, query: str, n_results: int) -> list[dict]:
        """Fallback text search if vector search fails."""
        try:
            # Simple text search using PostgreSQL
            result = self.supabase.table("embeddings").select(
                "chunk_text,section"
            ).eq("user_id", self.user_id).ilike(
                "chunk_text", f"%{query}%"
            ).limit(n_results).execute()

            return [
                {
                    "text": row["chunk_text"],
                    "section": row.get("section", ""),
                    "source": "materials",
                    "score": 0.5,  # Placeholder score
                }
                for row in result.data or []
            ]
        except Exception:
            return []

    def get_trade_context(
        self,
        ticker: str,
        direction: str,
        timeframe: str = "5m",
        entry_reason: str = "",
        setup_type: str = "",
        market_context: str = "",
        n_chunks: int = 5,
        max_chars: int = 8000,
    ) -> str:
        """
        Get relevant training material context for a trade analysis.
        """
        # Build query from trade details
        query_parts = []

        if direction.lower() == "long":
            query_parts.append("long trade buy entry bullish setup")
        else:
            query_parts.append("short trade sell entry bearish setup")

        if timeframe in ["1d", "daily"]:
            query_parts.append("daily chart swing trade position")
        elif timeframe in ["2h", "1h"]:
            query_parts.append("swing trade intraday hourly")
        else:
            query_parts.append("scalp day trade intraday 5-minute")

        if entry_reason:
            query_parts.append(entry_reason)
        if setup_type:
            query_parts.append(setup_type)
        if market_context:
            query_parts.append(market_context)

        query = " ".join(query_parts)

        # Retrieve relevant chunks
        chunks = self.retrieve(query, n_results=n_chunks)

        if not chunks:
            return ""

        # Format for LLM
        lines = ["=== RELEVANT TRAINING MATERIALS ==="]
        lines.append(f"(Retrieved based on: {direction} {timeframe} trade)")
        lines.append("")

        total_chars = 0
        for chunk in chunks:
            if total_chars >= max_chars:
                break

            chunk_text = chunk["text"]
            remaining = max_chars - total_chars

            if len(chunk_text) > remaining:
                chunk_text = chunk_text[:remaining] + "..."

            source_info = f"[Relevance: {chunk['score']:.0%}]"
            if chunk.get("section"):
                source_info = f"[{chunk['section']} | Relevance: {chunk['score']:.0%}]"

            lines.append(source_info)
            lines.append(chunk_text)
            lines.append("")

            total_chars += len(chunk_text) + len(source_info) + 2

        return "\n".join(lines)

    def get_status(self) -> dict:
        """Get RAG system status for this user."""
        try:
            # Count embeddings
            embed_result = self.supabase.table("embeddings").select(
                "id", count="exact"
            ).eq("user_id", self.user_id).execute()
            
            # Count materials
            mat_result = self.supabase.table("user_materials").select(
                "id,filename,chunk_count,indexed_at"
            ).eq("user_id", self.user_id).execute()
            
            files = []
            for m in mat_result.data or []:
                files.append({
                    "name": m["filename"],
                    "chunks": m["chunk_count"] or 0,
                    "indexed": m["indexed_at"] is not None,
                })

            return {
                "available": True,
                "indexed": (embed_result.count or 0) > 0,
                "total_chunks": embed_result.count or 0,
                "files": files,
                "needs_reindex": self.needs_reindex(),
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }


# Factory function
def get_user_materials_rag(supabase_client, user_id: str) -> SupabaseMaterialsRAG:
    """Get RAG instance for a specific user."""
    return SupabaseMaterialsRAG(supabase_client, user_id)


def get_relevant_materials_supabase(
    supabase_client,
    user_id: str,
    ticker: str,
    direction: str,
    timeframe: str = "5m",
    entry_reason: str = "",
    setup_type: str = "",
    market_context: str = "",
    n_chunks: int = 5,
    max_chars: int = 8000,
) -> str:
    """
    Convenience function to get relevant materials for a trade.
    """
    try:
        rag = get_user_materials_rag(supabase_client, user_id)
        return rag.get_trade_context(
            ticker=ticker,
            direction=direction,
            timeframe=timeframe,
            entry_reason=entry_reason,
            setup_type=setup_type,
            market_context=market_context,
            n_chunks=n_chunks,
            max_chars=max_chars,
        )
    except Exception as e:
        logger.warning(f"Failed to get relevant materials: {e}")
        return ""
