"""
RAG (Retrieval-Augmented Generation) for Training Materials.

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Automatically indexes uploaded materials and retrieves relevant chunks per trade.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Directories
MATERIALS_DIR = Path(__file__).parent.parent / "materials"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
INDEX_META_FILE = CHROMA_DIR / "index_meta.json"

# Chunking settings
CHUNK_SIZE = 1000  # ~250 tokens per chunk
CHUNK_OVERLAP = 200  # Overlap for context continuity
MIN_CHUNK_SIZE = 100  # Skip very small chunks


class MaterialsRAG:
    """
    RAG system for training materials.
    
    Indexes PDFs and text files into a vector database,
    then retrieves relevant chunks based on trade context.
    """
    
    def __init__(self):
        self._chroma_client = None
        self._collection = None
        self._embedding_model = None
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialization of ChromaDB and embedding model."""
        if self._initialized:
            return True
            
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create chroma directory if needed
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent ChromaDB client
            self._chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self._collection = self._chroma_client.get_or_create_collection(
                name="trading_materials",
                metadata={"hnsw:space": "cosine"}
            )
            
            self._initialized = True
            logger.info("📚 RAG system initialized with ChromaDB")
            return True
            
        except ImportError as e:
            logger.warning(f"RAG dependencies not installed: {e}. Run: pip install chromadb sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model optimized for semantic search
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("📚 Loaded embedding model: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
                return None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return None
        return self._embedding_model
    
    def _compute_materials_hash(self) -> str:
        """Compute hash of all materials to detect changes."""
        if not MATERIALS_DIR.exists():
            return ""
        
        hasher = hashlib.md5()
        for file_path in sorted(MATERIALS_DIR.glob("*")):
            if file_path.suffix.lower() in [".pdf", ".txt", ".md", ".markdown"]:
                hasher.update(file_path.name.encode())
                hasher.update(str(file_path.stat().st_mtime).encode())
                hasher.update(str(file_path.stat().st_size).encode())
        
        return hasher.hexdigest()
    
    def _load_index_meta(self) -> dict:
        """Load index metadata."""
        if INDEX_META_FILE.exists():
            try:
                return json.loads(INDEX_META_FILE.read_text())
            except (json.JSONDecodeError, IOError, OSError):
                pass
        return {}
    
    def _save_index_meta(self, meta: dict):
        """Save index metadata."""
        INDEX_META_FILE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_META_FILE.write_text(json.dumps(meta, indent=2))
    
    def needs_reindex(self) -> bool:
        """Check if materials need to be re-indexed."""
        current_hash = self._compute_materials_hash()
        if not current_hash:
            return False
        
        meta = self._load_index_meta()
        return meta.get("materials_hash") != current_hash
    
    def index_materials(self, force: bool = False) -> dict:
        """
        Index all materials into the vector database.
        
        Args:
            force: Re-index even if materials haven't changed
            
        Returns:
            dict with indexing stats
        """
        if not self._ensure_initialized():
            return {"error": "RAG system not available"}
        
        # Check if reindex needed
        current_hash = self._compute_materials_hash()
        if not force and not self.needs_reindex():
            meta = self._load_index_meta()
            return {
                "status": "up_to_date",
                "indexed_at": meta.get("indexed_at"),
                "chunks": meta.get("total_chunks", 0),
                "files": meta.get("files", [])
            }
        
        # Clear existing collection
        try:
            self._chroma_client.delete_collection("trading_materials")
            self._collection = self._chroma_client.create_collection(
                name="trading_materials",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass  # Collection may not exist, which is fine
        
        # Get embedding model
        model = self._get_embedding_model()
        if model is None:
            return {"error": "Embedding model not available"}
        
        # Process all materials
        all_chunks = []
        all_ids = []
        all_metadatas = []
        files_indexed = []
        
        if not MATERIALS_DIR.exists():
            return {"error": "Materials directory not found"}
        
        for file_path in sorted(MATERIALS_DIR.glob("*")):
            suffix = file_path.suffix.lower()
            if suffix not in [".pdf", ".txt", ".md", ".markdown"]:
                continue
            
            logger.debug(f"📚 Indexing: {file_path.name}")
            
            # Extract text
            if suffix == ".pdf":
                text = self._extract_pdf_text(file_path)
            else:
                text = self._extract_text_file(file_path)
            
            if not text:
                continue
            
            # Chunk the text
            chunks = self._chunk_text(text, file_path.name)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_{i}"
                all_chunks.append(chunk["text"])
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "source": file_path.name,
                    "chunk_index": i,
                    "section": chunk.get("section", ""),
                })
            
            files_indexed.append({
                "name": file_path.name,
                "chunks": len(chunks),
                "size_kb": round(file_path.stat().st_size / 1024, 1)
            })
            logger.debug(f"📚 Created {len(chunks)} chunks from {file_path.name}")
        
        if not all_chunks:
            return {"error": "No content to index"}
        
        # Create embeddings and add to collection
        logger.debug(f"📚 Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self._collection.add(
                ids=all_ids[i:end],
                embeddings=embeddings[i:end],
                documents=all_chunks[i:end],
                metadatas=all_metadatas[i:end]
            )
        
        # Save metadata
        meta = {
            "materials_hash": current_hash,
            "indexed_at": datetime.now().isoformat(),
            "total_chunks": len(all_chunks),
            "files": files_indexed
        }
        self._save_index_meta(meta)
        
        logger.info(f"📚 Indexed {len(all_chunks)} chunks from {len(files_indexed)} files")
        
        return {
            "status": "indexed",
            "chunks": len(all_chunks),
            "files": files_indexed
        }
    
    def _extract_pdf_text(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            text_parts = []
            
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.warning(f"Failed to extract PDF {file_path.name}: {e}")
            return None
    
    def _extract_text_file(self, file_path: Path) -> Optional[str]:
        """Extract text from text file."""
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding="latin-1")
            except (UnicodeDecodeError, IOError, OSError):
                return None
        except Exception as e:
            logger.warning(f"Failed to read {file_path.name}: {e}")
            return None
    
    def _chunk_text(self, text: str, source_name: str) -> list[dict]:
        """
        Split text into overlapping chunks.
        
        Tries to split on paragraph/section boundaries when possible.
        """
        chunks = []
        
        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        
        # Try to split on sections/headers first
        sections = re.split(r'\n(?=[A-Z][A-Z\s]{5,}(?:\n|:))', text)
        
        current_chunk = ""
        current_section = ""
        
        for section in sections:
            # Check if this looks like a header
            lines = section.strip().split('\n')
            if lines and len(lines[0]) < 100 and lines[0].isupper():
                current_section = lines[0][:50]
            
            paragraphs = section.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < MIN_CHUNK_SIZE:
                    continue
                
                # If adding this paragraph exceeds chunk size, save current and start new
                if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": current_section
                    })
                    # Keep overlap from end of previous chunk
                    overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append({
                "text": current_chunk.strip(),
                "section": current_section
            })
        
        return chunks
    
    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query (trade context)
            n_results: Number of chunks to return
            
        Returns:
            List of dicts with 'text', 'source', 'score'
        """
        if not self._ensure_initialized():
            return []
        
        # Check if we have indexed content
        if self._collection.count() == 0:
            # Try to index if materials exist
            if MATERIALS_DIR.exists() and any(MATERIALS_DIR.glob("*")):
                self.index_materials()
            
            if self._collection.count() == 0:
                return []
        
        # Get embedding model
        model = self._get_embedding_model()
        if model is None:
            return []
        
        # Create query embedding
        query_embedding = model.encode([query])[0].tolist()
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results["documents"]:
            return []
        
        # Format results
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            
            # Convert distance to similarity score (cosine distance -> similarity)
            score = 1 - distance
            
            chunks.append({
                "text": doc,
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", ""),
                "score": round(score, 3)
            })
        
        return chunks
    
    def get_trade_context(
        self,
        ticker: str,
        direction: str,
        timeframe: str = "5m",
        entry_reason: str = "",
        setup_type: str = "",
        market_context: str = "",
        n_chunks: int = 5,
        max_chars: int = 8000
    ) -> str:
        """
        Get relevant training material context for a trade analysis.
        
        Builds a smart query from trade details and retrieves relevant chunks.
        
        Args:
            ticker: Stock symbol
            direction: long/short
            timeframe: Trading timeframe
            entry_reason: Why the trade was entered
            setup_type: Type of setup if known
            market_context: Current market regime/context
            n_chunks: Number of chunks to retrieve
            max_chars: Maximum characters to return
            
        Returns:
            Formatted string with relevant material excerpts
        """
        # Build a query that captures the trade context
        query_parts = []
        
        # Direction-based context
        if direction.lower() == "long":
            query_parts.append("long trade buy entry bullish setup")
        else:
            query_parts.append("short trade sell entry bearish setup")
        
        # Timeframe context
        if timeframe in ["1d", "daily"]:
            query_parts.append("daily chart swing trade position")
        elif timeframe in ["2h", "1h"]:
            query_parts.append("swing trade intraday hourly")
        else:
            query_parts.append("scalp day trade intraday 5-minute")
        
        # Add entry reason if provided
        if entry_reason:
            query_parts.append(entry_reason)
        
        # Add setup type
        if setup_type:
            query_parts.append(setup_type)
        
        # Add market context
        if market_context:
            query_parts.append(market_context)
        
        # Combine into query
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
        for i, chunk in enumerate(chunks):
            if total_chars >= max_chars:
                break
            
            chunk_text = chunk["text"]
            remaining = max_chars - total_chars
            
            if len(chunk_text) > remaining:
                chunk_text = chunk_text[:remaining] + "..."
            
            source_info = f"[From: {chunk['source']}"
            if chunk.get("section"):
                source_info += f" - {chunk['section']}"
            source_info += f" | Relevance: {chunk['score']:.0%}]"
            
            lines.append(source_info)
            lines.append(chunk_text)
            lines.append("")
            
            total_chars += len(chunk_text) + len(source_info) + 2
        
        return "\n".join(lines)
    
    def get_status(self) -> dict:
        """Get RAG system status."""
        meta = self._load_index_meta()
        
        status = {
            "available": self._ensure_initialized(),
            "indexed": bool(meta.get("indexed_at")),
            "indexed_at": meta.get("indexed_at"),
            "total_chunks": meta.get("total_chunks", 0),
            "files": meta.get("files", []),
            "needs_reindex": self.needs_reindex()
        }
        
        return status


# Singleton instance
_rag_instance: Optional[MaterialsRAG] = None


def get_materials_rag() -> MaterialsRAG:
    """Get the global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MaterialsRAG()
    return _rag_instance


def get_relevant_materials(
    ticker: str,
    direction: str,
    timeframe: str = "5m",
    entry_reason: str = "",
    setup_type: str = "",
    market_context: str = "",
    n_chunks: int = 5,
    max_chars: int = 8000
) -> str:
    """
    Convenience function to get relevant materials for a trade.
    
    Returns empty string if RAG is not available or no materials indexed.
    """
    try:
        rag = get_materials_rag()
        return rag.get_trade_context(
            ticker=ticker,
            direction=direction,
            timeframe=timeframe,
            entry_reason=entry_reason,
            setup_type=setup_type,
            market_context=market_context,
            n_chunks=n_chunks,
            max_chars=max_chars
        )
    except Exception as e:
        logger.warning(f"Failed to get relevant materials: {e}")
        return ""
