"""
Materials Reader - Reads uploaded training materials for LLM context.

Supports PDF and text files from the materials/ directory.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Materials directory
MATERIALS_DIR = Path(__file__).parent.parent / "materials"


def get_materials_context(max_chars: int = 50000) -> str:
    """
    Read all materials from the materials/ directory and return as a single context string.
    
    Args:
        max_chars: Maximum total characters to return (to avoid token limits)
        
    Returns:
        Combined text from all materials, or empty string if no materials
    """
    if not MATERIALS_DIR.exists():
        return ""
    
    materials_text = []
    total_chars = 0
    
    # Get all supported files - prioritize text files over PDFs (more likely to be focused rules)
    all_files = list(MATERIALS_DIR.glob("*"))
    text_files = [f for f in all_files if f.suffix.lower() in [".txt", ".md", ".markdown"]]
    pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
    
    # Process text files first (user's custom rules), then PDFs
    files = sorted(text_files) + sorted(pdf_files)
    
    for file_path in files:
        if total_chars >= max_chars:
            break
            
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                content = _read_pdf(file_path)
            elif suffix in [".txt", ".md", ".markdown"]:
                content = _read_text(file_path)
            else:
                continue  # Skip unsupported files
            
            if content:
                # Add section header
                section = f"\n\n=== TRAINING MATERIAL: {file_path.name} ===\n{content}"
                
                # Truncate if needed
                remaining = max_chars - total_chars
                if len(section) > remaining:
                    if remaining > 1000:
                        section = section[:remaining] + "\n[... truncated due to length ...]"
                        logger.warning(f"📚 Material '{file_path.name}' truncated ({len(content):,} chars -> {remaining:,} chars)")
                    else:
                        logger.info(f"📚 Skipping '{file_path.name}' - character limit reached")
                        continue
                
                materials_text.append(section)
                total_chars += len(section)
                logger.debug(f"📚 Loaded '{file_path.name}': {len(content):,} chars")
                
        except Exception as e:
            logger.warning(f"Failed to read material {file_path.name}: {e}")
    
    if materials_text:
        header = "=== YOUR TRAINING MATERIALS (use this knowledge to inform your analysis) ===\n"
        return header + "".join(materials_text)
    
    return ""


def _read_pdf(file_path: Path) -> Optional[str]:
    """Read text content from a PDF file using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        return "\n\n".join(text_parts)
        
    except ImportError:
        logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")
        return None
    except Exception as e:
        logger.warning(f"Failed to read PDF {file_path}: {e}")
        return None


def _read_text(file_path: Path) -> Optional[str]:
    """Read content from a text file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding="latin-1")
        except Exception as e:
            logger.warning(f"Failed to read text file {file_path}: {e}")
            return None
    except Exception as e:
        logger.warning(f"Failed to read text file {file_path}: {e}")
        return None


def get_materials_summary() -> dict:
    """
    Get a summary of available materials.
    
    Returns:
        dict with 'count', 'files', and 'total_size_kb'
    """
    if not MATERIALS_DIR.exists():
        return {"count": 0, "files": [], "total_size_kb": 0}
    
    files = []
    total_size = 0
    
    for file_path in sorted(MATERIALS_DIR.glob("*")):
        if file_path.suffix.lower() in [".pdf", ".txt", ".md", ".markdown"]:
            size = file_path.stat().st_size
            files.append({
                "name": file_path.name,
                "size_kb": round(size / 1024, 1),
                "type": file_path.suffix.lower()
            })
            total_size += size
    
    return {
        "count": len(files),
        "files": files,
        "total_size_kb": round(total_size / 1024, 1)
    }


def has_materials() -> bool:
    """Check if any training materials are available."""
    if not MATERIALS_DIR.exists():
        return False
    
    for file_path in MATERIALS_DIR.glob("*"):
        if file_path.suffix.lower() in [".pdf", ".txt", ".md", ".markdown"]:
            return True
    
    return False
