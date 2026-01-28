"""
Training materials and RAG API routes.

Handles material upload, deletion, and RAG indexing.
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import MATERIALS_DIR
from app.web.dependencies import require_auth, require_write_auth
from app.web.utils import (
    ALLOWED_MATERIAL_EXTENSIONS,
    MAX_MATERIAL_SIZE,
    MAX_TOTAL_UPLOAD_SIZE,
    sanitize_filename,
    validate_upload_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/materials", tags=["materials"])


@router.post("", dependencies=[require_write_auth])
async def upload_materials(files: list[UploadFile] = File(...)):
    """Upload training materials (PDFs, text files)."""
    uploaded = 0
    errors = 0
    error_messages: list[str] = []
    total_size = 0

    MATERIALS_DIR.mkdir(exist_ok=True)

    for file in files:
        try:
            # Validate file type and size using our helper
            content = await validate_upload_file(
                file,
                ALLOWED_MATERIAL_EXTENSIONS,
                MAX_MATERIAL_SIZE,
                f"Material '{file.filename}'",
            )

            # Track total upload size
            total_size += len(content)
            if total_size > MAX_TOTAL_UPLOAD_SIZE:
                max_mb = MAX_TOTAL_UPLOAD_SIZE / (1024 * 1024)
                error_messages.append(f"Total upload size exceeds {max_mb:.0f} MB limit")
                errors += 1
                break

            # Sanitize filename and save
            filename = sanitize_filename(file.filename)
            file_path = MATERIALS_DIR / filename

            with open(file_path, "wb") as f:
                f.write(content)

            uploaded += 1

        except HTTPException as e:
            error_messages.append(str(e.detail))
            errors += 1
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
            error_messages.append(f"Error uploading {file.filename}: {str(e)}")
            errors += 1

    # Trigger RAG indexing in background
    if uploaded > 0:
        try:
            from app.materials_rag import get_materials_rag
            import asyncio

            async def index_async():
                rag = get_materials_rag()
                await asyncio.to_thread(rag.index_materials, True)

            asyncio.create_task(index_async())
            logger.info("📚 RAG indexing triggered after upload")
        except Exception as e:
            logger.warning(f"Could not trigger RAG indexing: {e}")

    response_data: dict = {
        "success": errors == 0,
        "uploaded": uploaded,
        "errors": errors,
        "message": f"Uploaded {uploaded} files" + (f" ({errors} errors)" if errors else ""),
    }
    if error_messages:
        response_data["error_details"] = error_messages

    return JSONResponse(response_data)


@router.get("/rag-status")
async def get_rag_status():
    """Get RAG indexing status."""
    try:
        from app.materials_rag import get_materials_rag

        rag = get_materials_rag()
        status = rag.get_status()
        return JSONResponse(status)
    except ImportError:
        return JSONResponse(
            {
                "available": False,
                "error": "RAG dependencies not installed. Run: pip install chromadb sentence-transformers",
            }
        )
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e)})


@router.post("/index", dependencies=[require_write_auth])
async def index_materials(force: bool = False):
    """Manually trigger RAG indexing of materials."""
    import asyncio

    try:
        from app.materials_rag import get_materials_rag

        rag = get_materials_rag()

        # Run indexing in thread to not block
        result = await asyncio.to_thread(rag.index_materials, force)
        return JSONResponse(result)
    except ImportError:
        return JSONResponse(
            {
                "error": "RAG dependencies not installed. Run: pip install chromadb sentence-transformers"
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)})


@router.delete("/{filename:path}", dependencies=[require_auth])
async def delete_material(filename: str):
    """Delete a training material file."""
    import urllib.parse

    filename = urllib.parse.unquote(filename)
    file_path = MATERIALS_DIR / filename

    if file_path.exists() and file_path.is_file():
        file_path.unlink()
        return JSONResponse({"message": f"Deleted {filename}"})

    raise HTTPException(status_code=404, detail="File not found")


@router.delete("", dependencies=[require_auth])
async def delete_all_materials():
    """Delete all training materials."""
    deleted = 0

    if MATERIALS_DIR.exists():
        for f in MATERIALS_DIR.iterdir():
            if f.is_file() and not f.name.startswith("."):
                f.unlink()
                deleted += 1

    return JSONResponse({"deleted": deleted, "message": f"Deleted {deleted} files"})
