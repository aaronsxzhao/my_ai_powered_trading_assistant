"""
Training materials and RAG API routes.

Supports both local storage (development) and Supabase Storage (production).
Materials are per-user when using Supabase.
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse

from app.config import MATERIALS_DIR
from app.web.dependencies import require_auth, require_write_auth, get_current_user, get_user_id
from app.web.utils import (
    ALLOWED_MATERIAL_EXTENSIONS,
    MAX_MATERIAL_SIZE,
    MAX_TOTAL_UPLOAD_SIZE,
    sanitize_filename,
    validate_upload_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/materials", tags=["materials"])


def _is_supabase_enabled() -> bool:
    """Check if Supabase is configured."""
    try:
        from app.db.supabase_client import is_supabase_configured
        return is_supabase_configured()
    except ImportError:
        return False


@router.post("")
async def upload_materials(
    request: Request,
    files: list[UploadFile] = File(...),
    user = Depends(get_current_user)
):
    """
    Upload training materials (PDFs, text files).
    
    For Supabase: Files are stored in user's folder in Storage bucket.
    For local: Files are stored in materials/ directory (shared).
    """
    uploaded = 0
    errors = 0
    error_messages: list[str] = []
    total_size = 0
    user_id = get_user_id(user)

    if _is_supabase_enabled():
        # Use Supabase Storage
        from app.db.supabase_client import get_service_client, SupabaseStorage
        
        try:
            client = get_service_client()
            storage = SupabaseStorage(client, "materials")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Storage not available: {e}")

        for file in files:
            try:
                # Validate file
                content = await validate_upload_file(
                    file,
                    ALLOWED_MATERIAL_EXTENSIONS,
                    MAX_MATERIAL_SIZE,
                    f"Material '{file.filename}'",
                )

                total_size += len(content)
                if total_size > MAX_TOTAL_UPLOAD_SIZE:
                    max_mb = MAX_TOTAL_UPLOAD_SIZE / (1024 * 1024)
                    error_messages.append(f"Total upload size exceeds {max_mb:.0f} MB limit")
                    errors += 1
                    break

                filename = sanitize_filename(file.filename)
                content_type = file.content_type or "application/octet-stream"

                # Upload to Supabase Storage
                storage_path = storage.upload_file(user_id, filename, content, content_type)

                # Record in database
                client.table("user_materials").upsert({
                    "user_id": user_id,
                    "filename": filename,
                    "storage_path": storage_path,
                    "file_size": len(content),
                    "mime_type": content_type,
                }).execute()

                uploaded += 1

            except HTTPException as e:
                error_messages.append(str(e.detail))
                errors += 1
            except Exception as e:
                logger.error(f"Error uploading {file.filename}: {e}")
                error_messages.append(f"Error uploading {file.filename}: {str(e)}")
                errors += 1

        # Trigger RAG indexing
        if uploaded > 0:
            try:
                from app.materials_rag_supabase import get_user_materials_rag
                import asyncio

                async def index_async():
                    rag = get_user_materials_rag(client, user_id)
                    await asyncio.to_thread(rag.index_materials, True)

                asyncio.create_task(index_async())
                logger.info(f"RAG indexing triggered for user {user_id}")
            except Exception as e:
                logger.warning(f"Could not trigger RAG indexing: {e}")

    else:
        # Use local storage (shared materials)
        MATERIALS_DIR.mkdir(exist_ok=True)

        for file in files:
            try:
                content = await validate_upload_file(
                    file,
                    ALLOWED_MATERIAL_EXTENSIONS,
                    MAX_MATERIAL_SIZE,
                    f"Material '{file.filename}'",
                )

                total_size += len(content)
                if total_size > MAX_TOTAL_UPLOAD_SIZE:
                    max_mb = MAX_TOTAL_UPLOAD_SIZE / (1024 * 1024)
                    error_messages.append(f"Total upload size exceeds {max_mb:.0f} MB limit")
                    errors += 1
                    break

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

        # Trigger local RAG indexing
        if uploaded > 0:
            try:
                from app.materials_rag import get_materials_rag
                import asyncio

                async def index_async():
                    rag = get_materials_rag()
                    await asyncio.to_thread(rag.index_materials, True)

                asyncio.create_task(index_async())
                logger.info("RAG indexing triggered after upload")
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


@router.get("")
async def list_materials(request: Request, user = Depends(get_current_user)):
    """List all materials for the current user."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.db.supabase_client import get_service_client
        
        client = get_service_client()
        result = client.table("user_materials").select("*").eq(
            "user_id", user_id
        ).order("created_at", desc=True).execute()
        
        materials = []
        for m in result.data or []:
            materials.append({
                "id": m["id"],
                "filename": m["filename"],
                "file_size": m["file_size"],
                "mime_type": m["mime_type"],
                "indexed_at": m["indexed_at"],
                "chunk_count": m["chunk_count"],
                "created_at": m["created_at"],
            })
        
        return JSONResponse({"materials": materials})
    else:
        # Local storage - list files in materials directory
        materials = []
        if MATERIALS_DIR.exists():
            for f in sorted(MATERIALS_DIR.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    ext = f.suffix.lower()
                    if ext in ALLOWED_MATERIAL_EXTENSIONS:
                        stat = f.stat()
                        materials.append({
                            "filename": f.name,
                            "file_size": stat.st_size,
                            "mime_type": _get_mime_type(ext),
                        })
        
        return JSONResponse({"materials": materials})


@router.get("/rag-status")
async def get_rag_status(request: Request, user = Depends(get_current_user)):
    """Get RAG indexing status for current user."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        try:
            from app.db.supabase_client import get_service_client
            
            client = get_service_client()
            
            # Count embeddings
            embed_result = client.table("embeddings").select(
                "id", count="exact"
            ).eq("user_id", user_id).execute()
            
            # Count materials
            mat_result = client.table("user_materials").select(
                "id", count="exact"
            ).eq("user_id", user_id).execute()
            
            return JSONResponse({
                "available": True,
                "indexed": (embed_result.count or 0) > 0,
                "total_chunks": embed_result.count or 0,
                "total_files": mat_result.count or 0,
                "storage": "supabase",
            })
        except Exception as e:
            return JSONResponse({"available": False, "error": str(e)})
    else:
        try:
            from app.materials_rag import get_materials_rag

            rag = get_materials_rag()
            status = rag.get_status()
            status["storage"] = "local"
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


@router.post("/index")
async def index_materials(
    request: Request,
    force: bool = False,
    user = Depends(get_current_user)
):
    """Manually trigger RAG indexing of materials."""
    import asyncio
    
    user_id = get_user_id(user)

    if _is_supabase_enabled():
        try:
            from app.db.supabase_client import get_service_client
            from app.materials_rag_supabase import get_user_materials_rag

            client = get_service_client()
            rag = get_user_materials_rag(client, user_id)
            result = await asyncio.to_thread(rag.index_materials, force)
            return JSONResponse(result)
        except ImportError as e:
            return JSONResponse({"error": f"RAG dependencies not installed: {e}"})
        except Exception as e:
            return JSONResponse({"error": str(e)})
    else:
        try:
            from app.materials_rag import get_materials_rag

            rag = get_materials_rag()
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


@router.delete("/{material_id}")
async def delete_material(
    material_id: str,
    request: Request,
    user = Depends(get_current_user)
):
    """Delete a training material."""
    import urllib.parse
    
    user_id = get_user_id(user)
    material_id = urllib.parse.unquote(material_id)

    if _is_supabase_enabled():
        from app.db.supabase_client import get_service_client, SupabaseStorage

        client = get_service_client()
        
        # Get material record
        result = client.table("user_materials").select("*").eq(
            "id", material_id
        ).eq("user_id", user_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Material not found")
        
        material = result.data
        
        # Delete from storage
        storage = SupabaseStorage(client, "materials")
        storage.delete_file(material["storage_path"])
        
        # Delete embeddings
        client.table("embeddings").delete().eq(
            "material_id", material_id
        ).execute()
        
        # Delete record
        client.table("user_materials").delete().eq(
            "id", material_id
        ).execute()
        
        return JSONResponse({"message": f"Deleted {material['filename']}"})
    else:
        # Local storage - material_id is the filename
        file_path = MATERIALS_DIR / material_id

        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return JSONResponse({"message": f"Deleted {material_id}"})

        raise HTTPException(status_code=404, detail="File not found")


@router.delete("")
async def delete_all_materials(request: Request, user = Depends(get_current_user)):
    """Delete all training materials for current user."""
    user_id = get_user_id(user)
    deleted = 0

    if _is_supabase_enabled():
        from app.db.supabase_client import get_service_client, SupabaseStorage

        client = get_service_client()
        
        # Get all materials
        result = client.table("user_materials").select("*").eq(
            "user_id", user_id
        ).execute()
        
        storage = SupabaseStorage(client, "materials")
        
        for material in result.data or []:
            try:
                storage.delete_file(material["storage_path"])
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {material['filename']}: {e}")
        
        # Delete all embeddings
        client.table("embeddings").delete().eq("user_id", user_id).execute()
        
        # Delete all material records
        client.table("user_materials").delete().eq("user_id", user_id).execute()
    else:
        if MATERIALS_DIR.exists():
            for f in MATERIALS_DIR.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    f.unlink()
                    deleted += 1

    return JSONResponse({"deleted": deleted, "message": f"Deleted {deleted} files"})


def _get_mime_type(ext: str) -> str:
    """Get MIME type from extension."""
    mime_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
    }
    return mime_types.get(ext.lower(), "application/octet-stream")
