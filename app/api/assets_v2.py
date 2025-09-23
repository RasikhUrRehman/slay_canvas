from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.asset import AssetCreate, AssetCreateWithFile, AssetRead
from app.schemas.collection import (
    CollectionCreate,
    CollectionRead,
    CollectionWithAssets,
)
from app.services.asset_knowledge_service import asset_knowledge_service
from app.services.assets_service import AssetService
from app.services.collection_service import CollectionService
from app.utils.auth import get_current_user_id
from app.utils.storage import delete_file_from_minio, upload_file_to_minio

router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["assets"])
asset_service = AssetService()
collection_service = CollectionService()


# -------------------------
# Collection Endpoints
# -------------------------

@router.post("/collections/", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
async def create_collection(
    workspace_id: int,
    request: CollectionCreate,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Create a new collection in a workspace."""
    try:
        return await collection_service.create_collection(db, workspace_id, current_user_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/collections/", response_model=List[CollectionRead])
async def list_collections(
    workspace_id: int,
    include_assets: bool = False,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List all collections in a workspace."""
    collections = await collection_service.list_collections(db, workspace_id, include_assets)
    return collections


@router.get("/collections/{collection_id}", response_model=CollectionWithAssets)
async def get_collection(
    workspace_id: int,
    collection_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get a collection with its assets."""
    collection = await collection_service.get_collection(db, workspace_id, collection_id, include_assets=True)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    workspace_id: int,
    collection_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete a collection and all its assets."""
    success = await collection_service.delete_collection(db, workspace_id, collection_id)
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return None


# -------------------------
# Direct Asset Endpoints (workspace level)
# -------------------------

@router.post("/assets/link/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def create_link_asset(
    workspace_id: int,
    request: AssetCreate,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Create a link-based asset (social, wiki, internet)."""
    if request.type not in ["social", "wiki", "internet"]:
        raise HTTPException(status_code=400, detail="Invalid asset type for link creation")
    
    try:
        return await asset_service.create_asset(db, workspace_id, current_user_id, request.type, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/assets/text/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def create_text_asset(
    workspace_id: int,
    request: AssetCreate,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Create a text asset."""
    request.type = "text"
    try:
        return await asset_service.create_asset(db, workspace_id, current_user_id, "text", request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/assets/file/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def upload_file_asset(
    workspace_id: int,
    file: UploadFile = File(...),
    asset_type: str = Form(...),  # "image", "audio", "document"
    title: Optional[str] = Form(None),
    collection_id: Optional[int] = Form(None),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Upload a file asset and store in MinIO."""
    if asset_type not in ["image", "audio", "document"]:
        raise HTTPException(status_code=400, detail="Invalid asset type for file upload")
    
    try:
        # Upload file to MinIO
        file_path = await upload_file_to_minio(
            file=file,
            bucket="assets",
            folder=f"workspace_{workspace_id}/{asset_type}s"
        )
        
        # Create asset record
        asset_data = AssetCreateWithFile(
            type=asset_type,
            title=title or file.filename,
            asset_metadata={
                "original_filename": file.filename,
                "content_type": file.content_type,
                "file_size": file.size
            },
            collection_id=collection_id
        )
        
        # Save to database with file_path
        asset = await asset_service.create_file_asset(db, workspace_id, current_user_id, asset_data, file_path)
        return asset
        
    except Exception as e:
        # If asset creation fails, clean up uploaded file
        if 'file_path' in locals():
            await delete_file_from_minio("assets", file_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload asset: {str(e)}")


@router.get("/assets/", response_model=List[AssetRead])
async def list_workspace_assets(
    workspace_id: int,
    asset_type: Optional[str] = None,
    collection_id: Optional[int] = None,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List assets in workspace, optionally filtered by type or collection."""
    try:
        return await asset_service.list_assets(db, workspace_id, asset_type, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/assets/{asset_id}", response_model=AssetRead)
async def get_asset(
    workspace_id: int,
    asset_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific asset."""
    asset = await asset_service.get_asset(db, workspace_id, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(
    workspace_id: int,
    asset_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete an asset."""
    # Get asset first to check if it has a file to delete from MinIO
    asset = await asset_service.get_asset(db, workspace_id, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # Delete from database
    success = await asset_service.delete_asset(db, workspace_id, asset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # Delete file from MinIO if it exists
    if asset.file_path:
        try:
            await delete_file_from_minio("assets", asset.file_path)
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Warning: Failed to delete file from MinIO: {e}")
    
    return None


# -------------------------
# Collection Asset Endpoints
# -------------------------

@router.post("/collections/{collection_id}/assets/link/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def add_link_to_collection(
    workspace_id: int,
    collection_id: int,
    request: AssetCreate,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Add a link asset to a collection."""
    request.collection_id = collection_id
    return await create_link_asset(workspace_id, request, current_user_id, db)


@router.post("/collections/{collection_id}/assets/file/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def upload_file_to_collection(
    workspace_id: int,
    collection_id: int,
    file: UploadFile = File(...),
    asset_type: str = Form(...),
    title: Optional[str] = Form(None),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Upload a file asset to a collection."""
    return await upload_file_asset(workspace_id, file, asset_type, title, collection_id, current_user_id, db)


@router.get("/collections/{collection_id}/assets/", response_model=List[AssetRead])
async def list_collection_assets(
    workspace_id: int,
    collection_id: int,
    asset_type: Optional[str] = None,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List assets in a specific collection."""
    return await list_workspace_assets(workspace_id, asset_type, collection_id, current_user_id, db)


# -------------------------
# Knowledge Base Linking Endpoints
# -------------------------

@router.post("/assets/{asset_id}/link-to-kb/{knowledge_base_id}")
async def link_asset_to_knowledge_base(
    workspace_id: int,
    asset_id: int,
    knowledge_base_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Link an asset to a knowledge base and create chunks in Milvus."""
    try:
        result = await asset_knowledge_service.link_asset_to_knowledge_base(
            db, asset_id, knowledge_base_id, current_user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link asset: {str(e)}")


@router.post("/collections/{collection_id}/link-to-kb/{knowledge_base_id}")
async def link_collection_to_knowledge_base(
    workspace_id: int,
    collection_id: int,
    knowledge_base_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Link a collection to a knowledge base and create chunks for all assets."""
    try:
        result = await asset_knowledge_service.link_collection_to_knowledge_base(
            db, collection_id, knowledge_base_id, current_user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link collection: {str(e)}")


@router.delete("/assets/{asset_id}/unlink-from-kb")
async def unlink_asset_from_knowledge_base(
    workspace_id: int,
    asset_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Unlink an asset from its knowledge base and remove chunks."""
    try:
        result = await asset_knowledge_service.unlink_asset_from_knowledge_base(
            db, asset_id, current_user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unlink asset: {str(e)}")


@router.delete("/collections/{collection_id}/unlink-from-kb")
async def unlink_collection_from_knowledge_base(
    workspace_id: int,
    collection_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Unlink a collection from its knowledge base and remove all chunks."""
    try:
        result = await asset_knowledge_service.unlink_collection_from_knowledge_base(
            db, collection_id, current_user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unlink collection: {str(e)}")
