import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

import aiofiles
from fastapi import UploadFile
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

# Keep the old local storage functionality
BASE_STORAGE = Path.cwd() / 'storage'
BASE_STORAGE.mkdir(parents=True, exist_ok=True)


def save_upload(file_bytes: bytes, filename: str, subfolder: str = '') -> Tuple[str, str]:
    """Legacy local file storage function."""
    folder = BASE_STORAGE / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    with open(path, 'wb') as f:
        f.write(file_bytes)
    return str(path), str(path.relative_to(Path.cwd()))


# MinIO client initialization
try:
    minio_client = Minio(
        endpoint=f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False  # Set to True for HTTPS
    )
except Exception as e:
    print(f"Warning: Failed to initialize MinIO client: {e}")
    minio_client = None


async def ensure_bucket_exists(bucket_name: str) -> None:
    """Ensure that a bucket exists, create it if it doesn't."""
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
    except S3Error as e:
        raise Exception(f"Error creating bucket {bucket_name}: {e}")


async def upload_file_to_minio(
    file: UploadFile, 
    bucket: str, 
    folder: str = "",
    filename: Optional[str] = None
) -> str:
    """
    Upload a file to MinIO and return the file path.
    
    Args:
        file: FastAPI UploadFile object
        bucket: MinIO bucket name
        folder: Folder path within bucket (optional)
        filename: Custom filename (optional, generates UUID if not provided)
    
    Returns:
        str: The file path in MinIO (object_name)
    """
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    try:
        # Ensure bucket exists
        await ensure_bucket_exists(bucket)
        
        # Generate filename if not provided
        if not filename:
            file_extension = os.path.splitext(file.filename or "")[1]
            filename = f"{uuid.uuid4()}{file_extension}"
        
        # Construct object name (full path in MinIO)
        object_name = f"{folder}/{filename}".lstrip("/")
        
        # Create temporary file
        temp_file_path = f"/tmp/{uuid.uuid4()}_{filename}"
        
        try:
            # Save uploaded file temporarily
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                content = await file.read()
                await temp_file.write(content)
            
            # Upload to MinIO
            minio_client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=temp_file_path,
                content_type=file.content_type
            )
            
            return object_name
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except S3Error as e:
        raise Exception(f"Error uploading file to MinIO: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during file upload: {e}")


async def get_file_from_minio(bucket: str, object_name: str) -> bytes:
    """
    Download a file from MinIO.
    
    Args:
        bucket: MinIO bucket name
        object_name: File path in MinIO
    
    Returns:
        bytes: File content
    """
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    try:
        response = minio_client.get_object(bucket, object_name)
        return response.read()
    except S3Error as e:
        raise Exception(f"Error downloading file from MinIO: {e}")
    finally:
        if 'response' in locals():
            response.close()
            response.release_conn()


async def delete_file_from_minio(bucket: str, object_name: str) -> bool:
    """
    Delete a file from MinIO.
    
    Args:
        bucket: MinIO bucket name
        object_name: File path in MinIO
    
    Returns:
        bool: True if successful
    """
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    try:
        minio_client.remove_object(bucket, object_name)
        return True
    except S3Error as e:
        raise Exception(f"Error deleting file from MinIO: {e}")


async def get_file_url(bucket: str, object_name: str, expires: int = 3600) -> str:
    """
    Get a presigned URL for a file in MinIO.
    
    Args:
        bucket: MinIO bucket name
        object_name: File path in MinIO
        expires: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        str: Presigned URL
    """
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    try:
        return minio_client.presigned_get_object(bucket, object_name, expires=expires)
    except S3Error as e:
        raise Exception(f"Error generating presigned URL: {e}")


def get_file_info(bucket: str, object_name: str) -> dict:
    """
    Get information about a file in MinIO.
    
    Args:
        bucket: MinIO bucket name
        object_name: File path in MinIO
    
    Returns:
        dict: File information
    """
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    try:
        stat = minio_client.stat_object(bucket, object_name)
        return {
            "size": stat.size,
            "last_modified": stat.last_modified,
            "content_type": stat.content_type,
            "etag": stat.etag
        }
    except S3Error as e:
        raise Exception(f"Error getting file info: {e}")
