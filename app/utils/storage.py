
# Cloudinary integration
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cloudinary
import cloudinary.uploader
from fastapi import UploadFile

from app.core.config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

# Keep the old local storage functionality for backward compatibility
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

async def upload_file_to_cloudinary(
    file: UploadFile, 
    folder: str = "",
    filename: Optional[str] = None
) -> str:
    """
    Upload a file to Cloudinary and return the secure URL.
    
    Args:
        file: FastAPI UploadFile object
        folder: Folder path within Cloudinary (optional)
        filename: Custom filename (optional, generates UUID if not provided)
    
    Returns:
        str: The secure URL from Cloudinary
    """
    try:
        # Generate filename if not provided
        if not filename:
            file_extension = os.path.splitext(file.filename or "")[1]
            filename = f"{uuid.uuid4()}{file_extension}"
        
        # Construct folder path
        folder_path = f"{folder}/{filename}".lstrip("/") if folder else filename
        
        # Read file contents
        contents = await file.read()
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            contents,
            public_id=folder_path.rsplit('.', 1)[0] if '.' in folder_path else folder_path,  # Remove extension from public_id
            folder=folder if folder else None,
            resource_type="auto"
        )
        
        return result["secure_url"]
        
    except Exception as e:
        raise Exception(f"Error uploading file to Cloudinary: {e}")

async def delete_file_from_cloudinary(file_path_or_url: str) -> bool:
    """
    Delete a file from Cloudinary.
    
    Args:
        file_path_or_url: Either the Cloudinary URL or public_id
    
    Returns:
        bool: True if successful
    """
    try:
        # Extract public_id from URL if it's a full URL
        if file_path_or_url.startswith('http'):
            # Extract public_id from Cloudinary URL
            # Format: https://res.cloudinary.com/cloud_name/image/upload/v1234567890/folder/filename.ext
            parts = file_path_or_url.split('/')
            if 'upload' in parts:
                upload_index = parts.index('upload')
                if upload_index + 1 < len(parts):
                    # Skip version if present (starts with 'v' followed by numbers)
                    start_index = upload_index + 1
                    if parts[start_index].startswith('v') and parts[start_index][1:].isdigit():
                        start_index += 1
                    
                    # Join remaining parts and remove extension
                    public_id = '/'.join(parts[start_index:])
                    public_id = public_id.rsplit('.', 1)[0] if '.' in public_id else public_id
                else:
                    public_id = file_path_or_url
            else:
                public_id = file_path_or_url
        else:
            public_id = file_path_or_url
        
        result = cloudinary.uploader.destroy(public_id)
        return result.get("result") == "ok"
    
    except Exception as e:
        raise Exception(f"Error deleting file from Cloudinary: {e}")

async def get_file_from_cloudinary(object_name: str) -> bytes:
    """
    Download a file from Cloudinary.
    
    Args:
        object_name: The Cloudinary URL
    
    Returns:
        bytes: File content (fetched via HTTP)
    """
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(object_name)
            response.raise_for_status()
            return response.content
    except Exception as e:
        raise Exception(f"Error downloading file from Cloudinary: {e}")

async def get_cloudinary_file_url(object_name: str) -> str:
    """
    Get a URL for a file from Cloudinary.
    
    Args:
        object_name: The Cloudinary URL or public_id
    
    Returns:
        str: The Cloudinary URL
    """
    # If it's already a URL, return as-is
    if object_name.startswith('http'):
        return object_name
    
    # Otherwise, assume it's a public_id and generate URL
    return cloudinary.utils.cloudinary_url(object_name)[0]

def get_cloudinary_file_info(object_name: str) -> dict:
    """
    Get information about a file in Cloudinary.
    
    Args:
        object_name: The Cloudinary URL or public_id
    
    Returns:
        dict: File information
    """
    # For Cloudinary, we can extract some info from URL
    return {
        "url": object_name,
        "public_id": object_name
    }
