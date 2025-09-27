import json
import logging
import random
import re
import time
from urllib.parse import urlparse

import instaloader
import requests

from app.core.config import settings
from engine.processors.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents for rotation to avoid detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
]

def get_instagram_media_urls(post_url, max_retries=2, retry_delay=1):
    """
    Extract media URLs from Instagram posts with fast processing and minimal delays.
    
    Args:
        post_url (str): Instagram post URL
        max_retries (int): Maximum number of retry attempts (kept low for speed)
        retry_delay (int): Base delay between retries in seconds
    
    Returns:
        dict: media_urls dict with images, videos, audio or None on failure
    """
    logger.info(f"Attempting to fetch Instagram media from URL: {post_url}")
    
    # Extract shortcode first to validate URL
    shortcode = _extract_shortcode(post_url)
    if not shortcode:
        logger.error(f"Invalid Instagram URL format: {post_url}")
        return None
    
    # Try primary method with minimal retries and delays
    result = _try_instaloader_primary(shortcode, max_retries, retry_delay)
    if result:
        return result
    
    # Skip fallback for now to avoid blocking - can be enabled later if needed
    logger.warning("Primary method failed, skipping fallback to avoid blocking main thread")
    return None


def _extract_shortcode(post_url):
    """Extract shortcode from Instagram URL"""
    parsed_url = urlparse(post_url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    shortcode = None
    if 'p' in path_parts:
        shortcode = path_parts[path_parts.index('p') + 1]
    elif 'reel' in path_parts:
        shortcode = path_parts[path_parts.index('reel') + 1]
    elif 'reels/audio' in path_parts:
        shortcode = path_parts[path_parts.index('audio') + 1]
    
    return shortcode


def _try_instaloader_primary(shortcode, max_retries, retry_delay):
    """Primary instaloader attempt with minimal delays for speed"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Primary method attempt {attempt + 1}/{max_retries}")
            
            # Rotate user agents to avoid detection
            user_agent = random.choice(USER_AGENTS)
            
            # Fast configuration - minimal delays
            loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False,
                user_agent=user_agent,
                request_timeout=15,  # Shorter timeout for speed
                max_connection_attempts=1,  # Fewer attempts
                sleep=False,  # Disable sleep for speed
                quiet=True
            )
            
            # Minimal delay - no more blocking!
            initial_delay = random.uniform(0.1, 0.5)  # Very short delay
            logger.info(f"Waiting {initial_delay:.1f}s before request...")
            time.sleep(initial_delay)
            
            # Fetch post
            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            logger.info(f"Successfully fetched post metadata for shortcode: {shortcode}")
            
            return _extract_media_from_post(post)
            
        except instaloader.exceptions.InstaloaderException as e:
            error_msg = str(e).lower()
            logger.error(f"Instaloader error on attempt {attempt + 1}: {str(e)}")
            
            # Much shorter delays for all error types
            if "403" in str(e) or "forbidden" in error_msg:
                logger.warning("403 Forbidden - Instagram blocking requests")
                delay = retry_delay + random.uniform(0.5, 1.0)  # Very short delay
            elif "401" in str(e) or "unauthorized" in error_msg:
                logger.warning("401 Unauthorized - Authentication issue")
                delay = retry_delay + random.uniform(0.2, 0.8)  # Very short delay
            elif "429" in str(e) or "rate limit" in error_msg:
                logger.warning("Rate limit detected")
                delay = retry_delay + random.uniform(1, 2)  # Short delay
            elif "404" in str(e) or "not found" in error_msg:
                logger.error("Post not found or private")
                return None  # Don't retry for 404s
            elif "timeout" in error_msg or "connection" in error_msg:
                logger.warning("Connection/timeout issue")
                delay = retry_delay + random.uniform(0.5, 1.5)  # Short delay
            else:
                delay = retry_delay + random.uniform(0.3, 1.0)  # Short delay
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                delay = retry_delay + random.uniform(0.5, 1.5)  # Short delay
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
    
    return None


def _try_instaloader_fallback(shortcode, max_retries, retry_delay):
    """Fallback instaloader attempt with minimal configuration"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Fallback method attempt {attempt + 1}/{max_retries}")
            
            # Minimal configuration to avoid detection
            loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False,
                request_timeout=60,  # Even longer timeout
                max_connection_attempts=1,
                sleep=True,
                quiet=True
            )
            
            # Longer delay for fallback
            delay_time = random.uniform(10, 20)
            logger.info(f"Fallback waiting {delay_time:.1f}s before request...")
            time.sleep(delay_time)
            
            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            logger.info(f"Fallback successfully fetched post for shortcode: {shortcode}")
            
            return _extract_media_from_post(post)
            
        except Exception as e:
            logger.error(f"Fallback error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt) + random.uniform(15, 30)
                logger.info(f"Fallback retrying in {delay:.1f} seconds...")
                time.sleep(delay)
    
    return None


def _extract_media_from_post(post):
    """Extract media URLs from Instagram post object"""
    media_urls = {"images": [], "videos": [], "audio": None}
    
    try:
        if post.is_video:
            if hasattr(post, 'video_url') and post.video_url:
                media_urls["videos"].append(post.video_url)
                logger.info(f"Added video URL: {post.video_url}")
        else:
            if hasattr(post, 'url') and post.url:
                media_urls["images"].append(post.url)
                logger.info(f"Added image URL: {post.url}")
        
        # Handle sidecar (multiple media in one post)
        if hasattr(post, 'typename') and post.typename == "GraphSidecar":
            logger.info("Processing sidecar post with multiple media")
            try:
                for i, node in enumerate(post.get_sidecar_nodes()):
                    if hasattr(node, 'is_video') and node.is_video:
                        if hasattr(node, 'video_url') and node.video_url:
                            media_urls["videos"].append(node.video_url)
                            logger.info(f"Added sidecar video {i+1}: {node.video_url}")
                    else:
                        if hasattr(node, 'display_url') and node.display_url:
                            media_urls["images"].append(node.display_url)
                            logger.info(f"Added sidecar image {i+1}: {node.display_url}")
            except Exception as e:
                logger.warning(f"Error processing sidecar nodes: {e}")
        
        logger.info(f"Successfully extracted media URLs: {len(media_urls['images'])} images, {len(media_urls['videos'])} videos")
        return media_urls
        
    except Exception as e:
        logger.error(f"Error extracting media from post: {e}")
        return None
