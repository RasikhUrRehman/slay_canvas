import instaloader
import logging
import time
import random
from urllib.parse import urlparse
from engine.processors.image_processor import ImageProcessor
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_instagram_media_urls(post_url, max_retries=3, retry_delay=2):
    """
    Extract media URLs from Instagram posts with enhanced error handling for deployment.
    
    Args:
        post_url (str): Instagram post URL
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Base delay between retries in seconds
    
    Returns:
        tuple: (media_urls dict, post object) or None on failure
    """
    logger.info(f"Attempting to fetch Instagram media from URL: {post_url}")
    
    # Enhanced Instaloader configuration for deployment
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        request_timeout=30,  # Increased timeout for deployment
        max_connection_attempts=3
    )
    try:
        loader.load_session_from_file("spodermaanle")
    except FileNotFoundError:
        loader.login("spodermaanle", "*Slaycanvas#")
        loader.save_session_to_file()
        
    # Add a small delay between requests to be respectful to Instagram's servers
    time.sleep(1)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            
            # Extract shortcode from URL
            parsed_url = urlparse(post_url)
            path_parts = parsed_url.path.strip('/').split('/')
            logger.debug(f"Parsed URL path parts: {path_parts}")

            shortcode = None
            if 'p' in path_parts:
                shortcode = path_parts[path_parts.index('p') + 1]
            elif 'reel' in path_parts:
                shortcode = path_parts[path_parts.index('reel') + 1]
            elif 'reels/audio' in path_parts:
                shortcode = path_parts[path_parts.index('audio') + 1]
            else:
                logger.error(f"Invalid Instagram URL format: {post_url}")
                return None

            logger.info(f"Extracted shortcode: {shortcode}")

            # Fetch post with timeout handling
            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            logger.info(f"Successfully fetched post metadata for shortcode: {shortcode}")

            media_urls = {"images": [], "videos": [], "audio": None}

            if post.is_video:
                media_urls["videos"].append(post.video_url)
                logger.info(f"Added video URL: {post.video_url}")
                # Instagram does not expose a direct audio-only URL
                # Must be extracted with ffmpeg after downloading
            else:
                media_urls["images"].append(post.url)
                logger.info(f"Added image URL: {post.url}")

            # Handle sidecar (multiple media in one post)
            if post.typename == "GraphSidecar":
                logger.info("Processing sidecar post with multiple media")
                for i, node in enumerate(post.get_sidecar_nodes()):
                    if node.is_video:
                        media_urls["videos"].append(node.video_url)
                        logger.info(f"Added sidecar video {i+1}: {node.video_url}")
                    else:
                        media_urls["images"].append(node.display_url)
                        logger.info(f"Added sidecar image {i+1}: {node.display_url}")

            logger.info(f"Successfully extracted media URLs: {len(media_urls['images'])} images, {len(media_urls['videos'])} videos")
            return media_urls

        except instaloader.exceptions.InstaloaderException as e:
            logger.error(f"Instaloader specific error on attempt {attempt + 1}: {str(e)}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning("Rate limit detected, increasing delay")
                delay = retry_delay * (2 ** attempt) + random.uniform(1, 3)
            elif "404" in str(e) or "not found" in str(e).lower():
                logger.error("Post not found or private - no retry needed")
                return None
            else:
                delay = retry_delay * (attempt + 1)
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                delay = retry_delay * (attempt + 1) + random.uniform(0.5, 1.5)
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
    
    logger.error(f"Failed to fetch Instagram media after {max_retries} attempts")
    return None

if __name__ == "__main__":
    url = "https://www.instagram.com/reel/DMDPPJSuZR4/?utm_source=ig_web_copy_link"#"https://www.instagram.com/reel/DOY2a8BkoYx/?utm_source=ig_web_copy_link&igsh=cG1zZ3JsMzc5bzA5" #"https://www.instagram.com/p/DOqbj1sjEjk/?utm_source=ig_web_copy_link&igsh=MTAwaGt4Y3U0eDdreA=="  
    media = get_instagram_media_urls(url)
    print(media)
    