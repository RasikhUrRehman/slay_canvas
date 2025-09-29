"""
Universal Web Content Extractor
A comprehensive tool to extract content from various web sources including:
- Regular webpages
- Instagram posts
- YouTube videos
- Social media content
- Images and media files
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import instaloader
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import logging
from datetime import datetime

# Import processors
from engine.processors.audio_processor import AudioTranscriber
from engine.processors.image_processor import ImageProcessor
from engine.processors.document_processor import DocumentProcessor
from app.core.config import settings
from engine.services.youtube import YouTubeProcessor
from engine.services.instagram import get_instagram_media_urls

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    """Data class to store extracted content"""
    url: str
    title: str = ""
    content: str = ""
    images: List[str] = None
    videos: List[str] = None
    audio: List[str] = None
    metadata: Dict[str, Any] = None
    extraction_time: str = ""
    content_type: str = ""
    success: bool = False
    error_message: str = ""
    transcriptions: Dict[str, Any] = None  # New field for all transcriptions
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.videos is None:
            self.videos = []
        if self.audio is None:
            self.audio = []
        if self.metadata is None:
            self.metadata = {}
        if self.transcriptions is None:
            self.transcriptions = {
                "text": "",  # Main text content
                "audio_transcription": "",  # Audio transcription
                "image_transcriptions": []  # List of {url: str, text: str}
            }
        if not self.extraction_time:
            self.extraction_time = datetime.now().isoformat()

class Extractor:
    """Universal web content extractor"""
    
    def __init__(self, headless=True, timeout=30):
        self.headless = headless
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.driver = None
        
        # Initialize processors
        self.audio_processor = AudioTranscriber()
        self.image_processor = ImageProcessor(settings.API_NINJAS_KEY)
        self.document_processor = DocumentProcessor()
        self.youtube_processor = YouTubeProcessor()
        
    def _get_selenium_driver(self):
        """Initialize Selenium WebDriver"""
        if self.driver is None:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--log-level=3")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
        return self.driver
    
    def process_content(self, url_or_content: str) -> Dict[str, Any]:
        """
        Router function that analyzes the input and routes to appropriate processor.
        Returns comprehensive transcription data for all content types.
        
        Args:
            url_or_content: URL or direct content to process
            
        Returns:
            Dict containing transcription data with structure:
            {
                "url": str,
                "content_type": str,
                "transcriptions": {
                    "text": str,
                    "audio_transcription": str,
                    "image_transcriptions": [{"url": str, "text": str}]
                },
                "metadata": dict,
                "success": bool,
                "error_message": str
            }
        """
        logger.info(f"Processing content: {url_or_content}")
        
        try:
            # Determine content type and route accordingly
            if self._is_url(url_or_content):
                extracted_content = self.extract_content(url_or_content)
            else:
                # Handle direct content (text, file paths, etc.)
                extracted_content = self._process_direct_content(url_or_content)
            
            # Convert to transcription format
            return self._format_transcription_response(extracted_content)
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return {
                "url": url_or_content,
                "content_type": "unknown",
                "transcriptions": {
                    "text": "",
                    "audio_transcription": "",
                    "image_transcriptions": []
                },
                "metadata": {},
                "success": False,
                "error_message": str(e)
            }
    
    def _is_url(self, content: str) -> bool:
        """Check if content is a URL"""
        try:
            result = urlparse(content)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _process_direct_content(self, content: str) -> ExtractedContent:
        """Process direct content (file paths, text, etc.)"""
        # Check if it's a file path
        if os.path.exists(content):
            if content.lower().endswith(('.pdf', '.docx', '.txt')):
                # Document processing
                processed_docs = self.document_processor.process_file(content)
                text_content = "\n".join([doc[0] for doc in processed_docs])
                
                return ExtractedContent(
                    url=content,
                    title=os.path.basename(content),
                    content=text_content,
                    content_type="document",
                    success=True,
                    transcriptions={
                        "text": text_content,
                        "audio_transcription": "",
                        "image_transcriptions": []
                    }
                )
            elif content.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                # Image processing
                image_text = self.image_processor.process(content)
                
                return ExtractedContent(
                    url=content,
                    title=os.path.basename(content),
                    content=image_text or "",
                    images=[content],
                    content_type="image",
                    success=True,
                    transcriptions={
                        "text": image_text or "",
                        "audio_transcription": "",
                        "image_transcriptions": [{"url": content, "text": image_text or ""}]
                    }
                )
            elif content.lower().endswith(('.mp3', '.wav', '.mp4', '.webm', '.m4a')):
                # Audio processing
                audio_text = self.audio_processor.transcribe(content)
                
                return ExtractedContent(
                    url=content,
                    title=os.path.basename(content),
                    content=audio_text or "",
                    audio=[content],
                    content_type="audio",
                    success=True,
                    transcriptions={
                        "text": audio_text or "",
                        "audio_transcription": audio_text or "",
                        "image_transcriptions": []
                    }
                )
        
        # If it's just text content
        return ExtractedContent(
            url="direct_content",
            title="Direct Text Content",
            content=content,
            content_type="text",
            success=True,
            transcriptions={
                "text": content,
                "audio_transcription": "",
                "image_transcriptions": []
            }
        )
    
    def _format_transcription_response(self, content: ExtractedContent) -> Dict[str, Any]:
        """Format extracted content as transcription response"""
        return {
            "url": content.url,
            "title": content.title,
            "content_type": content.content_type,
            "transcriptions": content.transcriptions,
            "metadata": content.metadata,
            "success": content.success,
            "error_message": content.error_message,
            "extraction_time": content.extraction_time
        }
    
    def extract_content(self, url: str) -> ExtractedContent:
        """Main method to extract content from any URL"""
        logger.info(f"Starting extraction from: {url}")
        
        # Determine content type and use appropriate extractor
        if "instagram.com" in url or "facebook.com" in url or "tiktok.com" in url:
            return self._extract_instagram(url)
        elif "youtube.com" in url or "youtu.be" in url:
            return self._extract_youtube(url)
        elif "twitter.com" in url or "x.com" in url:
            return self._extract_twitter(url)
        elif any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            return self._extract_image(url)
        elif any(ext in url.lower() for ext in ['.mp4', '.webm', '.avi', '.mov']):
            return self._extract_video_direct(url)
        else:
            return self._extract_webpage(url)
    
    def _extract_webpage(self, url: str) -> ExtractedContent:
        """Extract content from regular webpages with comprehensive processing"""
        try:
            # Try newspaper3k first for article extraction
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                # Process images found in the article
                image_transcriptions = []
                images = list(article.images) if article.images else []
                
                for image_url in images[:5]:  # Process first 5 images
                    try:
                        image_text = self.image_processor.process(image_url)
                        if image_text:
                            image_transcriptions.append({
                                "url": image_url,
                                "text": image_text
                            })
                    except Exception as e:
                        logger.warning(f"Failed to process webpage image {image_url}: {e}")
                
                # Combine all text content
                all_text = []
                if article.text:
                    all_text.append(article.text)
                for img_trans in image_transcriptions:
                    all_text.append(f"Image Text: {img_trans['text']}")
                
                return ExtractedContent(
                    url=url,
                    title=article.title or "",
                    content=article.text,
                    images=images,
                    metadata={
                        "authors": article.authors,
                        "publish_date": str(article.publish_date) if article.publish_date else "",
                        "keywords": article.keywords,
                        "summary": article.summary
                    },
                    transcriptions={
                        "text": "\n".join(all_text),
                        "audio_transcription": "",
                        "image_transcriptions": image_transcriptions
                    },
                    content_type="webpage",
                    success=True
                )
        except Exception as e:
            logger.warning(f"Newspaper3k failed for {url}: {e}")
        
        # Fallback to BeautifulSoup
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text().strip()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract images
            images = []
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src:
                    images.append(urljoin(url, src))
            
            # Extract videos
            videos = []
            for video in soup.find_all('video'):
                src = video.get('src')
                if src:
                    videos.append(urljoin(url, src))
            
            # Process images for text extraction
            image_transcriptions = []
            for image_url in images[:5]:  # Process first 5 images
                try:
                    image_text = self.image_processor.process(image_url)
                    if image_text:
                        image_transcriptions.append({
                            "url": image_url,
                            "text": image_text
                        })
                except Exception as e:
                    logger.warning(f"Failed to process webpage image {image_url}: {e}")
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            # Combine all text content
            all_text = []
            if content:
                all_text.append(content)
            for img_trans in image_transcriptions:
                all_text.append(f"Image Text: {img_trans['text']}")
            
            return ExtractedContent(
                url=url,
                title=title,
                content=content,
                images=images[:10],  # Limit to first 10 images
                videos=videos,
                metadata=metadata,
                transcriptions={
                    "text": "\n".join(all_text),
                    "audio_transcription": "",
                    "image_transcriptions": image_transcriptions
                },
                content_type="webpage",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to extract webpage {url}: {e}")
            return ExtractedContent(
                url=url,
                content_type="webpage",
                success=False,
                error_message=str(e)
            )
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'content|main|post'))
        
        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit content length
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        return metadata
    
    def _extract_instagram(self, url: str) -> ExtractedContent:
        """Extract content from Instagram posts using Instagram processor"""
        try:
            print("using instagram")
            # Use the Instagram processor to get media URLs
            media_urls = get_instagram_media_urls(url)
            print(media_urls)
            print("+"*20)
            if not media_urls:
                raise ValueError(f"Failed to extractadsfasdfasdfasdf Instagram media URLs {media_urls}, with url {url}")
            
            image_transcriptions = []
            if media_urls and media_urls.get("image_urls"):
                for image_url in media_urls.get("image_urls", []):
                    try:
                        image_text = self.image_processor.process(image_url)
                        print("Image text")
                        print(image_text)
                        if image_text:
                            image_transcriptions.append({
                                "url": image_url,
                                "text": image_text
                            })
                    except Exception as e:
                        logger.warning(f"Failed to process Instagram image {image_url}: {e}")
            
            # For videos, we would need to extract audio and transcribe
            audio_transcriptions = []
            if media_urls and media_urls.get("video_urls"):
                for video_url in media_urls.get("video_urls", []):
                    try:
                        # Note: Audio extraction from Instagram videos would require additional processing
                        # This is a placeholder for future implementation
                        audio_text = self.audio_processor.transcribe_from_url(video_url)
                        if audio_text:
                            audio_transcriptions.append(audio_text)
                    except Exception as e:
                        logger.warning(f"Failed to process Instagram video audio {video_url}: {e}")
            
            # Combine all transcriptions
            all_text = []
            post_text = media_urls.get("text")
            if post_text:
                all_text.append(f" Post Description: {post_text}")
            for img_trans in image_transcriptions:
                all_text.append(f"Image Text: {img_trans['text']}")
            for audio_text in audio_transcriptions:
                all_text.append(f"Audio Transcription: {audio_text}")
            
            return ExtractedContent(
                url=url,
                title= "Social post", # f"Instagram post by @{post.owner_username}",
                content= "", # post.caption or "",
                images=media_urls.get("image_urls", []) if media_urls else [],
                videos=media_urls.get("video_urls", []) if media_urls else [],
                metadata={
                    "uploader": media_urls.get("uploader", ""),
                    "title": media_urls.get("title", ""),
                    "extractor": media_urls.get("extractor",""),
                    "comments": "",#post.comments,
                    "date": "", #str(post.date),
                    "hashtags": "",#post.caption_hashtags,
                    "mentions": ""#post.caption_mentions
                },
                transcriptions={
                    "text": "\n".join(all_text),
                    "audio_transcription": "\n".join(audio_transcriptions),
                    "image_transcriptions": image_transcriptions
                },
                content_type="instagram",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to extract Instagram content: {e}")
            return ExtractedContent(
                url=url,
                content_type="instagram",
                success=False,
                error_message=str(e)
            )
    
    def _extract_youtube(self, url: str) -> ExtractedContent:
        """Extract content from YouTube videos using fast pytubefix approach"""
        try:
            from pytubefix import YouTube
            
            # Initialize YouTube object for fast metadata extraction
            yt = YouTube(url)
            
            # Get basic metadata
            metadata = {
                "title": yt.title,
                "author": yt.author,
                "channel_url": yt.channel_url,
                "description": yt.description,
                "publish_date": yt.publish_date.strftime("%Y-%m-%d") if yt.publish_date else None,
                "views": yt.views,
                "length_seconds": yt.length,
                "thumbnail_url": yt.thumbnail_url,
                "keywords": yt.keywords,
                "rating": yt.rating,
            }
            
            # Get only lowest resolution streams for performance
            audio_streams = yt.streams.filter(only_audio=True).order_by('abr').asc()
            video_streams = yt.streams.filter(adaptive=True, only_video=True).order_by('resolution').asc()
            
            # Get the lowest quality audio and video URLs
            lowest_audio_url = audio_streams.first().url if audio_streams else None
            lowest_video_url = video_streams.first().url if video_streams else None
            
            # Use audio transcription from YouTube processor
            audio_transcript = ""
            try:
                audio_transcript = self.youtube_processor.process_audio(url)
                logger.info("Audio transcription completed")
            except Exception as e:
                logger.warning(f"Audio transcription failed: {e}")
            
            # Process thumbnail for text extraction (only the main thumbnail)
            image_transcriptions = []
            if yt.thumbnail_url:
                try:
                    thumb_text = self.image_processor.process(yt.thumbnail_url)
                    if thumb_text:
                        image_transcriptions.append({
                            "url": yt.thumbnail_url,
                            "text": thumb_text
                        })
                        logger.info("Thumbnail text extraction completed")
                except Exception as e:
                    logger.warning(f"Failed to process thumbnail: {e}")
            
            # Combine all transcriptions
            all_text = []
            if yt.description:
                all_text.append(yt.description)
            if audio_transcript:
                all_text.append(f"Audio Transcription: {audio_transcript}")
            for img_trans in image_transcriptions:
                all_text.append(f"Thumbnail Text: {img_trans['text']}")
            
            # Build stream info for metadata (only lowest quality)
            streams_info = []
            if lowest_audio_url:
                audio_stream = audio_streams.first()
                streams_info.append({
                    "type": "audio",
                    "mime_type": audio_stream.mime_type,
                    "abr": getattr(audio_stream, "abr", None),
                    "filesize_mb": round(audio_stream.filesize / (1024 * 1024), 2) if audio_stream.filesize else None,
                })
            
            if lowest_video_url:
                video_stream = video_streams.first()
                streams_info.append({
                    "type": "video",
                    "mime_type": video_stream.mime_type,
                    "resolution": getattr(video_stream, "resolution", None),
                    "fps": getattr(video_stream, "fps", None),
                    "filesize_mb": round(video_stream.filesize / (1024 * 1024), 2) if video_stream.filesize else None,
                })
            
            metadata["available_streams"] = streams_info
            
            return ExtractedContent(
                url=url,
                title=yt.title or "",
                content=yt.description or "",
                images=[yt.thumbnail_url] if yt.thumbnail_url else [],
                videos=[url],
                audio=[url] if audio_transcript else [],
                metadata=metadata,
                transcriptions={
                    "text": "\n".join(all_text),
                    "audio_transcription": audio_transcript or "",
                    "image_transcriptions": image_transcriptions
                },
                content_type="youtube",
                success=True
            )
                
        except Exception as e:
            logger.error(f"Failed to extract YouTube content: {e}")
            return ExtractedContent(
                url=url,
                content_type="youtube",
                success=False,
                error_message=str(e)
            )
    
    def _is_valid_tweet_image(self, src: str) -> bool:
        """Check if the image URL is a valid tweet image (not UI elements or thumbnails)"""
        if not src:
            return False
        
        # Filter out common UI elements and unwanted images
        unwanted_patterns = [
            'profile_images',  # Profile pictures
            'sticky/videos',   # UI video thumbnails
            'abs.twimg.com',   # Twitter UI assets
            'emoji',           # Emoji images
            'avatar',          # Avatar images
            'default_profile', # Default profile images
            'icon',            # Icon images
            'logo',            # Logo images
            'badge',           # Badge images
            'verified',        # Verification badges
        ]
        
        # Check if the image URL contains unwanted patterns
        src_lower = src.lower()
        for pattern in unwanted_patterns:
            if pattern in src_lower:
                return False
        
        # Accept images from pbs.twimg.com (actual tweet content)
        # but filter out video thumbnails that are just for UI
        if 'pbs.twimg.com' in src:
            # Accept media images but not amplify video thumbs unless they're the main content
            if 'media' in src or 'tweet_video' in src:
                return True
            if 'amplify_video_thumb' in src:
                # Accept small/medium quality versions for lower file sizes
                return 'name=small' in src or 'name=medium' in src or 'name=thumb' in src
        
        # Accept video.twimg.com images (could be video thumbnails that are main content)
        if 'video.twimg.com' in src and '.jpg' in src:
            return True
            
        return True  # Accept other images by default
    
    def _extract_twitter(self, url: str) -> ExtractedContent:
        """Extract content from Twitter/X posts using Selenium"""
        try:
            driver = self._get_selenium_driver()
            driver.get(url)
            time.sleep(5)  # Wait for content to load
            
            # Extract tweet text
            tweet_text = ""
            try:
                tweet_element = driver.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                tweet_text = tweet_element.text
            except:
                # Fallback selectors
                try:
                    tweet_element = driver.find_element(By.CSS_SELECTOR, '.tweet-text')
                    tweet_text = tweet_element.text
                except:
                    try:
                        tweet_element = driver.find_element(By.CSS_SELECTOR, '[data-testid="tweetTextContainer"]')
                        tweet_text = tweet_element.text
                    except:
                        pass
            
            # Extract single main image from the tweet
            image_url = None
            try:
                # Priority selectors for actual tweet images (not thumbnails or UI elements)
                img_selectors = [
                    '[data-testid="tweetPhoto"] img',  # Main tweet photo
                    '[data-testid="card.wrapper"] img[alt*="Image"]',  # Card images
                    'article img[alt*="Image"]',  # Images within the tweet article
                    '.media-image img'  # Media container images
                ] # TODO: Implement extractor for multiple image extraction on a single tweet
                
                for selector in img_selectors:
                    try:
                        img_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for img in img_elements:
                            src = img.get_attribute('src')
                            print("Valid image URL:",src)
                            image_url = src
                            # if src and self._is_valid_tweet_image(src):
                                
                            #     # Always prioritize small quality for lower file size
                            #     base_url = src.split('?')[0]  # Remove query parameters
                            #     image_url = f"{base_url}?name=small"
                            #     break
                        if image_url:
                            break
                    except:
                        continue
                        
                # If no image found with selectors, check page source for image URLs
                if not image_url:
                    page_source = driver.page_source
                    # Look for media URLs in the page source
                    media_patterns = [
                        r'https://pbs\.twimg\.com/media/[^"\'<>\s]+\.(jpg|jpeg|png|gif|webp)[^"\'<>\s]*',
                        r'https://pbs\.twimg\.com/tweet_video_thumb/[^"\'<>\s]+\.(jpg|jpeg|png)[^"\'<>\s]*'
                    ]
                    
                    for pattern in media_patterns:
                        matches = re.findall(pattern, page_source, re.IGNORECASE)
                        if matches:
                            # Take the first match and ensure it's small quality
                            raw_url = matches[0][0] if isinstance(matches[0], tuple) else matches[0]
                            base_url = raw_url.split('?')[0]
                            image_url = f"{base_url}?name=small"
                            break
                            
            except Exception as img_error:
                logger.warning(f"Error extracting image: {img_error}")
            
            # Create images list with single image if found
            images = [image_url] if image_url else []
            
            # Extract videos - prioritize single high-quality video
            video_url = None
            try:
                # Check page source for video URLs to get the highest quality one
                page_source = driver.page_source
                
                # Look for video URLs, prioritizing high quality formats
                video_url_patterns = [
                    r'https://video\.twimg\.com/amplify_video/[^"\'<>\s]+/vid/avc1/1920x1080/[^"\'<>\s]+\.mp4[^"\'<>\s]*',  # 1080p
                    r'https://video\.twimg\.com/amplify_video/[^"\'<>\s]+/vid/avc1/1280x720/[^"\'<>\s]+\.mp4[^"\'<>\s]*',   # 720p
                    r'https://video\.twimg\.com/amplify_video/[^"\'<>\s]+/vid/avc1/640x360/[^"\'<>\s]+\.mp4[^"\'<>\s]*',    # 360p
                    r'https://video\.twimg\.com/tweet_video/[^"\'<>\s]+\.mp4[^"\'<>\s]*',  # Tweet video
                    r'https://video\.twimg\.com/[^"\'<>\s]+\.mp4[^"\'<>\s]*',  # Any twimg video
                    r'"video_url":"([^"]*\.mp4[^"]*)"',
                    r'"media_url_https":"([^"]*\.mp4[^"]*)"'
                ]
                
                # Try patterns in order of priority (highest quality first)
                for pattern in video_url_patterns:
                    matches = re.findall(pattern, page_source, re.IGNORECASE)
                    if matches:
                        # Take the first match for this quality level
                        video_url = matches[0]
                        if isinstance(video_url, tuple):
                            video_url = video_url[0]
                        # Decode escaped URLs
                        video_url = video_url.replace('\\/', '/').replace('\\u0026', '&')
                        break
                
                # If no video found in page source, try DOM selectors
                if not video_url:
                    video_selectors = [
                        'video',
                        '[data-testid="videoPlayer"] video',
                        '[data-testid="previewInterstitial"] video'
                    ]
                    
                    for selector in video_selectors:
                        try:
                            video_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            for video in video_elements:
                                src = video.get_attribute('src')
                                if src and '.mp4' in src:
                                    video_url = src
                                    break
                            if video_url:
                                break
                        except:
                            continue
                            
            except Exception as video_error:
                logger.warning(f"Error extracting video: {video_error}")
            
            # Create videos list with single video if found
            videos = [video_url] if video_url else []
            
            # Extract user info
            username = ""
            display_name = ""
            try:
                # Try different selectors for username
                username_selectors = [
                    '[data-testid="User-Name"] [dir="ltr"]',
                    '[data-testid="User-Names"] [dir="ltr"]',
                    '.username',
                    '[data-testid="UserName"]'
                ]
                
                for selector in username_selectors:
                    try:
                        username_element = driver.find_element(By.CSS_SELECTOR, selector)
                        username = username_element.text
                        break
                    except:
                        continue
                        
                # Try to get display name
                try:
                    display_name_element = driver.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"] span:first-child')
                    display_name = display_name_element.text
                except:
                    pass
                    
            except:
                pass
            
            # Extract additional metadata
            metadata = {
                "username": username,
                "display_name": display_name,
                "retweets": 0,
                "likes": 0,
                "replies": 0
            }
            
            # Try to extract engagement metrics
            try:
                # Look for retweet count
                retweet_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="retweet"] [data-testid="app-text-transition-container"]')
                if retweet_elements:
                    metadata["retweets"] = retweet_elements[0].text
                    
                # Look for like count
                like_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="like"] [data-testid="app-text-transition-container"]')
                if like_elements:
                    metadata["likes"] = like_elements[0].text
                    
                # Look for reply count
                reply_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="reply"] [data-testid="app-text-transition-container"]')
                if reply_elements:
                    metadata["replies"] = reply_elements[0].text
            except:
                pass
            
            print("Image URL:",image_url)
            print("Video URL:",video_url)
            # Process media for transcriptions
            image_transcriptions = []
            if images:
                for image_url in images:
                    try:
                        image_text = self.image_processor.process(image_url)
                        if image_text:
                            image_transcriptions.append({
                                "url": image_url,
                                "text": image_text
                            })
                    except Exception as e:
                        logger.warning(f"Failed to process Twitter image {image_url}: {e}")
            
            # Process video for audio transcription
            audio_transcription = ""
            if videos:
                for video_url in videos:
                    try:
                        audio_text = self.audio_processor.transcribe_from_url(video_url)
                        if audio_text:
                            audio_transcription = audio_text
                            break  # Use first successful transcription
                    except Exception as e:
                        logger.warning(f"Failed to process Twitter video audio {video_url}: {e}")
            
            # Combine all text content
            all_text = []
            if tweet_text:
                all_text.append(tweet_text)
            if audio_transcription:
                all_text.append(f"Audio Transcription: {audio_transcription}")
            for img_trans in image_transcriptions:
                all_text.append(f"Image Text: {img_trans['text']}")
            
            return ExtractedContent(
                url=url,
                title=f"Tweet by {display_name} (@{username})" if username else "Twitter Post",
                content=tweet_text,
                images=images,  # Single image or empty list
                videos=videos,  # Single video or empty list
                metadata=metadata,
                transcriptions={
                    "text": "\n".join(all_text),
                    "audio_transcription": audio_transcription,
                    "image_transcriptions": image_transcriptions
                },
                content_type="twitter",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to extract Twitter content: {e}")
            return ExtractedContent(
                url=url,
                content_type="twitter",
                success=False,
                error_message=str(e)
            )
    
    def _extract_image(self, url: str) -> ExtractedContent:
        """Extract content from direct image URLs"""
        try:
            # Process the image using image processor
            image_text = self.image_processor.process(url)
            
            # Get image metadata
            response = self.session.head(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', 'Unknown')
            
            return ExtractedContent(
                url=url,
                title=f"Image: {os.path.basename(urlparse(url).path)}",
                content=image_text or "",
                images=[url],
                metadata={
                    "content_type": content_type,
                    "content_length": content_length
                },
                transcriptions={
                    "text": image_text or "",
                    "audio_transcription": "",
                    "image_transcriptions": [{"url": url, "text": image_text or ""}]
                },
                content_type="image",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to extract image: {e}")
            return ExtractedContent(
                url=url,
                content_type="image",
                success=False,
                error_message=str(e)
            )
    
    def process_uploaded_image(self, image_path: str, original_filename: str = None) -> Dict[str, Any]:
        """
        Process a directly uploaded image file.
        
        Args:
            image_path: Path to the uploaded image file
            original_filename: Original filename of the uploaded image
            
        Returns:
            Dict containing transcription data with structure:
            {
                "url": str,
                "content_type": str,
                "transcriptions": {
                    "text": str,
                    "audio_transcription": str,
                    "image_transcriptions": [{"url": str, "text": str}]
                },
                "metadata": dict,
                "success": bool,
                "error_message": str
            }
        """
        logger.info(f"Processing uploaded image: {image_path}")
        
        try:
            # Validate file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Validate file is an image
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff')
            if not image_path.lower().endswith(valid_extensions):
                raise ValueError(f"Invalid image format. Supported formats: {valid_extensions}")
            
            # Process the image using image processor
            image_text = self.image_processor.process(image_path)
            
            # Get file metadata
            file_stats = os.stat(image_path)
            file_size = file_stats.st_size
            
            # Use original filename if provided, otherwise use the file path
            display_name = original_filename or os.path.basename(image_path)
            
            extracted_content = ExtractedContent(
                url=image_path,
                title=f"Uploaded Image: {display_name}",
                content=image_text or "",
                images=[image_path],
                metadata={
                    "original_filename": original_filename,
                    "file_size": file_size,
                    "file_path": image_path,
                    "processed_at": datetime.now().isoformat()
                },
                transcriptions={
                    "text": image_text or "",
                    "audio_transcription": "",
                    "image_transcriptions": [{"url": image_path, "text": image_text or ""}]
                },
                content_type="uploaded_image",
                success=True
            )
            
            # Convert to transcription format
            return self._format_transcription_response(extracted_content)
            
        except Exception as e:
            logger.error(f"Failed to process uploaded image: {e}")
            return {
                "url": image_path,
                "content_type": "uploaded_image",
                "transcriptions": {
                    "text": "",
                    "audio_transcription": "",
                    "image_transcriptions": []
                },
                "metadata": {
                    "original_filename": original_filename,
                    "error": str(e)
                },
                "success": False,
                "error_message": str(e)
            }

    def _extract_video_direct(self, url: str) -> ExtractedContent:
        """Extract direct video content with audio transcription"""
        try:
            response = self.session.head(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', 'Unknown')
            
            # Try to transcribe audio from the video
            audio_transcription = ""
            try:
                audio_text = self.audio_processor.transcribe_from_url(url)
                if audio_text:
                    audio_transcription = audio_text
            except Exception as e:
                logger.warning(f"Failed to transcribe video audio {url}: {e}")
            
            return ExtractedContent(
                url=url,
                title=f"Video: {os.path.basename(urlparse(url).path)}",
                content=audio_transcription or "Direct video URL",
                videos=[url],
                audio=[url] if audio_transcription else [],
                metadata={
                    "content_type": content_type,
                    "content_length": content_length
                },
                transcriptions={
                    "text": audio_transcription or "",
                    "audio_transcription": audio_transcription,
                    "image_transcriptions": []
                },
                content_type="video",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to extract video: {e}")
            return ExtractedContent(
                url=url,
                content_type="video",
                success=False,
                error_message=str(e)
            )
    
    def save_content(self, content: ExtractedContent, filename: str = None) -> str:
        """Save extracted content to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_content_{timestamp}.json"
        
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(content), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Content saved to: {filepath}")
        return filepath
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None
        self.session.close()

def main():
    """Example usage demonstrating the router functionality"""
    extractor = Extractor()

    # Example URLs and content to test
    test_items = [
        #"https://www.geeksforgeeks.org", 
        #"https://en.wikipedia.org/wiki/Pakistan",
        #"https://www.youtube.com/watch?v=fqvRz5u4QJE&list=RDfqvRz5u4QJE&start_radio=1",
        #"https://x.com/realDonaldTrump/status/1967654825308590378",
        #"https://x.com/elonmusk/status/1968589844491215324",
        #"https://x.com/elonmusk/status/1967938574835388553",
        #"https://x.com/ValentinaForUSA/status/1967664132444045739",
        #"https://www.instagram.com/p/DOOQ0p6DBPI/?utm_source=ig_web_copy_link&igsh=MTkyb3VtbjVpbWprNg==", #image
        #"https://www.instagram.com/reel/DOvPrnYjZPZ/?utm_source=ig_web_copy_link&igsh=MXdodmVjMG56MXVybA==", #video
        #"https://www.instagram.com/reel/DHECrYAzKoa/?utm_source=ig_web_copy_link&igsh=dTFzMzdyb3R0YTVv",
        #"https://www.instagram.com/reel/DOY2a8BkoYx/?utm_source=ig_web_copy_link&igsh=cG1zZ3JsMzc5bzA5",
        #"https://www.instagram.com/reel/DMDPPJSuZR4/?utm_source=ig_web_copy_link",
        "https://www.tiktok.com/@mbedite/video/7534034640798108959?is_from_webapp=1&sender_device=pc"
        #"https://www.facebook.com/share/v/19tECkUWis/"
        #"uploads/video_WordPress Blog & n8n Automation for Beginners Step-by-Step Guide.mp4",
        #"uploads/audio_Ed Sheeran - Perfect (Lyrics).m4a",
        #"https://www.youtube.com/watch?v=ba7mB8oueCY&list=RDba7mB8oueCY&start_radio=1"
        #"uploads/pic2.png"
        #"uploads/audio_Lewis Capaldi - Someone You Loved (Lyrics).m4a",
        #"hello hi there",    
    ]
    
    print("🚀 Testing Universal Content Extractor with Router")
    print("="*60)
    
    for item in test_items:
        print(f"\n{'='*50}")
        print(f"Processing: {item}")
        print('='*50)
        
        # Use the new router function
        result = extractor.process_content(item)
        
        if result["success"]:
            print(f"✅ Success!")
            print("="*40)
            print(result)
            print("="*40)

            print(f"Content Type: {result['content_type']}")
            print(f"Title: {result.get('title', 'N/A')}")
            
            # Show transcription results
            transcriptions = result["transcriptions"]
            if transcriptions["text"]:
                print(f"📝 Text Content: {transcriptions['text'][:200]}...")
            if transcriptions["audio_transcription"]:
                print(f"🎵 Audio Transcription: {transcriptions['audio_transcription'][:200]}...")
            if transcriptions["image_transcriptions"]:
                print(f"🖼️ Image Transcriptions: {len(transcriptions['image_transcriptions'])} images processed")
                for img_trans in transcriptions["image_transcriptions"][:2]:  # Show first 2
                    print(f"  - {img_trans['url']}: {img_trans['text'][:100]}...")
            
            print(f"⏰ Extraction Time: {result['extraction_time']}")

            # assume your extractor gave you a dictionary called `result`
            print("complete:",result["transcriptions"]["text"])
            
            
        else:
            print(f"❌ Failed: {result['error_message']}")
    
    extractor.close()
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
