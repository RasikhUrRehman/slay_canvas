# # # # Install required packages:
# # # # pip install httpx parsel jmespath selenium webdriver-manager

# # # import json
# # # import time
# # # from typing import Dict, Optional
# # # from parsel import Selector
# # # from selenium import webdriver
# # # from selenium.webdriver.chrome.service import Service
# # # from webdriver_manager.chrome import ChromeDriverManager
# # # from selenium.webdriver.chrome.options import Options
# # # import jmespath
# # # import logging

# # # # Set up logging
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # def fetch_tiktok_video_details(video_url: str, proxy: Optional[str] = None, retries: int = 3) -> Dict:
# # #     """
# # #     Fetch details from a TikTok video URL using Selenium for dynamic content rendering.
    
# # #     Args:
# # #         video_url (str): The URL of the TikTok video, e.g., "https://www.tiktok.com/@username/video/123456789"
# # #         proxy (Optional[str]): Proxy server in format 'http://host:port' or None
# # #         retries (int): Number of retry attempts if the request fails
    
# # #     Returns:
# # #         dict: A dictionary containing video details like ID, description, video URL, cover image, download URL, author info, stats, etc.
    
# # #     Raises:
# # #         ValueError: If video data cannot be extracted after retries
# # #     """
# # #     # Set up Chrome options
# # #     chrome_options = Options()
# # #     chrome_options.add_argument("--headless")  # Run in headless mode
# # #     chrome_options.add_argument("--disable-gpu")
# # #     chrome_options.add_argument("--no-sandbox")
# # #     chrome_options.add_argument("--disable-dev-shm-usage")
# # #     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
# # #     if proxy:
# # #         chrome_options.add_argument(f'--proxy-server={proxy}')
    
# # #     # Initialize WebDriver
# # #     driver = None
# # #     attempt = 0
    
# # #     while attempt < retries:
# # #         try:
# # #             logging.info(f"Attempt {attempt + 1}/{retries} to fetch {video_url}")
# # #             driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
# # #             driver.get(video_url)
            
# # #             # Wait for the page to load (adjust time as needed)
# # #             time.sleep(3)
            
# # #             # Get page source and parse
# # #             selector = Selector(driver.page_source)
# # #             script_data = selector.xpath("//script[@id='__UNIVERSAL_DATA_FOR_REHYDRATION__']/text()").get()
            
# # #             if not script_data:
# # #                 raise ValueError("Could not find video data in the page. The page might be blocked or structure changed.")
            
# # #             # Load JSON data
# # #             data = json.loads(script_data)
# # #             post_data = data["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"]["itemStruct"]
            
# # #             # Extract relevant fields with JMESPath
# # #             parsed_data = jmespath.search(
# # #                 """{
# # #                 id: id,
# # #                 desc: desc,
# # #                 createTime: createTime,
# # #                 video: video.{duration: duration, ratio: ratio, cover: cover, playAddr: playAddr, downloadAddr: downloadAddr, bitrate: bitrate},
# # #                 author: author.{id: id, uniqueId: uniqueId, nickname: nickname, avatarLarger: avatarLarger, signature: signature, verified: verified},
# # #                 stats: stats,
# # #                 locationCreated: locationCreated,
# # #                 diversificationLabels: diversificationLabels,
# # #                 suggestedWords: suggestedWords,
# # #                 contents: contents[].{textExtra: textExtra[].{hashtagName: hashtagName}}
# # #                 }""",
# # #                 post_data
# # #             )
            
# # #             # Handle video URLs
# # #             parsed_data['video_url'] = parsed_data['video']['playAddr']
# # #             parsed_data['download_url'] = parsed_data['video']['downloadAddr']
# # #             parsed_data['cover_image'] = parsed_data['video']['cover']
# # #             del parsed_data['video']['playAddr']
# # #             del parsed_data['video']['downloadAddr']
            
# # #             logging.info("Successfully extracted video details")
# # #             return parsed_data
        
# # #         except Exception as e:
# # #             logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
# # #             attempt += 1
# # #             if attempt < retries:
# # #                 time.sleep(2 ** attempt)  # Exponential backoff
# # #             else:
# # #                 raise ValueError(f"Failed to fetch video data after {retries} attempts: {str(e)}")
        
# # #         finally:
# # #             if driver:
# # #                 driver.quit()
    
# # #     return {}

# # # # Example usage
# # # if __name__ == "__main__":
# # #     video_url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"  # Replace with your TikTok video URL
# # #     # proxy = "http://your-proxy:port"  # Optional: Add proxy if needed
# # #     try:
# # #         details = fetch_tiktok_video_details(video_url, proxy=None, retries=3)
# # #         print(json.dumps(details, indent=2, ensure_ascii=False))
# # #     except Exception as e:
# # #         logging.error(f"Failed to fetch video details: {str(e)}")

# # # from TikTokApi import TikTokApi
# # # import requests

# # # api = TikTokApi()

# # # video_url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"

# # # video = api.video(url=video_url)

# # # data = video.bytes()
# # # with open("video.mp4", "wb") as f:
# # #     f.write(data)

# # # from TikTokApi import TikTokApi

# # # # Initialize API
# # # with TikTokApi() as api:
# # #     # Replace with the TikTok video URL
# # #     url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"
    
# # #     video = api.video(url=url)

# # #     # Get direct video URL
# # #     video_url = video.bytes()
# # #     print("Video file (bytes):", len(video_url))

# # #     # Or get video data (contains image URLs, etc.)
# # #     info = video.info()
# # #     print("Cover image:", info['itemInfo']['itemStruct']['video']['cover'])
# # #     print("Video play URL:", info['itemInfo']['itemStruct']['video']['playAddr'])


# # import os
# # from PIL import Image
# # import google.generativeai as genai
# # from dotenv import load_dotenv

# # load_dotenv()

# # # Configure the API key (set via environment variable)
# # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # # Function to transcribe text from an image file object
# # def transcribe_image(image_file, model_name="gemini-2.0-flash"):
# #     # Initialize the model
# #     model = genai.GenerativeModel(model_name)
    
# #     # Load the image from the file object
# #     image = Image.open(image_file)
    
# #     # Prompt to extract/transcribe text from the image
# #     prompt = "Transcribe all visible content and text in this image accurately."
    
# #     # Generate content with text prompt and image
# #     response = model.generate_content([prompt, image])
  
# #     # Print the transcribed text
# #     transcribed_text = response.text
# #     print("Transcribed Text:\n", transcribed_text)
    
# #     return transcribed_text

# # # Example usage
# # if __name__ == "__main__":
# #     # Example with a file opened in binary mode
# #     with open("Uploads/youyube.png", "rb") as image_file:
# #         result = transcribe_image(image_file)

# # requirements: pip install yt-dlp

# # from yt_dlp import YoutubeDL

# # def get_instagram_video_url(reel_url: str) -> str:
# #     ydl_opts = {
# #         "format": "best",       # best quality available
# #         "quiet": True,          # suppress logs
# #         "skip_download": True,  # don’t actually download
# #     }

# #     with YoutubeDL(ydl_opts) as ydl:
# #         try:
# #             info = ydl.extract_info(reel_url, download=False)
# #         except Exception as e:
# #             print(f"Error extracting: {e}")
# #             return None

# #     # Sometimes 'formats' has multiple options (video only, audio only, etc.)
# #     if "formats" in info and info["formats"]:
# #         # Pick best format (last one is usually best quality with audio)
# #         best_format = info["formats"][-1]
# #         return best_format.get("url")
# #     else:
# #         return info.get("url")

# # if __name__ == "__main__":
# #     reel = "https://www.instagram.com/reels/DPGkCJAjNnN/?hl=en"  # replace with actual reel URL
# #     video_url = get_instagram_video_url(reel)
# #     if video_url:
# #         print("Direct video URL:", video_url)
# #     else:
# #         print("Failed to get video URL")

# from yt_dlp import YoutubeDL

# def get_instagram_post_info(post_url: str) -> dict:
#     """
#     Extracts Instagram post information including video URL, audio URL, thumbnail, and metadata.
    
#     Returns a dictionary with keys:
#     - video_url
#     - audio_url
#     - thumbnail_url
#     - title
#     - uploader
#     - description
#     - duration
#     - webpage_url
#     - full_formats (list of all available formats)
#     """
#     ydl_opts = {
#         "format": "best",       # best available format
#         "quiet": True,          # suppress logs
#         "skip_download": True,  # don’t actually download
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         try:
#             info = ydl.extract_info(post_url, download=False)
#         except Exception as e:
#             print(f"Error extracting: {e}")
#             return None

#     result = {
#         "video_url": None,
#         "audio_url": None,
#         "thumbnail_url": info.get("thumbnail"),
#         "title": info.get("title"),
#         "uploader": info.get("uploader"),
#         "description": info.get("description"),
#         "duration": info.get("duration"),
#         "webpage_url": info.get("webpage_url"),
#         "full_formats": info.get("formats", [])
#     }

#     # Find best video+audio format
#     if "formats" in info and info["formats"]:
#         # Prefer a format that has both video+audio
#         for f in reversed(info["formats"]):
#             if f.get("vcodec") != "none" and f.get("acodec") != "none":
#                 result["video_url"] = f.get("url")
#                 break

#         # Get audio-only format
#         for f in info["formats"]:
#             if f.get("vcodec") == "none" and f.get("acodec") != "none":
#                 result["audio_url"] = f.get("url")
#                 break

#     # fallback if direct url available
#     if not result["video_url"]:
#         result["video_url"] = info.get("url")

#     return result


# if __name__ == "__main__":
#     #reel = "https://www.instagram.com/reels/DPGkCJAjNnN/?hl=en"  # replace with actual reel/post URL
#     reel = "https://www.instagram.com/p/DHLK6zBtGHy/?utm_source=ig_web_copy_link&igsh=aGQ0Nno3OTEzZWlu"
#     post_info = get_instagram_post_info(reel)

#     if post_info:
#         print("\n=== Instagram Post Info ===")
#         print("Title:", post_info["title"])
#         print("Uploader:", post_info["uploader"])
#         print("Duration (s):", post_info["duration"])
#         print("Thumbnail:", post_info["thumbnail_url"])
#         print("Video URL:", post_info["video_url"])
#         print("Audio URL:", post_info["audio_url"])
#         print("Full formats available:", len(post_info["full_formats"]))
#     else:
#         print("Failed to extract post info")


# from yt_dlp import YoutubeDL
# from typing import Optional, Dict, Any, List


# def _pick_best_video_url(formats: List[Dict[str, Any]]) -> Optional[str]:
#     if not formats:
#         return None
#     # prefer formats that contain both video+audio
#     candidates = [f for f in formats if f.get("vcodec") != "none" and f.get("acodec") != "none"]
#     if not candidates:
#         # fallback to any video (may be video-only)
#         candidates = [f for f in formats if f.get("vcodec") != "none"]
#     if not candidates:
#         return None
#     # sort by height, tbr (total bit rate), filesize
#     candidates.sort(key=lambda f: (
#         f.get("height") or 0,
#         f.get("tbr") or 0,
#         f.get("filesize") or 0
#     ), reverse=True)
#     return candidates[0].get("url")


# def _pick_best_audio_url(formats: List[Dict[str, Any]]) -> Optional[str]:
#     if not formats:
#         return None
#     audios = [f for f in formats if (f.get("acodec") != "none") and (f.get("vcodec") == "none")]
#     if not audios:
#         # if no pure audio, maybe there's a combined format -> use its url as fallback
#         combined = [f for f in formats if f.get("acodec") != "none"]
#         if not combined:
#             return None
#         combined.sort(key=lambda f: (f.get("abr") or f.get("tbr") or 0, f.get("filesize") or 0), reverse=True)
#         return combined[0].get("url")
#     audios.sort(key=lambda f: (f.get("abr") or f.get("tbr") or 0, f.get("filesize") or 0), reverse=True)
#     return audios[0].get("url")


# def _pick_image_url(info: Dict[str, Any]) -> Optional[str]:
#     # direct url (single-image posts often put the image in 'url' or 'thumbnail')
#     for k in ("url", "thumbnail"):
#         u = info.get(k)
#         if u:
#             return u
#     # thumbnails list: pick the biggest by height
#     thumbs = info.get("thumbnails") or []
#     if thumbs:
#         thumbs_sorted = sorted(thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0), reverse=True)
#         return thumbs_sorted[0].get("url")
#     return None


# def extract_media_from_entry(entry: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
#     """
#     Normalize an entry (single image/video item) to a dict containing:
#     - media_type: 'video'|'image'|'unknown'
#     - video_url, audio_url (if any)
#     - thumbnail
#     - formats (raw formats list)
#     - metadata (id, ext, description, duration, etc)
#     """
#     media = {
#         "media_type": None,
#         "video_url": None,
#         "audio_url": None,
#         "image_url": None,
#         "thumbnail": entry.get("thumbnail"),
#         "formats": entry.get("formats") or [],
#         "metadata": {
#             "id": entry.get("id"),
#             "ext": entry.get("ext"),
#             "title": entry.get("title") or entry.get("id"),
#             "description": entry.get("description"),
#             "uploader": entry.get("uploader"),
#             "duration": entry.get("duration"),
#             "webpage_url": entry.get("webpage_url") or entry.get("original_url") or entry.get("url"),
#         },
#     }

#     # Video case: formats present or is_video True
#     if media["formats"]:
#         media["video_url"] = _pick_best_video_url(media["formats"])
#         media["audio_url"] = _pick_best_audio_url(media["formats"])
#         media["media_type"] = "video" if media["video_url"] else "audio_or_unknown"
#     else:
#         # No formats: probably an image post
#         img = _pick_image_url(entry)
#         if img:
#             media["image_url"] = img
#             media["media_type"] = "image"
#         else:
#             # fallback to url field
#             url = entry.get("url")
#             if url and url.endswith((".jpg", ".jpeg", ".png", ".webp")):
#                 media["image_url"] = url
#                 media["media_type"] = "image"
#             else:
#                 media["media_type"] = "unknown"

#     # If thumbnail is missing, try to set it from available image
#     if not media["thumbnail"]:
#         media["thumbnail"] = media["image_url"]

#     if debug:
#         print("DEBUG entry id:", entry.get("id"))
#         print("  media_type:", media["media_type"])
#         print("  video_url:", bool(media["video_url"]))
#         print("  audio_url:", bool(media["audio_url"]))
#         print("  image_url:", bool(media["image_url"]))
#         print("  thumbnail:", bool(media["thumbnail"]))
#         print("  #formats:", len(media["formats"]))

#     return media


# def get_instagram_media_urls(post_url: str,
#                              cookiefile: Optional[str] = None,
#                              username: Optional[str] = None,
#                              password: Optional[str] = None,
#                              debug: bool = False) -> Optional[Dict[str, Any]]:
#     """
#     Extract instagram post(s) info for /p/ (posts), /reel/ and others.
#     Returns a dict with top-level metadata and a 'media' list with one item per image/video in the post.
#     If the post is private or extraction fails, returns None.
#     You can pass cookiefile or username/password if required to access private posts.
#     """
#     ydl_opts = {
#         "format": "best",
#         "quiet": True,
#         "skip_download": True,
#         "noplaylist": True,  # we'll manually handle entries if returned
#         # some sites require a browser-like UA
#         "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
#         # allow extraction of otherwise-unplayable formats (sometimes helpful)
#         "allow_unplayable_formats": True,
#     }

#     if cookiefile:
#         ydl_opts["cookiefile"] = cookiefile
#     if username and password:
#         ydl_opts["username"] = username
#         ydl_opts["password"] = password

#     if debug:
#         print("Running yt-dlp with options:", {k: v for k, v in ydl_opts.items() if k != "password"})

#     with YoutubeDL(ydl_opts) as ydl:
#         try:
#             info = ydl.extract_info(post_url, download=False)
#         except Exception as e:
#             if debug:
#                 print("Extraction error:", e)
#             return None

#     # Normalize top-level metadata
#     result = {
#         "id": info.get("id"),
#         "title": info.get("title") or info.get("id"),
#         "uploader": info.get("uploader"),
#         "description": info.get("description"),
#         "webpage_url": info.get("webpage_url") or post_url,
#         "thumbnail": info.get("thumbnail"),
#         "duration": info.get("duration"),
#         "extractor": info.get("extractor"),
#         "media": []
#     }

#     # If yt-dlp returned 'entries' (carousel/gallery or playlist-like), iterate them
#     if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
#         entries = info.get("entries") or []
#         for entry in entries:
#             # some entries might be None (if extraction partially failed)
#             if not entry:
#                 continue
#             result["media"].append(extract_media_from_entry(entry, debug=debug))
#     else:
#         # single-item post — info itself is the entry
#         result["media"].append(extract_media_from_entry(info, debug=debug))

#     return result


# from yt_dlp import YoutubeDL
# from typing import Optional, Dict, Any, List


# def _pick_best_video_url(formats: List[Dict[str, Any]]) -> Optional[str]:
#     if not formats:
#         return None
#     candidates = [f for f in formats if f.get("vcodec") != "none" and f.get("acodec") != "none"]
#     if not candidates:
#         candidates = [f for f in formats if f.get("vcodec") != "none"]
#     if not candidates:
#         return None
#     candidates.sort(key=lambda f: (
#         f.get("height") or 0,
#         f.get("tbr") or 0,
#         f.get("filesize") or 0
#     ), reverse=True)
#     return candidates[0].get("url")


# def _pick_image_url(info: Dict[str, Any]) -> Optional[str]:
#     for k in ("url", "thumbnail"):
#         u = info.get(k)
#         if u:
#             return u
#     thumbs = info.get("thumbnails") or []
#     if thumbs:
#         thumbs_sorted = sorted(thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0), reverse=True)
#         return thumbs_sorted[0].get("url")
#     return None


# def extract_media_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     """Return a normalized single entry dict: {type, url}"""
#     if entry.get("formats"):  # video
#         return {
#             "type": "video",
#             "url": _pick_best_video_url(entry["formats"]),
#             "thumbnail": entry.get("thumbnail"),
#         }
#     img = _pick_image_url(entry)
#     if img:
#         return {
#             "type": "image",
#             "url": img,
#             "thumbnail": entry.get("thumbnail") or img,
#         }
#     return {"type": "unknown", "url": None, "thumbnail": entry.get("thumbnail")}


# def get_instagram_media_urls(post_url: str,
#                              cookiefile: Optional[str] = None,
#                              username: Optional[str] = None,
#                              password: Optional[str] = None,
#                              debug: bool = False) -> Optional[Dict[str, Any]]:
#     """
#     Extract Instagram media (images/videos) with flattened structure.
#     Returns dict:
#     {
#         "image_urls": [...],
#         "video_urls": [...],
#         "thumbnail": "...",
#         "text": "...",
#         "metadata": {...}
#     }
#     """
#     ydl_opts = {
#         "format": "best",
#         "quiet": True,
#         "skip_download": True,
#         "noplaylist": True,
#         "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
#     }
#     if cookiefile:
#         ydl_opts["cookiefile"] = cookiefile
#     if username and password:
#         ydl_opts["username"] = username
#         ydl_opts["password"] = password

#     with YoutubeDL(ydl_opts) as ydl:
#         try:
#             info = ydl.extract_info(post_url, download=False)
#         except Exception as e:
#             if debug:
#                 print("Extraction error:", e)
#             return None

#     # Collect entries
#     entries = []
#     if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
#         entries = info.get("entries") or []
#     else:
#         entries = [info]

#     image_urls, video_urls, thumbnails = [], [], []
#     for entry in entries:
#         if not entry:
#             continue
#         media = extract_media_from_entry(entry)
#         if media["type"] == "video" and media["url"]:
#             video_urls.append(media["url"])
#         elif media["type"] == "image" and media["url"]:
#             image_urls.append(media["url"])
#         if media.get("thumbnail"):
#             thumbnails.append(media["thumbnail"])

#     result = {
#         "image_urls": image_urls,
#         "video_urls": video_urls,
#         "thumbnail": thumbnails[0] if thumbnails else info.get("thumbnail"),
#         "text": info.get("description"),
#         "metadata": {
#             "id": info.get("id"),
#             "title": info.get("title"),
#             "uploader": info.get("uploader"),
#             "duration": info.get("duration"),
#             "webpage_url": info.get("webpage_url") or post_url,
#             "extractor": info.get("extractor"),
#         }
#     }
#     return result


# from yt_dlp import YoutubeDL
# from typing import Optional, Dict, Any, List


# def _pick_best_video_url(formats: List[Dict[str, Any]]) -> Optional[str]:
#     if not formats:
#         return None
#     candidates = [f for f in formats if f.get("vcodec") != "none" and f.get("acodec") != "none"]
#     if not candidates:
#         candidates = [f for f in formats if f.get("vcodec") != "none"]
#     if not candidates:
#         return None
#     candidates.sort(key=lambda f: (
#         f.get("height") or 0,
#         f.get("tbr") or 0,
#         f.get("filesize") or 0
#     ), reverse=True)
#     return candidates[0].get("url")


# # def _pick_image_url(info: Dict[str, Any]) -> Optional[str]:
# #     # Check common keys
# #     for k in ("display_url", "url", "thumbnail"):
# #         u = info.get(k)
# #         if u and isinstance(u, str) and u.startswith("http"):
# #             return u

# #     # Check thumbnails list
# #     thumbs = info.get("thumbnails") or []
# #     if thumbs:
# #         thumbs_sorted = sorted(
# #             thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0), reverse=True
# #         )
# #         for t in thumbs_sorted:
# #             if t.get("url"):
# #                 return t["url"]

# #     # Last resort: sometimes image url is in "formats" as an mp4 disguised as jpg
# #     if info.get("formats"):
# #         for f in info["formats"]:
# #             if f.get("url") and str(f.get("url")).endswith((".jpg", ".jpeg", ".png", ".webp")):
# #                 return f["url"]

# #     return None
# # def _pick_image_url(info: Dict[str, Any]) -> Optional[str]:
# #     # Check common top-level keys
# #     for k in ("display_url", "url", "thumbnail", "thumbnail_url", "media_url"):
# #         u = info.get(k)
# #         if u and isinstance(u, str) and u.startswith("http"):
# #             return u

# #     # Check thumbnails list
# #     thumbs = info.get("thumbnails") or []
# #     if thumbs:
# #         thumbs_sorted = sorted(
# #             thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0), reverse=True
# #         )
# #         for t in thumbs_sorted:
# #             if t.get("url"):
# #                 return t["url"]

# #     # Check nested media or entries
# #     for key in ("media", "entries"):
# #         if key in info:
# #             media_list = info[key] if isinstance(info[key], list) else [info[key]]
# #             for item in media_list:
# #                 if isinstance(item, dict):
# #                     for k in ("url", "display_url", "thumbnail", "thumbnail_url", "media_url"):
# #                         u = item.get(k)
# #                         if u and isinstance(u, str) and u.startswith("http"):
# #                             return u

# #     # Last resort: check formats for image-like URLs
# #     if info.get("formats"):
# #         for f in info["formats"]:
# #             if f.get("url") and str(f.get("url")).endswith((".jpg", ".jpeg", ".png", ".webp")):
# #                 return f["url"]

# #     return None

# def _pick_image_url(info: Dict[str, Any]) -> Optional[str]:
#     # Check common top-level keys
#     for k in ("display_url", "url", "thumbnail", "thumbnail_url", "media_url"):
#         u = info.get(k)
#         if u and isinstance(u, str) and u.startswith("http"):
#             return u

#     # Check carousel_media for /p/ posts with multiple images
#     carousel = info.get("carousel_media") or []
#     if carousel:
#         for item in carousel:
#             for k in ("media_url", "thumbnail_url", "url", "display_url"):
#                 u = item.get(k)
#                 if u and isinstance(u, str) and u.startswith("http"):
#                     return u

#     # Check thumbnails list
#     thumbs = info.get("thumbnails") or []
#     if thumbs:
#         thumbs_sorted = sorted(
#             thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0), reverse=True
#         )
#         for t in thumbs_sorted:
#             if t.get("url"):
#                 return t["url"]

#     # Check nested media or entries
#     for key in ("media", "entries"):
#         if key in info:
#             media_list = info[key] if isinstance(info[key], list) else [info[key]]
#             for item in media_list:
#                 if isinstance(item, dict):
#                     for k in ("url", "display_url", "thumbnail", "thumbnail_url", "media_url"):
#                         u = item.get(k)
#                         if u and isinstance(u, str) and u.startswith("http"):
#                             return u

#     # Last resort: check formats for image-like URLs
#     if info.get("formats"):
#         for f in info["formats"]:
#             if f.get("url") and str(f.get("url")).endswith((".jpg", ".jpeg", ".png", ".webp")):
#                 return f["url"]

#     return None



# def extract_media_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
#     if entry.get("formats"):  # likely video
#         return {
#             "type": "video",
#             "url": _pick_best_video_url(entry["formats"]),
#             "thumbnail": entry.get("thumbnail"),
#         }
#     img = _pick_image_url(entry)
#     if img:
#         return {"type": "image", "url": img, "thumbnail": entry.get("thumbnail") or img}
#     return {"type": "unknown", "url": None, "thumbnail": entry.get("thumbnail")}


# # def get_instagram_media_urls(post_url: str, debug: bool = False) -> Optional[Dict[str, Any]]:
# #     ydl_opts = {
# #         "format": "best",
# #         "quiet": True,
# #         "skip_download": True,
# #         "noplaylist": True,
# #         "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
# #     }

# #     with YoutubeDL(ydl_opts) as ydl:
# #         try:
# #             info = ydl.extract_info(post_url, download=False)
# #         except Exception as e:
# #             if debug:
# #                 print("Extraction error:", e)
# #             return None

# #     # Collect entries
# #     if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
# #         entries = info.get("entries") or []
# #     else:
# #         entries = [info]

# #     image_urls, video_urls, thumbnails = [], [], []
# #     for entry in entries:
# #         if not entry:
# #             continue
# #         media = extract_media_from_entry(entry)
# #         if media["type"] == "video" and media["url"]:
# #             video_urls.append(media["url"])
# #         elif media["type"] == "image" and media["url"]:
# #             image_urls.append(media["url"])
# #         if media.get("thumbnail"):
# #             thumbnails.append(media["thumbnail"])

# #     # Fallback: if no images found but top-level has thumbnails
# #     if not image_urls and info.get("thumbnails"):
# #         for t in info["thumbnails"]:
# #             if t.get("url"):
# #                 image_urls.append(t["url"])

# #     result = {
# #         "image_urls": image_urls,
# #         "video_urls": video_urls,
# #         "thumbnail": thumbnails[0] if thumbnails else info.get("thumbnail"),
# #         "text": info.get("description"),
# #         "metadata": {
# #             "id": info.get("id"),
# #             "title": info.get("title"),
# #             "uploader": info.get("uploader"),
# #             "duration": info.get("duration"),
# #             "webpage_url": info.get("webpage_url") or post_url,
# #             "extractor": info.get("extractor"),
# #         },
# #     }

# #     if debug:
# #         from pprint import pprint
# #         print("DEBUG raw info keys:", list(info.keys()))
# #         if "entries" in info:
# #             print(f"DEBUG entries: {len(info['entries'])}")
# #         pprint(result)

# #     return result

# def get_instagram_media_urls(post_url: str, debug: bool = False) -> Optional[Dict[str, Any]]:
#     ydl_opts = {
#         "format": "best",
#         "quiet": True,
#         "skip_download": True,
#         "noplaylist": True,
#         "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         try:
#             info = ydl.extract_info(post_url, download=False)
#         except Exception as e:
#             if debug:
#                 print("Extraction error:", str(e))
#             return {"error": f"Failed to extract info: {str(e)}"}

#     # Collect entries
#     if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
#         entries = info.get("entries") or []
#     else:
#         entries = [info]

#     image_urls, video_urls, thumbnails = [], [], []
#     for entry in entries:
#         if not entry:
#             continue
#         media = extract_media_from_entry(entry)
#         if media["type"] == "video" and media["url"]:
#             video_urls.append(media["url"])
#         elif media["type"] == "image" and media["url"]:
#             image_urls.append(media["url"])
#         if media.get("thumbnail"):
#             thumbnails.append(media["thumbnail"])

#     # Fallback: check top-level thumbnails
#     if not image_urls and info.get("thumbnails"):
#         for t in info["thumbnails"]:
#             if t.get("url"):
#                 image_urls.append(t["url"])

#     result = {
#         "image_urls": image_urls,
#         "video_urls": video_urls,
#         "thumbnail": thumbnails[0] if thumbnails else info.get("thumbnail"),
#         "text": info.get("description"),
#         "metadata": {
#             "id": info.get("id"),
#             "title": info.get("title"),
#             "uploader": info.get("uploader"),
#             "duration": info.get("duration"),
#             "webpage_url": info.get("webpage_url") or post_url,
#             "extractor": info.get("extractor"),
#         },
#     }

#     if debug:
#         from pprint import pprint
#         print("DEBUG raw info keys:", list(info.keys()))
#         if "entries" in info:
#             print(f"DEBUG entries: {len(info['entries'])}")
#         pprint(result)

#     # Add warning if no media found
#     if not image_urls and not video_urls:
#         result["warning"] = "No image or video URLs found. Post may be private, require authentication, or have an unsupported format."

#     return result


from yt_dlp import YoutubeDL
from typing import Optional, Dict, Any, List
import random


def _pick_best_video_url(formats: List[Dict[str, Any]]) -> Optional[str]:
    if not formats:
        return None
    candidates = [f for f in formats if f.get("vcodec") != "none" and f.get("acodec") != "none"]
    if not candidates:
        candidates = [f for f in formats if f.get("vcodec") != "none"]
    if not candidates:
        return None
    candidates.sort(key=lambda f: (
        f.get("height") or 0,
        f.get("tbr") or 0,
        f.get("filesize") or 0
    ), reverse=True)
    return candidates[0].get("url")


def _pick_image_url(info: Dict[str, Any]) -> Optional[str]:
    # Check common top-level keys
    for k in ("display_url", "url", "thumbnail", "thumbnail_url", "media_url", "image_url"):
        u = info.get(k)
        if u and isinstance(u, str) and u.startswith("http"):
            return u

    # Check carousel_media for /p/ posts with multiple images
    carousel = info.get("carousel_media") or []
    if carousel:
        for item in carousel:
            for k in ("media_url", "thumbnail_url", "url", "display_url", "image_url"):
                u = item.get(k)
                if u and isinstance(u, str) and u.startswith("http"):
                    return u

    # Check thumbnails list
    thumbs = info.get("thumbnails") or []
    if thumbs:
        thumbs_sorted = sorted(
            thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0), reverse=True
        )
        for t in thumbs_sorted:
            if t.get("url"):
                return t["url"]

    # Check nested media or entries
    for key in ("media", "entries"):
        if key in info:
            media_list = info[key] if isinstance(info[key], list) else [info[key]]
            for item in media_list:
                if isinstance(item, dict):
                    for k in ("url", "display_url", "thumbnail", "thumbnail_url", "media_url", "image_url"):
                        u = item.get(k)
                        if u and isinstance(u, str) and u.startswith("http"):
                            return u

    # Last resort: check formats for image-like URLs
    if info.get("formats"):
        for f in info["formats"]:
            if f.get("url") and str(f.get("url")).endswith((".jpg", ".jpeg", ".png", ".webp")):
                return f["url"]

    return None


def extract_media_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if entry.get("formats"):  # likely video
        return {
            "type": "video",
            "url": _pick_best_video_url(entry["formats"]),
            "thumbnail": entry.get("thumbnail"),
        }
    img = _pick_image_url(entry)
    if img:
        return {"type": "image", "url": img, "thumbnail": entry.get("thumbnail") or img}
    return {"type": "unknown", "url": None, "thumbnail": entry.get("thumbnail")}


def get_instagram_media_urls(post_url: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    # Rotate user agents to avoid rate limits
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    ]
    ydl_opts = {
        "format": "best",
        "quiet": False,  # Set to False to see detailed yt_dlp errors
        "skip_download": True,
        "noplaylist": True,
        "user_agent": random.choice(user_agents)
        #"cookiefile": "cookies.txt",  # Path to cookies file (see instructions below)
    }

    """
    To generate cookies.txt:
    1. Log in to Instagram in your browser.
    2. Install a browser extension like "Get cookies.txt" (Chrome/Firefox).
    3. Export cookies to a cookies.txt file.
    4. Place cookies.txt in the same directory as this script or update the path above.
    """

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(post_url, download=False)
        except Exception as e:
            if debug:
                print("Extraction error:", str(e))
            return {"error": f"Failed to extract info: {str(e)}"}

    # Collect entries
    entries = []
    if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
        entries = info.get("entries") or []
    else:
        entries = [info]

    image_urls, video_urls, thumbnails = [], [], []
    for entry in entries:
        if not entry:
            continue
        media = extract_media_from_entry(entry)
        if media["type"] == "video" and media["url"]:
            video_urls.append(media["url"])
        elif media["type"] == "image" and media["url"]:
            image_urls.append(media["url"])
        if media.get("thumbnail"):
            thumbnails.append(media["thumbnail"])

    # Check carousel_media for /p/ posts
    if not image_urls and info.get("carousel_media"):
        for item in info["carousel_media"]:
            img_url = _pick_image_url(item)
            if img_url and img_url not in image_urls:  # Avoid duplicates
                image_urls.append(img_url)

    # Fallback: check top-level thumbnails
    if not image_urls and info.get("thumbnails"):
        for t in info["thumbnails"]:
            if t.get("url") and t["url"] not in image_urls:  # Avoid duplicates
                image_urls.append(t["url"])

    result = {
        "image_urls": image_urls,
        "video_urls": video_urls,
        "thumbnail": thumbnails[0] if thumbnails else info.get("thumbnail"),
        "text": info.get("description") or info.get("title"),
        "metadata": {
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url") or post_url,
            "extractor": info.get("extractor"),
        },
    }

    if debug:
        from pprint import pprint
        print("DEBUG raw info keys:", list(info.keys()))
        if "entries" in info:
            print(f"DEBUG entries: {len(info['entries'])}")
        if "carousel_media" in info:
            print(f"DEBUG carousel_media: {len(info['carousel_media'])} items")
        pprint(result)

    # Add warning if no media found
    if not image_urls and not video_urls:
        result["warning"] = (
            "No image or video URLs found. Possible causes: "
            "1. Post is private (ensure cookies.txt is from an account following the poster). "
            "2. Instagram's structure changed (check debug output for new keys). "
            "3. Rate limits or CAPTCHAs (try a different user agent or proxy)."
        )

    return result


if __name__ == "__main__":
    # Example usage:
    # single image post: https://www.instagram.com/p/XXXXXXXXX/
    # reel: https://www.instagram.com/reel/XXXXXXXXX/
    #post = "https://www.tiktok.com/@mbedite/video/7534034640798108959?is_from_webapp=1&sender_device=pc" #tiktok
    #post = "https://www.instagram.com/p/DO0KpvcDZlK/?utm_source=ig_web_copy_link&igsh=MWFnaGx1bTYzcWtxZA=="
    #post = "https://www.instagram.com/p/DHLK6zBtGHy/?utm_source=ig_web_copy_link&igsh=aGQ0Nno3OTEzZWlu"
    #post = "https://www.facebook.com/share/v/1783QjXJEw/"
    #post = "https://www.facebook.com/share/p/1X4H8gGrjj/"
    #post = "https://www.facebook.com/share/p/1D7pXG7zNp/"
    #post = "https://www.facebook.com/share/v/19ewXDgBzW/"
    #post = "https://www.facebook.com/share/p/1F7V61XS7t/"
    post = "https://www.instagram.com/reel/DOf52aUjCXh/?utm_source=ig_web_copy_link&igsh=MXg3M3A4aHAxYnVqNQ=="
    info = get_instagram_media_urls(post, debug=False)
    if not info:
        print("Failed to extract — maybe the post is private or requires login/cookies.")
    else:
        from pprint import pprint
        pprint(info)