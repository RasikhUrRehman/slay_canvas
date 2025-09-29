
# def get_instagram_media_urls(post_url: str) -> dict:
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
