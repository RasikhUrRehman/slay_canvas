# # # Install required packages:
# # # pip install httpx parsel jmespath selenium webdriver-manager

# # import json
# # import time
# # from typing import Dict, Optional
# # from parsel import Selector
# # from selenium import webdriver
# # from selenium.webdriver.chrome.service import Service
# # from webdriver_manager.chrome import ChromeDriverManager
# # from selenium.webdriver.chrome.options import Options
# # import jmespath
# # import logging

# # # Set up logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # def fetch_tiktok_video_details(video_url: str, proxy: Optional[str] = None, retries: int = 3) -> Dict:
# #     """
# #     Fetch details from a TikTok video URL using Selenium for dynamic content rendering.
    
# #     Args:
# #         video_url (str): The URL of the TikTok video, e.g., "https://www.tiktok.com/@username/video/123456789"
# #         proxy (Optional[str]): Proxy server in format 'http://host:port' or None
# #         retries (int): Number of retry attempts if the request fails
    
# #     Returns:
# #         dict: A dictionary containing video details like ID, description, video URL, cover image, download URL, author info, stats, etc.
    
# #     Raises:
# #         ValueError: If video data cannot be extracted after retries
# #     """
# #     # Set up Chrome options
# #     chrome_options = Options()
# #     chrome_options.add_argument("--headless")  # Run in headless mode
# #     chrome_options.add_argument("--disable-gpu")
# #     chrome_options.add_argument("--no-sandbox")
# #     chrome_options.add_argument("--disable-dev-shm-usage")
# #     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
# #     if proxy:
# #         chrome_options.add_argument(f'--proxy-server={proxy}')
    
# #     # Initialize WebDriver
# #     driver = None
# #     attempt = 0
    
# #     while attempt < retries:
# #         try:
# #             logging.info(f"Attempt {attempt + 1}/{retries} to fetch {video_url}")
# #             driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
# #             driver.get(video_url)
            
# #             # Wait for the page to load (adjust time as needed)
# #             time.sleep(3)
            
# #             # Get page source and parse
# #             selector = Selector(driver.page_source)
# #             script_data = selector.xpath("//script[@id='__UNIVERSAL_DATA_FOR_REHYDRATION__']/text()").get()
            
# #             if not script_data:
# #                 raise ValueError("Could not find video data in the page. The page might be blocked or structure changed.")
            
# #             # Load JSON data
# #             data = json.loads(script_data)
# #             post_data = data["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"]["itemStruct"]
            
# #             # Extract relevant fields with JMESPath
# #             parsed_data = jmespath.search(
# #                 """{
# #                 id: id,
# #                 desc: desc,
# #                 createTime: createTime,
# #                 video: video.{duration: duration, ratio: ratio, cover: cover, playAddr: playAddr, downloadAddr: downloadAddr, bitrate: bitrate},
# #                 author: author.{id: id, uniqueId: uniqueId, nickname: nickname, avatarLarger: avatarLarger, signature: signature, verified: verified},
# #                 stats: stats,
# #                 locationCreated: locationCreated,
# #                 diversificationLabels: diversificationLabels,
# #                 suggestedWords: suggestedWords,
# #                 contents: contents[].{textExtra: textExtra[].{hashtagName: hashtagName}}
# #                 }""",
# #                 post_data
# #             )
            
# #             # Handle video URLs
# #             parsed_data['video_url'] = parsed_data['video']['playAddr']
# #             parsed_data['download_url'] = parsed_data['video']['downloadAddr']
# #             parsed_data['cover_image'] = parsed_data['video']['cover']
# #             del parsed_data['video']['playAddr']
# #             del parsed_data['video']['downloadAddr']
            
# #             logging.info("Successfully extracted video details")
# #             return parsed_data
        
# #         except Exception as e:
# #             logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
# #             attempt += 1
# #             if attempt < retries:
# #                 time.sleep(2 ** attempt)  # Exponential backoff
# #             else:
# #                 raise ValueError(f"Failed to fetch video data after {retries} attempts: {str(e)}")
        
# #         finally:
# #             if driver:
# #                 driver.quit()
    
# #     return {}

# # # Example usage
# # if __name__ == "__main__":
# #     video_url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"  # Replace with your TikTok video URL
# #     # proxy = "http://your-proxy:port"  # Optional: Add proxy if needed
# #     try:
# #         details = fetch_tiktok_video_details(video_url, proxy=None, retries=3)
# #         print(json.dumps(details, indent=2, ensure_ascii=False))
# #     except Exception as e:
# #         logging.error(f"Failed to fetch video details: {str(e)}")

# # from TikTokApi import TikTokApi
# # import requests

# # api = TikTokApi()

# # video_url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"

# # video = api.video(url=video_url)

# # data = video.bytes()
# # with open("video.mp4", "wb") as f:
# #     f.write(data)

# from TikTokApi import TikTokApi

# # Initialize API
# with TikTokApi() as api:
#     # Replace with the TikTok video URL
#     url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"
    
#     video = api.video(url=url)

#     # Get direct video URL
#     video_url = video.bytes()
#     print("Video file (bytes):", len(video_url))

#     # Or get video data (contains image URLs, etc.)
#     info = video.info()
#     print("Cover image:", info['itemInfo']['itemStruct']['video']['cover'])
#     print("Video play URL:", info['itemInfo']['itemStruct']['video']['playAddr'])

# import requests
# from urllib.parse import quote_plus

# def tiktok_oembed(post_url):
#     # post_url: full tiktok post URL, e.g. "https://www.tiktok.com/@username/video/1234567890"
#     endpoint = "https://www.tiktok.com/oembed?url=" + quote_plus(post_url)
#     r = requests.get(endpoint, headers={"User-Agent": "Mozilla/5.0"})
#     r.raise_for_status()
#     return r.json()

# if __name__ == "__main__":
#     url = "https://www.tiktok.com/@scout2015/video/6718335390845095173"  # replace with any public post URL
#     data = tiktok_oembed(url)
#     print("oEmbed JSON keys:", data.keys())
#     print("Thumbnail (image) URL:", data.get("thumbnail_url") or data.get("thumbnail_url"))
#     print("Author:", data.get("author_name"))
#     print("Title/HTML:", data.get("title") or data.get("html"))


# from yt_dlp import YoutubeDL


# def get_fresh_video_url(post_url):
#     opts = {"quiet": True, "skip_download": True}
#     with YoutubeDL(opts) as ydl:
#         info = ydl.extract_info(post_url, download=False)
#     # pick best format
#     best = max(info["formats"], key=lambda f: f.get("height") or 0)
#     return best["url"]


# def extract_tiktok(post_url):
#     ydl_opts = {
#         "quiet": True,
#         "skip_download": True,
#         "forcejson": True,
#         "nocheckcertificate": True,
#     }
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(post_url, download=False)
#     # info contains lots of fields
#     return {
#         "id": info.get("id"),
#         "uploader": info.get("uploader"),
#         "title": info.get("title"),
#         "description": info.get("description"),
#         "upload_date": info.get("upload_date"),
#         "thumbnail": info.get("thumbnail"),
#         "formats": info.get("formats"),  # formats list with direct URLs
#         "webpage_url": info.get("webpage_url"),
#     }

# if __name__ == "__main__":
#     url = "https://www.tiktok.com/@scout2015/video/6718335390845095173"
#     print(get_fresh_video_url(url))

#     meta = extract_tiktok(url)
#     print("Thumbnail:", meta["thumbnail"])
#     # find a direct video URL (choose best format)
#     formats = meta["formats"] or []
#     # formats entries have 'url' and 'format_id' etc.
#     if formats:
#         best_video = formats[-1].get("url")  # often last is best; inspect formats to choose
#         print("Direct video URL (sample):", best_video)
#     print("Description / Text:", meta["description"][:300])



# from yt_dlp import YoutubeDL

# def get_video_with_headers(post_url):
#     ydl_opts = {
#         "quiet": True,
#         "skip_download": True,
#     }
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(post_url, download=False)
#     best = max(info["formats"], key=lambda f: f.get("height") or 0)
#     return {
#         "url": best["url"],
#         "http_headers": info.get("http_headers", {})  # needed headers
#     }

# if __name__ == "__main__":
#     video = get_video_with_headers("https://www.tiktok.com/@scout2015/video/6718335390845095173")
#     print("Video URL:", video["url"])
#     print("Headers:", video["http_headers"])


# from tiktok_scraper import TikTokScraper
# import os

# def download_tiktok_video(video_url, output_dir):
#     """
#     Downloads a TikTok video from the given URL and saves it to the specified directory.

#     Args:
#         video_url (str): URL of the TikTok video.
#         output_dir (str): Destination directory where the downloaded video will be saved.
#     """
#     try:
#         scraper = TikTokScraper()
#         video_path = scraper.download_video(video_url, output_dir)
#         print("Downloaded video:", video_path)

#         # Additional functionality: Rename the downloaded video
#         video_name = os.path.basename(video_path)
#         new_video_name = f"tiktok_{video_name}"
#         new_video_path = os.path.join(output_dir, new_video_name)
#         os.rename(video_path, new_video_path)
#         print("Renamed video:", new_video_path)

#         # Additional functionality: Get video metadata
#         video_metadata = scraper.get_video_metadata(video_url)
#         print("Video metadata:", video_metadata)

#     except Exception as e:
#         print("Error:", str(e))

# # Example usage
# if __name__ == "__main__":
#     tiktok_url = "https://www.tiktok.com/@mellapuspitaa/video/7548321802850733320?is_from_webapp=1&sender_device=pc"
#     download_directory = "/path/to/directory"
#     download_tiktok_video(tiktok_url, download_directory)

# import yt_dlp
# import os
# import re
# from typing import Optional, Dict, Any
# from datetime import datetime

# class TikTokDownloader:
#     def __init__(self, save_path: str = 'tiktok_videos'):
#         """
#         Initialize TikTok downloader with configurable save path
        
#         Args:
#             save_path (str): Directory where videos will be saved
#         """
#         self.save_path = save_path
#         self.create_save_directory()
    
#     def create_save_directory(self) -> None:
#         """Create the save directory if it doesn't exist"""
#         if not os.path.exists(self.save_path):
#             os.makedirs(self.save_path)
    
#     @staticmethod
#     def validate_url(url: str) -> bool:
#         """
#         Validate if the provided URL is a TikTok URL
        
#         Args:
#             url (str): URL to validate
            
#         Returns:
#             bool: True if valid, False otherwise
#         """
#         tiktok_pattern = r'https?://((?:vm|vt|www)\.)?tiktok\.com/.*'
#         return bool(re.match(tiktok_pattern, url))
    
#     @staticmethod
#     def progress_hook(d: Dict[str, Any]) -> None:
#         """
#         Hook to display download progress
        
#         Args:
#             d (Dict[str, Any]): Progress information dictionary
#         """
#         if d['status'] == 'downloading':
#             progress = d.get('_percent_str', 'N/A')
#             speed = d.get('_speed_str', 'N/A')
#             eta = d.get('_eta_str', 'N/A')
#             print(f"Downloading: {progress} at {speed} ETA: {eta}", end='\r')
#         elif d['status'] == 'finished':
#             print("\nDownload completed, finalizing...")
    
#     def get_filename(self, video_url: str, custom_name: Optional[str] = None) -> str:
#         """
#         Generate filename for the video
        
#         Args:
#             video_url (str): Video URL
#             custom_name (Optional[str]): Custom name for the video file
            
#         Returns:
#             str: Generated filename
#         """
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         if custom_name:
#             return f"{custom_name}_{timestamp}.mp4"
#         return f"tiktok_{timestamp}.mp4"
    
#     def download_video(self, video_url: str, custom_name: Optional[str] = None) -> Optional[str]:
#         """
#         Download TikTok video
        
#         Args:
#             video_url (str): URL of the TikTok video
#             custom_name (Optional[str]): Custom name for the video file
            
#         Returns:
#             Optional[str]: Path to downloaded file if successful, None otherwise
#         """
#         if not self.validate_url(video_url):
#             print("Error: Invalid TikTok URL")
#             return None

#         filename = self.get_filename(video_url, custom_name)
#         output_path = os.path.join(self.save_path, filename)
        
#         ydl_opts = {
#             'outtmpl': output_path,
#             'format': 'best',
#             'noplaylist': True,
#             'quiet': False,
#             'progress_hooks': [self.progress_hook],
#             'cookiesfrombrowser': ('chrome',),  # Use Chrome cookies for authentication
#             'extractor_args': {'tiktok': {'webpage_download': True}},
#             'http_headers': {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#         }

#         try:
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([video_url])
#                 print(f"\nVideo successfully downloaded: {output_path}")
#                 return output_path
                
#         except yt_dlp.utils.DownloadError as e:
#             print(f"Error downloading video: {str(e)}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {str(e)}")
        
#         return None

# # Example usage
# if __name__ == "__main__":
#     downloader = TikTokDownloader(save_path='downloaded_tiktoks')
#     video_url = "https://www.tiktok.com/@zachking/video/6768504823336815877"
    
#     # Basic usage
#     downloader.download_video(video_url)
    
#     # With custom filename
#     downloader.download_video(video_url, custom_name="zach_king_magic")
