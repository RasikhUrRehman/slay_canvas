# # Install required packages:
# # pip install httpx parsel jmespath selenium webdriver-manager

# import json
# import time
# from typing import Dict, Optional
# from parsel import Selector
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
# import jmespath
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def fetch_tiktok_video_details(video_url: str, proxy: Optional[str] = None, retries: int = 3) -> Dict:
#     """
#     Fetch details from a TikTok video URL using Selenium for dynamic content rendering.
    
#     Args:
#         video_url (str): The URL of the TikTok video, e.g., "https://www.tiktok.com/@username/video/123456789"
#         proxy (Optional[str]): Proxy server in format 'http://host:port' or None
#         retries (int): Number of retry attempts if the request fails
    
#     Returns:
#         dict: A dictionary containing video details like ID, description, video URL, cover image, download URL, author info, stats, etc.
    
#     Raises:
#         ValueError: If video data cannot be extracted after retries
#     """
#     # Set up Chrome options
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")  # Run in headless mode
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
#     if proxy:
#         chrome_options.add_argument(f'--proxy-server={proxy}')
    
#     # Initialize WebDriver
#     driver = None
#     attempt = 0
    
#     while attempt < retries:
#         try:
#             logging.info(f"Attempt {attempt + 1}/{retries} to fetch {video_url}")
#             driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#             driver.get(video_url)
            
#             # Wait for the page to load (adjust time as needed)
#             time.sleep(3)
            
#             # Get page source and parse
#             selector = Selector(driver.page_source)
#             script_data = selector.xpath("//script[@id='__UNIVERSAL_DATA_FOR_REHYDRATION__']/text()").get()
            
#             if not script_data:
#                 raise ValueError("Could not find video data in the page. The page might be blocked or structure changed.")
            
#             # Load JSON data
#             data = json.loads(script_data)
#             post_data = data["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"]["itemStruct"]
            
#             # Extract relevant fields with JMESPath
#             parsed_data = jmespath.search(
#                 """{
#                 id: id,
#                 desc: desc,
#                 createTime: createTime,
#                 video: video.{duration: duration, ratio: ratio, cover: cover, playAddr: playAddr, downloadAddr: downloadAddr, bitrate: bitrate},
#                 author: author.{id: id, uniqueId: uniqueId, nickname: nickname, avatarLarger: avatarLarger, signature: signature, verified: verified},
#                 stats: stats,
#                 locationCreated: locationCreated,
#                 diversificationLabels: diversificationLabels,
#                 suggestedWords: suggestedWords,
#                 contents: contents[].{textExtra: textExtra[].{hashtagName: hashtagName}}
#                 }""",
#                 post_data
#             )
            
#             # Handle video URLs
#             parsed_data['video_url'] = parsed_data['video']['playAddr']
#             parsed_data['download_url'] = parsed_data['video']['downloadAddr']
#             parsed_data['cover_image'] = parsed_data['video']['cover']
#             del parsed_data['video']['playAddr']
#             del parsed_data['video']['downloadAddr']
            
#             logging.info("Successfully extracted video details")
#             return parsed_data
        
#         except Exception as e:
#             logging.error(f"Error on attempt {attempt + 1}: {str(e)}")
#             attempt += 1
#             if attempt < retries:
#                 time.sleep(2 ** attempt)  # Exponential backoff
#             else:
#                 raise ValueError(f"Failed to fetch video data after {retries} attempts: {str(e)}")
        
#         finally:
#             if driver:
#                 driver.quit()
    
#     return {}

# # Example usage
# if __name__ == "__main__":
#     video_url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"  # Replace with your TikTok video URL
#     # proxy = "http://your-proxy:port"  # Optional: Add proxy if needed
#     try:
#         details = fetch_tiktok_video_details(video_url, proxy=None, retries=3)
#         print(json.dumps(details, indent=2, ensure_ascii=False))
#     except Exception as e:
#         logging.error(f"Failed to fetch video details: {str(e)}")

# from TikTokApi import TikTokApi
# import requests

# api = TikTokApi()

# video_url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"

# video = api.video(url=video_url)

# data = video.bytes()
# with open("video.mp4", "wb") as f:
#     f.write(data)

from TikTokApi import TikTokApi

# Initialize API
with TikTokApi() as api:
    # Replace with the TikTok video URL
    url = "https://www.tiktok.com/@yasirali.1919/video/7537649701600922887?is_from_webapp=1&sender_device=pc"
    
    video = api.video(url=url)

    # Get direct video URL
    video_url = video.bytes()
    print("Video file (bytes):", len(video_url))

    # Or get video data (contains image URLs, etc.)
    info = video.info()
    print("Cover image:", info['itemInfo']['itemStruct']['video']['cover'])
    print("Video play URL:", info['itemInfo']['itemStruct']['video']['playAddr'])