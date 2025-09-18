from urllib.parse import urlparse
import requests
import json
import os
from typing import Dict, Any, Optional
from app.core.config import settings
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class ImageProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key


    def compress_image(self, image_path, target_kb=190):
        """
        Compress an image into memory (BytesIO) so its size is <= target_kb.
        Returns a BytesIO buffer ready to send.
        """
        img = Image.open(image_path)

        # Convert RGBA/Palette -> RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        quality = 95
        buffer = io.BytesIO()

        while True:
            buffer.seek(0)
            buffer.truncate(0)
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_kb = buffer.tell() / 1024

            if size_kb <= target_kb or quality <= 20:
                break
            quality -= 5  # reduce quality step by step

        buffer.seek(0)
        print(f"✅ Compressed image in memory ({size_kb:.2f} KB)")
        return buffer

    def image_to_text(self, image_buffer: io.BytesIO):
        url = "https://api.api-ninjas.com/v1/imagetotext"
        headers = {"X-Api-Key": self.api_key}

        files = {"image": ("compressed.jpg", image_buffer, "image/jpeg")}
        response = requests.post(url, headers=headers, files=files)

        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None

        return response.json()

    def extract_text(self, ocr_result):
        """
        Takes the OCR JSON result from API Ninjas and returns
        only the extracted text as a single string.
        """
        if not ocr_result:
            return ""

        texts = [item["text"] for item in ocr_result if "text" in item]
        return " ".join(texts)

    def process_image_from_file(self, image_path: str, target_kb=190):
        """
        Full pipeline: compress image, send to API, extract text.
        """
        compressed_buffer = self.compress_image(image_path, target_kb=target_kb)
        ocr_result = self.image_to_text(compressed_buffer)
        extracted_text = self.extract_text(ocr_result)
        return extracted_text

    def compress_image_from_pil(self, img: Image.Image, target_kb=190):
        """
        Compress a PIL image into memory (BytesIO) so its size is <= target_kb.
        Returns a BytesIO buffer ready to send.
        """
        # Convert RGBA/Palette -> RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        quality = 95
        buffer = io.BytesIO()

        while True:
            buffer.seek(0)
            buffer.truncate(0)
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_kb = buffer.tell() / 1024

            if size_kb <= target_kb or quality <= 20:
                break
            quality -= 5  # reduce quality step by step

        buffer.seek(0)
        print(f"✅ Compressed image in memory ({size_kb:.2f} KB)")
        return buffer
    
    def process_image_from_url(self, image_url: str, target_kb=190):
        """
        Full pipeline: fetch image from URL, compress, send to API, extract text.
        """
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            print(f"Error fetching image: {response.status_code}")
            return None

        img = Image.open(io.BytesIO(response.content))
        compressed_buffer = self.compress_image_from_pil(img, target_kb=target_kb)
        ocr_result = self.image_to_text(compressed_buffer)
        return self.extract_text(ocr_result)

    def process(self, source: str, target_kb=190):
        parsed = urlparse(source)

        if parsed.scheme in ("http", "https"):  # URL
            return self.process_image_from_url(source, target_kb=target_kb)
        elif os.path.isfile(source):  # Local file
            return self.process_image_from_file(source, target_kb=target_kb)
        else:
            print("❌ Invalid source: must be a valid file path or URL")
            return None

if __name__ == "__main__":
    YOUR_API_KEY = settings.API_NINJAS_KEY
    #IMAGE_FILE = "https://instagram.fmfg1-1.fna.fbcdn.net/v/t51.2885-15/549468712_18484415512076000_6849235617603501235_n.jpg?stp=dst-jpg_e35_p1080x1080_sh0.08_tt6&_nc_ht=instagram.fmfg1-1.fna.fbcdn.net&_nc_cat=1&_nc_oc=Q6cZ2QEUdk8VEj5bBIO1lbtB_FtfNhQsiBCQUVH-yYeFXfZpKhZfoG2RISrqm6mSYFLeVfc&_nc_ohc=hbYB3UVdyakQ7kNvwH4p4Cm&_nc_gid=YK3mgRuowGdPvNk-Jlbukw&edm=AE-LrgUBAAAA&ccb=7-5&oh=00_AfZ0pKelOZdYK81083KUEO5DM0I_ZbfRvO-kE6CwHqyxYQ&oe=68CF844B&_nc_sid=8353fa"
    IMAGE_FILE = "downloads\pic2.png"
    image_processor = ImageProcessor(YOUR_API_KEY)

    res = image_processor.process(IMAGE_FILE, target_kb=190)
    print(res)
