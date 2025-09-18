import requests
import os
import io
from dotenv import load_dotenv
from urllib.parse import urlparse
from engine.settings import ServiceSettings

load_dotenv()

class AudioTranscriber:
    def __init__(self):
        self.base_url = "https://api.deepgram.com/v1/listen"
        self.api_key = ServiceSettings.DEEPGRAM_API_KEY

    def transcribe_from_file(self, file_path: str):
        url = self.base_url

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/mp3"  # Change this depending on your file format
        }

        with open(file_path, "rb") as audio:
            response = requests.post(url, headers=headers, data=audio)

        print(response.status_code)
        if response.status_code == 200:
            result = response.json()
            # Extract transcript text
            transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript
        else:
            print("Error:", response.status_code, response.text)
            return None

    def transcribe_from_url(self, media_url: str):
        url = self.base_url

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "url": media_url
        }

        response = requests.post(url, headers=headers, json=payload)
        print(response.status_code)
        print(response.text)
        if response.status_code == 200:
            result = response.json()
            # Extract transcript text
            transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript
        else:
            print("Error:", response.status_code, response.text)
            return None
    
    def transcribe_from_buffer(self, buffer: io.BytesIO):
        """
        Transcribe audio directly from an in-memory BytesIO buffer.
        """
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/mp3"  # Adjust if necessary
        }

        buffer.seek(0)  # reset pointer
        response = requests.post(self.base_url, headers=headers, data=buffer)

        print(response.status_code)
        if response.status_code == 200:
            result = response.json()
            return result['results']['channels'][0]['alternatives'][0]['transcript']
        else:
            print("Error:", response.status_code, response.text)
            return None

    def transcribe(self, source):
        """
        Redirector function:
        - If source is BytesIO -> call transcribe_from_buffer
        - If source is a URL -> call transcribe_from_url
        - If source is a file path -> call transcribe_from_file
        """
        if isinstance(source, io.BytesIO):  # Handle buffer
            return self.transcribe_from_buffer(source)

        if isinstance(source, str):  # Handle URL or file path
            parsed = urlparse(source)

            if parsed.scheme in ("http", "https"):
                return self.transcribe_from_url(source)
            elif os.path.isfile(source):
                return self.transcribe_from_file(source)

        print("❌ Invalid source: must be BytesIO buffer, valid file path, or URL")
        return None
        
if __name__ == "__main__":
    # Example
    file_path = "downloads/video_WordPress Blog & n8n Automation for Beginners Step-by-Step Guide.mp4"
    
    with open(file_path, "rb") as f:
        buffer = io.BytesIO(f.read())
    
    transcriber = AudioTranscriber()

    #print(transcriber.transcribe(file_path))
    print(transcriber.transcribe(buffer))

