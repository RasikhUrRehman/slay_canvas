# import os
# import requests

# # Replace with your real token from https://nlpcloud.com
# NLPCLOUD_TOKEN = os.getenv("NLPCLOUD_TOKEN", "2903fbbc7893ce39a382eea93a34e4f1d632cb84")
# MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # must be exact from docs

# url = f"https://api.nlpcloud.io/v1/{MODEL_NAME}/embeddings"
# headers = {
#     "Authorization": f"Token {NLPCLOUD_TOKEN}",
#     "Content-Type": "application/json"
# }

# data = {
#     "sentences": [
#         "John Does works for Google."
#     ]
# }

# response = requests.post(url, headers=headers, json=data)

# print("Status code:", response.status_code)
# print("Raw response:", response.text)  # 👈 always check this first

# try:
#     result = response.json()
#     print("Parsed JSON:", result)
# except Exception as e:
#     print("Could not parse JSON:", e)

import os
import requests
from typing import List, Optional, Union, Dict
from dotenv import load_dotenv
load_dotenv()

class NLPCloudClient:
    def __init__(self, model_name: str, token: Optional[str] = None):
        """
        Initialize NLP Cloud client.
        
        :param model_name: The exact model name from NLP Cloud (e.g., 'paraphrase-multilingual-mpnet-base-v2').
        :param token: NLP Cloud API token. If not provided, reads from NLPCLOUD_TOKEN env var.
        """
        self.model_name = model_name
        self.token = token or os.getenv("NLPCLOUD_TOKEN")
        if not self.token:
            raise ValueError("NLP Cloud API token not provided and not found in environment.")
        
        self.base_url = f"https://api.nlpcloud.io/v1/{self.model_name}/embeddings"
        self.headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json"
        }

    def get_embeddings(self, sentences):
        """
        Get embeddings for one or more sentences.
        
        :param sentences: List of sentences (strings).
        :return: Parsed JSON response or None if error.
        """
        if isinstance(sentences, str):
            sentences = [sentences]  # allow single string
        
        payload = {"sentences": sentences}
        response = requests.post(self.base_url, headers=self.headers, json=payload)

        print("Status code:", response.status_code)
        print("Raw response:", response.text)

        try:
            return response.json()
        except Exception as e:
            print("Could not parse JSON:", e)
            return None

if __name__ == "__main__":
    client = NLPCloudClient("paraphrase-multilingual-mpnet-base-v2")

    result = client.get_embeddings("""This ired in 1843,[74] and subsequently, through a series of wars and treaties, the East India Company, and later, after the post-Sepoy Mutiny (1857–1858), direct rule by Queen Victoria of the British Empire, acquired most of the region.[75] Key conflicts included those against the Baloch Talpur dynasty, resolved by the Battle of Miani (1843) in Sindh,[76] the Anglo-Sikh Wars (1845–1849),[77] and the Anglo–Afghan Wars (1839–1919).[78] By 1893, all modern Pakistan was part of the British Indian Empire, and remained so until independence in 1947.[79]""")

    if result:
        print("Parsed JSON:", result)
    print(len(result['embeddings'][0]))
