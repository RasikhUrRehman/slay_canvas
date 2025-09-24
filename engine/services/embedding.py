
# import os
# import requests
# from typing import List, Optional, Union, Dict
# from dotenv import load_dotenv
# load_dotenv()

# class NLPCloudClient:
#     def __init__(self, model_name: str, token: Optional[str] = None):
#         """
#         Initialize NLP Cloud client.
        
#         :param model_name: The exact model name from NLP Cloud (e.g., 'paraphrase-multilingual-mpnet-base-v2').
#         :param token: NLP Cloud API token. If not provided, reads from NLPCLOUD_TOKEN env var.
#         """
#         self.model_name = model_name
#         self.token = token or os.getenv("NLPCLOUD_TOKEN")
#         if not self.token:
#             raise ValueError("NLP Cloud API token not provided and not found in environment.")
        
#         self.base_url = f"https://api.nlpcloud.io/v1/{self.model_name}/embeddings"
#         self.headers = {
#             "Authorization": f"Token {self.token}",
#             "Content-Type": "application/json"
#         }

#     def get_embeddings(self, sentences):
#         """
#         Get embeddings for one or more sentences.
        
#         :param sentences: List of sentences (strings).
#         :return: Parsed JSON response or None if error.
#         """
#         if isinstance(sentences, str):
#             sentences = [sentences]  # allow single string
        
#         payload = {"sentences": sentences}
#         response = requests.post(self.base_url, headers=self.headers, json=payload)

#         print("Status code:", response.status_code)
#         print("Raw response:", response.text)

#         try:
#             return response.json()
#         except Exception as e:
#             print("Could not parse JSON:", e)
#             return None

# if __name__ == "__main__":
#     client = NLPCloudClient("paraphrase-multilingual-mpnet-base-v2")

#     result = client.get_embeddings("""This ired in 1843,[74] and subsequently, through a series of wars and treaties, the East India Company, and later, after the post-Sepoy Mutiny (1857–1858), direct rule by Queen Victoria of the British Empire, acquired most of the region.[75] Key conflicts included those against the Baloch Talpur dynasty, resolved by the Battle of Miani (1843) in Sindh,[76] the Anglo-Sikh Wars (1845–1849),[77] and the Anglo–Afghan Wars (1839–1919).[78] By 1893, all modern Pakistan was part of the British Indian Empire, and remained so until independence in 1947.[79]""")

#     if result:
#         print("Parsed JSON:", result)
#     print(len(result['embeddings'][0]))

# import os
# from huggingface_hub import InferenceClient
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# from app.core.config import settings
# load_dotenv()

# class EmbeddingService:
#     """
#     Wrapper around Hugging Face InferenceClient for generating embeddings.
#     """

#     def __init__(
#         self,
#         api_key: str | None = None,
#         provider: str = "nebius",
#         model: str = "Qwen/Qwen3-Embedding-8B",
#     ):
#         """
#         Initialize the embedding service.

#         Args:
#             api_key (str | None): Hugging Face API token. Defaults to environment variable HF_TOKEN.
#             provider (str): Inference provider (default: 'nebius').
#             model (str): Model ID to use for embeddings.
#         """
#         self.api_key = api_key or settings.HF_TOKEN
#         if not self.api_key:
#             raise ValueError("API key not provided. Set HF_TOKEN environment variable or pass it explicitly.")

#         self.model = model
#         self.client = InferenceClient(provider=provider, api_key=self.api_key)

#     def get_embedding(self, text: str) -> dict:
#             """
#             Generate an embedding vector for a given text.

#             Args:
#                 text (str): Input text.

#             Returns:
#                 dict: {"embeddings": [float, ...]}
#             """
#             result = self.client.feature_extraction(text, model=self.model)
#             result = result[0]

#             # Hugging Face returns a nested list [[...]], so flatten it
#             embedding = result[0] if isinstance(result, list) and isinstance(result[0], list) else result

#             return {"embeddings": embedding}


# if __name__ == "__main__":
#     # Example usage
#     service = EmbeddingService()
#     embedding = service.get_embedding("""Today is a sunny day and I will get some ice cream.""")
#     print(f"Embedding length: {len(embedding['embeddings'])}")
#     print(embedding['embeddings'][:10])


import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class EmbeddingService:
    """
    Wrapper around OpenAI API for generating embeddings.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize the embedding service.

        Args:
            api_key (str | None): OpenAI API key. Defaults to environment variable OPENAI_API_KEY.
            model (str): Model ID to use for embeddings.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "test")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable or pass it explicitly."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def get_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generate an embedding vector for a given text.

        Args:
            text (str): Input text.

        Returns:
            dict: {"embeddings": [float, ...]}
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        embedding = response.data[0].embedding
        return {"embeddings": embedding}


if __name__ == "__main__":
    # Example usage
    service = EmbeddingService()
    embedding = service.get_embedding("Today is a sunny day and I will get some ice cream.")
    print(f"Embedding length: {len(embedding['embeddings'])}")
    print(embedding['embeddings'][:10])

