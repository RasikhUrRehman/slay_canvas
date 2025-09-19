import os
import time
import json
import requests
from typing import List, Dict, Optional, Union, Generator, Iterator
from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv
from app.core.config import settings

load_dotenv()


class OpenRouterClient:
    def __init__(self, model: str, api_key: Optional[str] = None, system_prompt: str = "You are a helpful assistant."):
        """
        :param api_key: Your OpenRouter API key (or from env var OPENROUTER_API_KEY).
        :param system_prompt: Default system prompt for all chats.
        """
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. "
                             "Set OPENROUTER_API_KEY env var or pass api_key directly.")
        
        self.model = model or settings.OPENROUTER_MODEL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.default_system_prompt = system_prompt
        self.base_url = "https://openrouter.ai/api/v1"

    def list_models(self, limit: int = 20) -> List[str]:
        """Return a list of available model IDs."""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data[:limit]]
        except Exception as e:
            print("Error fetching models:", e)
            return []

    def chat(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 256,
        retries: int = 3,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a chat request.
        :param messages: Either a plain string or a list of {role, content} dicts.
        :param system_prompt: Override the default system prompt for this call.
        """
        if isinstance(messages, str):
            # Wrap into full conversation with system + user
            messages = [
                {"role": "system", "content": system_prompt or self.default_system_prompt},
                {"role": "user", "content": messages},
            ]
        else:
            # Insert system prompt at the start if not present
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt or self.default_system_prompt
                })

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                result = resp.choices[0].message.content or ""
                return result.strip() or "[Model returned no content]"

            except OpenAIError as e:
                print(f"Attempt {attempt+1}/{retries} failed with error: {e}")

                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    raise

    def chat_stream(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Send a streaming chat request using OpenAI client.
        :param messages: Either a plain string or a list of {role, content} dicts.
        :param system_prompt: Override the default system prompt for this call.
        :return: Iterator of text chunks
        """
        if isinstance(messages, str):
            # Wrap into full conversation with system + user
            messages = [
                {"role": "system", "content": system_prompt or self.default_system_prompt},
                {"role": "user", "content": messages},
            ]
        else:
            # Insert system prompt at the start if not present
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt or self.default_system_prompt
                })

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except OpenAIError as e:
            yield f"Error: {e}"

    def chat_stream_raw(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Send a streaming chat request using raw requests (as per your sample).
        :param messages: Either a plain string or a list of {role, content} dicts.
        :param system_prompt: Override the default system prompt for this call.
        :return: Iterator of text chunks
        """
        if isinstance(messages, str):
            # Wrap into full conversation with system + user
            messages = [
                {"role": "system", "content": system_prompt or self.default_system_prompt},
                {"role": "user", "content": messages},
            ]
        else:
            # Insert system prompt at the start if not present
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt or self.default_system_prompt
                })

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            yield f"Error: {e}"

    def chat_completion(
        self,
        messages: Union[str, List[Dict[str, str]]],
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Send a non-streaming chat completion request.
        :param messages: Either a plain string or a list of {role, content} dicts.
        :param model: Override the default model for this call.
        :param max_tokens: Maximum tokens to generate.
        :param temperature: Sampling temperature.
        :param system_prompt: Override the default system prompt for this call.
        :return: Full API response
        """
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": system_prompt or self.default_system_prompt},
                {"role": "user", "content": messages},
            ]
        else:
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": system_prompt or self.default_system_prompt
                })

        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": response.choices[0].message.role
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            }
            
        except OpenAIError as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Default system prompt
    or_client = OpenRouterClient(model="mistralai/mistral-small-3.1-24b-instruct:free", system_prompt="You are a wise philosopher.")

    print("\nDefault system prompt example:")
    print(or_client.chat(
        "What is the meaning of life?",
        system_prompt="You are a philosopher."
    ))