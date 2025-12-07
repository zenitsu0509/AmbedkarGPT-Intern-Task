"""
LLM Client - Interface for local LLM using Ollama.

Provides a unified interface to interact with Mistral 7B via Ollama.
"""

import json
from typing import Optional, Dict, List, Generator
import requests


class LLMClient:
    """
    Client for interacting with local LLM via Ollama.
    
    Supports Mistral 7B and other models available through Ollama.
    """
    
    def __init__(
        self,
        model: str = "mistral:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_window: int = 4096
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Ollama model name (e.g., 'mistral:7b', 'llama3:8b')
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            context_window: Maximum context window size
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if our model is available
                model_available = any(
                    self.model in name or name in self.model
                    for name in model_names
                )
                
                if model_available:
                    print(f"✓ Connected to Ollama. Model '{self.model}' is available.")
                else:
                    print(f"⚠ Model '{self.model}' not found. Available: {model_names}")
                    print(f"  Run: ollama pull {self.model}")
            else:
                print(f"⚠ Ollama server responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"⚠ Cannot connect to Ollama at {self.base_url}")
            print("  Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"⚠ Error connecting to Ollama: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            if stream:
                return self._stream_generate(url, payload)
            else:
                # Longer timeout for first model load (300s = 5 min)
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be loading or overloaded."
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure it's running."
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _stream_generate(self, url: str, payload: dict) -> Generator[str, None, None]:
        """Stream the generation response."""
        payload["stream"] = True
        
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat with the LLM using message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Assistant's response
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            return f"Error in chat: {e}"
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Get embeddings from Ollama (if supported by model).
        
        Note: Most models don't support this directly. Use sentence-transformers instead.
        """
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding")
        except Exception as e:
            print(f"Embeddings not available: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the LLM is available and responding."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except:
            pass
        return []
