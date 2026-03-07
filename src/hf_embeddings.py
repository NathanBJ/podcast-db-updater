# =============================================================================
# hf_embeddings.py - HuggingFace API Embedding Function for ChromaDB
# =============================================================================

import os
import time
from typing import List, Union, cast

from huggingface_hub import InferenceClient
import numpy as np

# ChromaDB expects embedding functions to inherit from EmbeddingFunction
# or implement the same interface
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class HuggingFaceAPIEmbedding(EmbeddingFunction[Documents]):
    """
    ChromaDB-compatible embedding function using HuggingFace Inference API.
    Properly implements the ChromaDB EmbeddingFunction interface.
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_token: str = None,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.max_retries = max_retries
        
        if not self.api_token:
            raise ValueError(
                "HuggingFace API token required. "
                "Set HF_TOKEN environment variable or pass api_token parameter. "
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        
        self.client = InferenceClient(token=self.api_token)
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return f"huggingface-api-{self.model_name}"
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Embed documents. This is the main method called by ChromaDB.
        
        Args:
            input: List of strings to embed
            
        Returns:
            List of embeddings, where each embedding is a list of floats
        """
        if not input:
            return []
        
        embeddings: Embeddings = []
        
        for text in input:
            embedding = self._embed_single(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                result = self.client.feature_extraction(
                    text,
                    model=self.model_name
                )
                
                # Convert result to list of floats
                return self._normalize_embedding(result)
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "loading" in error_str or "503" in error_str:
                    wait_time = 20
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                elif "rate" in error_str or "429" in error_str:
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                elif attempt < self.max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise RuntimeError(f"Failed to get embeddings: {e}")
        
        raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts")
    
    def _normalize_embedding(self, result) -> List[float]:
        """
        Normalize the API response to a flat list of floats.
        """
        arr = np.array(result)
        
        # Handle different response shapes
        if arr.ndim == 1:
            # Already a 1D vector
            pass
        elif arr.ndim == 2:
            # Token-level embeddings (num_tokens, embedding_dim) - mean pool
            arr = np.mean(arr, axis=0)
        elif arr.ndim == 3:
            # Batch of token embeddings (1, num_tokens, embedding_dim)
            arr = np.mean(arr[0], axis=0)
        else:
            raise ValueError(f"Unexpected embedding shape: {arr.shape}")
        
        # Ensure we return a flat list of Python floats
        return arr.astype(float).tolist()


def get_embedding_function(model_name: str = None) -> HuggingFaceAPIEmbedding:
    """Factory function to create embedding function."""
    return HuggingFaceAPIEmbedding(
        model_name=model_name or HuggingFaceAPIEmbedding.DEFAULT_MODEL
    )


if __name__ == "__main__":
    print("Testing HuggingFace API embeddings...")
    
    embed_fn = get_embedding_function()
    
    test_texts = [
        "Bonjour, comment allez-vous?",
        "Le temps est magnifique aujourd'hui.",
    ]
    
    print(f"Embedding {len(test_texts)} texts...")
    embeddings = embed_fn(test_texts)
    
    print(f"Got {len(embeddings)} embeddings")
    for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
        print(f"  [{i}] '{text[:30]}' -> type={type(emb).__name__}, len={len(emb)}")
        if emb:
            print(f"       First 5 values: {emb[:5]}")
            print(f"       Value types: {[type(v).__name__ for v in emb[:3]]}")
    
    print("✅ Success!")
    