from abc import ABC, abstractmethod
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY


class Embedder(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        pass


class OpenAIEmbedder(Embedder):
    """OpenAI embedding implementation using LangChain"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=OPENAI_API_KEY
        )
        self.model = model
        # Set dimension based on model
        self._dimension = 1536 if model == "text-embedding-3-small" else 3072
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LangChain OpenAI wrapper"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        return self._dimension


class GeminiEmbedder(Embedder):
    """Gemini embedding implementation using LangChain"""
    
    def __init__(self, model: str = "gemini-embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=GEMINI_API_KEY
        )
        self.model = model
        self.output_dimensionality = 1536
        self._dimension = 1536
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LangChain Gemini wrapper with custom dimensionality"""
        try:
            return self.embeddings.embed_documents(
                texts, 
                output_dimensionality=self.output_dimensionality
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate Gemini embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        return self._dimension


def get_embedder(provider: str = None) -> Embedder:
    """Factory function to get the appropriate embedder based on config or provider argument
    
    Args:
        provider: Embedding provider ('openai', 'gemini', 'auto')
    """
    # Use provided provider or fall back to config
    embedding_provider = provider if provider is not None else EMBEDDING_PROVIDER
    
    # Handle "auto" provider selection
    if embedding_provider == "auto":
        if OPENAI_API_KEY:
            embedding_provider = "openai"
        elif GEMINI_API_KEY:
            embedding_provider = "gemini"
        else:
            raise ValueError("No API keys found for automatic provider selection")
    
    if embedding_provider == "openai":
        return OpenAIEmbedder()
    elif embedding_provider == "gemini":
        return GeminiEmbedder()
    else:
        raise ValueError(f"Invalid embedding provider: {embedding_provider}")