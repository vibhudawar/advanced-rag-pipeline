from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from config import COHERE_API_KEY
import cohere

class Reranker(ABC):
    """Abstract base class for document rerankers"""

    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 5, min_score: float = 0.1) -> List[Document]:
        """Rerank documents based on relevance to query

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Maximum number of documents to return
            min_score: Minimum relevance score threshold

        Returns:
            Filtered and reranked documents
        """
        pass


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker implementation using LangChain"""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)
        self.model_name = "MS-MARCO-MiniLM-L-6-v2"

    def rerank(self, query: str, documents: List[Document], top_k: int = 5, min_score: float = 0.1) -> List[Document]:
        """Rerank documents using Cross-Encoder

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Maximum number of documents to return
            min_score: Minimum relevance score threshold (default: 0.1)

        Returns:
            Filtered and reranked documents with score >= min_score
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # Filter and sort documents by score
        reranked_docs = []
        for score, doc in zip(scores, documents):
            if score < min_score:
                continue

            # Add score to metadata
            reranked_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    'rerank_score': float(score),
                    'rerank_model': self.model_name
                }
            )
            reranked_docs.append((score, reranked_doc))

        # Sort by score descending and return top_k
        reranked_docs.sort(key=lambda x: x[0], reverse=True)
        result = [doc for score, doc in reranked_docs[:top_k]]

        return result


class CohereReranker(Reranker):
    """Cohere reranker implementation using direct Cohere API v2"""
    
    def __init__(self, model: str = "rerank-v3.5"):
        cohere_api_key = COHERE_API_KEY
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        self.client = cohere.ClientV2(api_key=cohere_api_key)
        self.model = model
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5, min_score: float = 0.1) -> List[Document]:
        """Rerank documents using Cohere API v2

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Maximum number of documents to return
            min_score: Minimum relevance score threshold (default: 0.1)

        Returns:
            Filtered and reranked documents with score >= min_score
        """
        try:
            if not documents:
                return []

            # Extract document texts for reranking
            doc_texts = [doc.page_content for doc in documents]

            # Call Cohere rerank API
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k,
            )

            # Create reranked documents based on response, filtering by min_score
            reranked_docs = []
            for result in response.results:
                # Filter out documents with score below threshold
                if result.relevance_score < min_score:
                    continue

                original_doc = documents[result.index]
                # Create a copy of the document with updated metadata
                reranked_doc = Document(
                    page_content=original_doc.page_content,
                    metadata={
                        **original_doc.metadata,
                        'rerank_position': len(reranked_docs) + 1,
                        'rerank_model': self.model,
                        'rerank_score': result.relevance_score
                    }
                )
                reranked_docs.append(reranked_doc)

            return reranked_docs
            
        except Exception as e:
            # If reranking fails, return original documents
            print(f"⚠️ Cohere reranking failed: {str(e)}, returning original order")
            return documents[:top_k]


def get_reranker(provider: str = "auto") -> Reranker:
    """Factory function to get the appropriate reranker"""
    
    if provider == "cohere":
        try:
            return CohereReranker()
        except (ValueError, ImportError) as e:
            print(f"⚠️ Cohere reranker not available: {str(e)}")
    
    elif provider == "auto":
        try:
            return CohereReranker()
        except (ValueError, ImportError):
            pass
        try:
            return CrossEncoderReranker()
        except (ImportError, Exception):
            pass
    
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")


def rerank_documents(query: str, documents: List[Document], top_k: int = 5, provider: str = "auto") -> List[Document]:
    """Convenience function to rerank documents"""
    reranker = get_reranker(provider)
    return reranker.rerank(query, documents, top_k)
# 