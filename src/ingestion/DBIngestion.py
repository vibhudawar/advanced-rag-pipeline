from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_pinecone import PineconeVectorStore as LangChainPineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
import pinecone
import uuid
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from langsmith import traceable


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> None:
        """Create a new index"""
        pass
    
    @abstractmethod
    def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors in the index"""
        pass
    
    @abstractmethod
    def query_vectors(self, index_name: str, query_vector: List[float], 
                     top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query vectors from the index"""
        pass
    
    @abstractmethod
    def delete_index(self, index_name: str) -> None:
        """Delete an index"""
        pass
    
    @abstractmethod
    def list_indexes(self) -> List[str]:
        """List all available indexes"""
        pass
    
    @abstractmethod
    def add_documents(self, index_name: str, documents: List[Dict[str, Any]], embedder) -> None:
        """Add documents to the vector store using embedder"""
        pass
    
    @abstractmethod
    def similarity_search(self, index_name: str, query: str, embedder, top_k: int = 10, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search and return Document objects"""
        pass


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation using LangChain"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._langchain_stores = {}
    
    def create_index(self, index_name: str, dimension: int, metric: str = "cosine") -> None:
        """Create a new Pinecone index"""
        try:
            if index_name not in [index.name for index in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=pinecone.ServerlessSpec(
                        cloud='aws',
                        region=PINECONE_ENVIRONMENT
                    )
                )
            print(f"Index '{index_name}' created or already exists")
        except Exception as e:
            raise RuntimeError(f"Failed to create Pinecone index: {str(e)}")
    
    def _get_langchain_store(self, index_name: str, embedder):
        """Get or create a LangChain Pinecone vector store instance"""
        if index_name not in self._langchain_stores:
            self._langchain_stores[index_name] = LangChainPineconeVectorStore(
                index_name=index_name,
                embedding=embedder.embeddings  # Use the LangChain embeddings object
            )
        return self._langchain_stores[index_name]
    
    def add_documents(self, index_name: str, documents: List[Dict[str, Any]], embedder) -> None:
        """Add documents to the vector store using LangChain integration"""
        try:
            # Convert our document format to LangChain Document format
            langchain_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc['text'],
                    metadata=doc.get('metadata', {})
                )
                langchain_docs.append(langchain_doc)
            
            # Get LangChain vector store instance
            vector_store = self._get_langchain_store(index_name, embedder)
            
            # Add documents using LangChain
            vector_store.add_documents(langchain_docs)
            print(f"Added {len(langchain_docs)} documents to index '{index_name}'")
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Pinecone: {str(e)}")
    
    def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors to Pinecone index (fallback method for raw vectors)"""
        try:
            index = self.pc.Index(index_name)
            
            # Prepare vectors for Pinecone format
            formatted_vectors = []
            for vector_data in vectors:
                formatted_vectors.append({
                    'id': vector_data.get('id', str(uuid.uuid4())),
                    'values': vector_data['values'],
                    'metadata': vector_data.get('metadata', {})
                })
            
            # Batch upsert (Pinecone handles batching internally)
            index.upsert(vectors=formatted_vectors)
            print(f"Upserted {len(formatted_vectors)} vectors to index '{index_name}'")
            
        except Exception as e:
            raise RuntimeError(f"Failed to upsert vectors to Pinecone: {str(e)}")
    
    def query_vectors(self, index_name: str, query_vector: List[float], 
                     top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query vectors from Pinecone index"""
        try:
            index = self.pc.Index(index_name)
            
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
                include_values=False
            )
            
            return [
                {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                for match in results.matches
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to query Pinecone index: {str(e)}")
    
    @traceable(name="similarity_search")
    def similarity_search(self, index_name: str, query: str, embedder, top_k: int = 10,
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search using LangChain integration"""
        try:
            vector_store = self._get_langchain_store(index_name, embedder)

            # Perform similarity search
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict
            )

            # Convert to Document objects with score in metadata
            documents = []
            for doc, score in results:
                # Create a new Document with the score added to metadata
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, 'score': score}
                )
                documents.append(new_doc)

            return documents

        except Exception as e:
            print(f"[ERROR] Similarity search failed: {str(e)}")
            raise RuntimeError(f"Failed to perform similarity search: {str(e)}")
    
    def delete_index(self, index_name: str) -> None:
        """Delete a Pinecone index"""
        try:
            self.pc.delete_index(index_name)
            # Remove from cache
            if index_name in self._langchain_stores:
                del self._langchain_stores[index_name]
            print(f"Index '{index_name}' deleted")
        except Exception as e:
            raise RuntimeError(f"Failed to delete Pinecone index: {str(e)}")
    
    def list_indexes(self) -> List[str]:
        """List all Pinecone indexes"""
        try:
            return [index.name for index in self.pc.list_indexes()]
        except Exception as e:
            raise RuntimeError(f"Failed to list Pinecone indexes: {str(e)}")

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for a Pinecone index"""
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            print(f"[WARN] Failed to get index stats: {e}")
            return {"total_vector_count": 0, "dimension": 0, "namespaces": {}}


def get_vector_store() -> VectorStore:
    """Factory function to get the appropriate vector store"""
    return PineconeVectorStore()