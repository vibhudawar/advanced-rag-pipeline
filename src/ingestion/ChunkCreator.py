from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from src.utils.AgenticChunkerHelper import (
    PropositionExtractor,
    ChunkManager,
    LLMChunkDecider
)


class Chunker(ABC):
    """Abstract base class for text chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        pass


class RecursiveChunker(Chunker):
    """Recursive character text splitter chunker"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text using recursive character splitting"""
        if metadata is None:
            metadata = {}
        
        chunks = self.splitter.split_text(text)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            })
            
            chunked_documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return chunked_documents


class SemanticChunkerFunction(Chunker):
    """Semantic chunking based on sentence similarity using embeddings"""

    def __init__(self, embedder):
        """
        Initialize semantic chunker with embedder

        Args:
            embedder: Embedder instance (from src.ingestion.EmbeddingCreator)
        """
        self.embedder = embedder
        self.splitter = SemanticChunker(embeddings=embedder, breakpoint_threshold_type="percentile")

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text using semantic similarity between sentences"""
        if metadata is None:
            metadata = {}

        chunks = self.splitter.split_text(text)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks),
                'chunking_method': 'semantic',
            })

            chunked_documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })

        return chunked_documents


class AgenticChunker(Chunker):
    """
    Agentic chunking using LLM to reason about chunk creation.

    This chunker uses an LLM to intelligently group related propositions
    into semantically coherent chunks with automatically generated titles
    and summaries.
    """

    def __init__(self, llm):
        """
        Initialize agentic chunker with LLM.

        Args:
            llm: LangChain chat model instance (e.g., ChatOpenAI, ChatGoogleGenerativeAI)
        """
        self.proposition_extractor = PropositionExtractor(llm)
        self.chunk_manager = ChunkManager(id_truncate_limit=5)
        self.llm_decider = LLMChunkDecider(llm, id_truncate_limit=5)
        self.generate_new_metadata = True

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text using LLM-guided agentic chunking.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if metadata is None:
            metadata = {}

        # Reset chunk manager for new document
        self.chunk_manager = ChunkManager(id_truncate_limit=5)

        # 1. Extract propositions from text
        propositions = self.proposition_extractor.extract_propositions(text)

        if not propositions:
            # Fallback: return entire text as single chunk
            return self._create_fallback_chunk(text, metadata)

        # 2. Process each proposition
        for proposition in propositions:
            self._process_proposition(proposition)

        # 3. Convert internal chunks to standard format
        return self._format_chunks(metadata)

    def _process_proposition(self, proposition: str):
        """Process a single proposition by adding it to a chunk."""
        if self.chunk_manager.is_empty():
            # First proposition: create new chunk
            self._create_chunk_for_proposition(proposition)
            return

        # Find relevant existing chunk
        chunk_outline = self.chunk_manager.get_chunk_outline()
        chunk_id = self.llm_decider.find_relevant_chunk(proposition, chunk_outline)

        if chunk_id:
            # Add to existing chunk
            self._add_to_existing_chunk(chunk_id, proposition)
        else:
            # Create new chunk
            self._create_chunk_for_proposition(proposition)

    def _create_chunk_for_proposition(self, proposition: str):
        """Create a new chunk for a proposition."""
        summary = self.llm_decider.generate_new_chunk_summary(proposition)
        title = self.llm_decider.generate_new_chunk_title(summary)
        self.chunk_manager.create_chunk(proposition, summary, title)

    def _add_to_existing_chunk(self, chunk_id: str, proposition: str):
        """Add a proposition to an existing chunk and update metadata."""
        self.chunk_manager.add_proposition_to_chunk(chunk_id, proposition)

        if self.generate_new_metadata:
            chunk = self.chunk_manager.get_chunk(chunk_id)
            if chunk:
                # Update chunk metadata
                new_summary = self.llm_decider.update_chunk_summary(
                    chunk['propositions'],
                    chunk['summary']
                )
                new_title = self.llm_decider.update_chunk_title(
                    chunk['propositions'],
                    new_summary,
                    chunk['title']
                )
                self.chunk_manager.update_chunk_metadata(chunk_id, new_summary, new_title)

    def _format_chunks(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format internal chunks into standard output format."""
        chunked_documents = []
        all_chunks = self.chunk_manager.get_all_chunks()
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x['chunk_index'])

        for i, chunk in enumerate(sorted_chunks):
            # Combine all propositions in the chunk
            chunk_text = ' '.join(chunk['propositions'])

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk_text),
                'total_chunks': len(sorted_chunks),
                'chunking_method': 'agentic',
                'chunk_title': chunk.get('title', ''),
                'chunk_summary': chunk.get('summary', ''),
                'proposition_count': len(chunk['propositions'])
            })

            chunked_documents.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })

        return chunked_documents


def get_chunker(strategy: str = "recursive", **kwargs) -> Chunker:
    """
    Factory function to get appropriate chunker

    Args:
        strategy: Chunking strategy ('recursive', 'semantic', 'agentic')
        **kwargs: Strategy-specific arguments
            - For recursive: chunk_size, chunk_overlap
            - For semantic: embedder (required)
            - For agentic: llm (required)

    Returns:
        Chunker instance
    """
    if strategy == "recursive":
        # Extract only recursive-specific kwargs
        recursive_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['chunk_size', 'chunk_overlap']
        }
        return RecursiveChunker(**recursive_kwargs)

    elif strategy == "semantic":
        # Semantic chunker requires embedder
        embedder = kwargs.get('embedder')
        if embedder is None:
            raise ValueError("Semantic chunking requires 'embedder' parameter")
        return SemanticChunkerFunction(embedder=embedder)

    elif strategy == "agentic":
        # Agentic chunker requires LLM
        llm = kwargs.get('llm')
        if llm is None:
            raise ValueError("Agentic chunking requires 'llm' parameter")
        return AgenticChunker(llm=llm)

    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}. "
                        f"Supported strategies: recursive, token, semantic, agentic")