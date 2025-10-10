"""
Helper classes and utilities for agentic chunking.

This module contains the implementation details for agentic chunking,
including proposition extraction and chunk management logic.
"""

from typing import List, Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field

from src.utils.PromptsConstants import (
    PROPOSITION_EXTRACTION_PROMPT,
    FIND_RELEVANT_CHUNK_PROMPT,
    NEW_CHUNK_SUMMARY_PROMPT,
    NEW_CHUNK_TITLE_PROMPT,
    UPDATE_CHUNK_SUMMARY_PROMPT,
    UPDATE_CHUNK_TITLE_PROMPT
)


class PropositionExtractor:
    """
    Extracts atomic propositions from text using LLM.

    Propositions are self-contained, standalone statements that represent
    a single fact or idea from the source text.
    """

    def __init__(self, llm):
        """
        Initialize proposition extractor with LLM.

        Args:
            llm: LangChain chat model instance (e.g., ChatOpenAI, ChatGoogleGenerativeAI)
        """
        self.llm = llm

        # Define Pydantic schema for proposition extraction
        class Sentences(BaseModel):
            """List of atomic propositions extracted from text"""
            sentences: List[str] = Field(description="List of extracted atomic propositions")

        self.sentences_schema = Sentences
        self.prompt = PROPOSITION_EXTRACTION_PROMPT

        # Create structured output chain using modern approach
        self.structured_llm = self.llm.with_structured_output(Sentences)

    def extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic propositions from text.

        Args:
            text: Input text to extract propositions from

        Returns:
            List of proposition strings
        """
        try:
            # Create chain with structured output
            chain = self.prompt | self.structured_llm

            # Extract propositions
            result = chain.invoke({"text": text})

            if result and hasattr(result, 'sentences') and result.sentences:
                return result.sentences

            # Fallback: return sentences if extraction fails
            return self._fallback_extraction(text)

        except Exception as e:
            # Fallback to simple sentence splitting on error
            print(f"[WARN] Proposition extraction failed: {e}, using fallback")
            return self._fallback_extraction(text)

    @staticmethod
    def _fallback_extraction(text: str) -> List[str]:
        """Fallback method for proposition extraction using simple sentence splitting."""
        return [s.strip() + '.' for s in text.split('.') if s.strip()]


class ChunkManager:
    """
    Manages chunk storage and operations for agentic chunking.

    This class handles the internal state of chunks during the agentic
    chunking process, including creation, updates, and lookups.
    """

    def __init__(self, id_truncate_limit: int = 5):
        """
        Initialize chunk manager.

        Args:
            id_truncate_limit: Length of chunk IDs (default: 5)
        """
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.id_truncate_limit = id_truncate_limit

    def create_chunk(self, proposition: str, summary: str, title: str) -> str:
        """
        Create a new chunk with the given proposition, summary, and title.

        Args:
            proposition: The initial proposition for the chunk
            summary: Generated summary for the chunk
            title: Generated title for the chunk

        Returns:
            The chunk ID
        """
        chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]

        self.chunks[chunk_id] = {
            'chunk_id': chunk_id,
            'propositions': [proposition],
            'title': title,
            'summary': summary,
            'chunk_index': len(self.chunks)
        }

        return chunk_id

    def add_proposition_to_chunk(self, chunk_id: str, proposition: str):
        """
        Add a proposition to an existing chunk.

        Args:
            chunk_id: ID of the chunk to add to
            proposition: Proposition to add
        """
        if chunk_id in self.chunks:
            self.chunks[chunk_id]['propositions'].append(proposition)

    def update_chunk_metadata(self, chunk_id: str, summary: str = None, title: str = None):
        """
        Update chunk's summary and/or title.

        Args:
            chunk_id: ID of the chunk to update
            summary: New summary (optional)
            title: New title (optional)
        """
        if chunk_id in self.chunks:
            if summary is not None:
                self.chunks[chunk_id]['summary'] = summary
            if title is not None:
                self.chunks[chunk_id]['title'] = title

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)

    def get_all_chunks(self) -> Dict[str, Dict[str, Any]]:
        """Get all chunks."""
        return self.chunks

    def get_chunk_outline(self) -> str:
        """
        Get a string representation of all current chunks.

        Returns:
            Formatted string with chunk IDs, names, and summaries
        """
        chunk_outline = ""
        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = (
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Chunk Name: {chunk['title']}\n"
                f"Chunk Summary: {chunk['summary']}\n\n"
            )
            chunk_outline += single_chunk_string
        return chunk_outline

    def is_empty(self) -> bool:
        """Check if there are no chunks."""
        return len(self.chunks) == 0


class LLMChunkDecider:
    """
    Uses LLM to make decisions about chunk creation and updates.

    This class encapsulates all LLM interactions for the agentic chunking process.
    """

    def __init__(self, llm, id_truncate_limit: int = 5):
        """
        Initialize LLM chunk decider.

        Args:
            llm: LangChain chat model instance
            id_truncate_limit: Expected length of chunk IDs for validation
        """
        self.llm = llm
        self.id_truncate_limit = id_truncate_limit

    def find_relevant_chunk(self, proposition: str, chunk_outline: str) -> Optional[str]:
        """
        Find the most relevant existing chunk for a proposition.

        Args:
            proposition: The proposition to find a chunk for
            chunk_outline: String representation of all current chunks

        Returns:
            Chunk ID if a match is found, None otherwise
        """
        # Define schema for chunk ID extraction
        class ChunkID(BaseModel):
            """Extracting the chunk id"""
            chunk_id: Optional[str] = Field(description="The ID of the most relevant chunk")

        # Create structured output chain
        structured_llm = self.llm.with_structured_output(ChunkID)
        runnable = FIND_RELEVANT_CHUNK_PROMPT | structured_llm

        try:
            result = runnable.invoke({
                "proposition": proposition,
                "current_chunk_outline": chunk_outline
            })

            if result and hasattr(result, 'chunk_id') and result.chunk_id:
                chunk_found = result.chunk_id

                # Validate chunk ID length
                if len(chunk_found) == self.id_truncate_limit:
                    return chunk_found

        except Exception as e:
            print(f"[WARN] Chunk matching failed: {e}")
            pass

        return None

    def generate_new_chunk_summary(self, proposition: str) -> str:
        """
        Generate a summary for a new chunk.

        Args:
            proposition: The initial proposition for the chunk

        Returns:
            Generated summary
        """
        runnable = NEW_CHUNK_SUMMARY_PROMPT | self.llm
        summary = runnable.invoke({"proposition": proposition}).content
        return summary

    def generate_new_chunk_title(self, summary: str) -> str:
        """
        Generate a title for a new chunk based on its summary.

        Args:
            summary: The chunk summary

        Returns:
            Generated title
        """
        runnable = NEW_CHUNK_TITLE_PROMPT | self.llm
        title = runnable.invoke({"summary": summary}).content
        return title

    def update_chunk_summary(self, propositions: List[str], current_summary: str) -> str:
        """
        Update a chunk's summary after adding new propositions.

        Args:
            propositions: All propositions in the chunk
            current_summary: Current summary of the chunk

        Returns:
            Updated summary
        """
        runnable = UPDATE_CHUNK_SUMMARY_PROMPT | self.llm
        new_summary = runnable.invoke({
            "proposition": "\n".join(propositions),
            "current_summary": current_summary
        }).content
        return new_summary

    def update_chunk_title(self, propositions: List[str], current_summary: str, current_title: str) -> str:
        """
        Update a chunk's title after adding new propositions.

        Args:
            propositions: All propositions in the chunk
            current_summary: Current summary of the chunk
            current_title: Current title of the chunk

        Returns:
            Updated title
        """
        runnable = UPDATE_CHUNK_TITLE_PROMPT | self.llm
        updated_title = runnable.invoke({
            "proposition": "\n".join(propositions),
            "current_summary": current_summary,
            "current_title": current_title
        }).content
        return updated_title
