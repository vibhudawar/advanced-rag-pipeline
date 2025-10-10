from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER
from src.utils.PromptsConstants import MQE_PROMPT, RAG_SYSTEM_PROMPT

class LLMGenerator(ABC):
    """Abstract base class for LLM generators"""

    @abstractmethod
    def generate_stream(self, query: str, context_documents: List[Document], **kwargs):
        """Generate response with streaming based on query and context documents"""
        pass


class OpenAIGenerator(LLMGenerator):
    """OpenAI LLM generator using LangChain"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.model_name = model
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        # Create the chain
        self.chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    

    def generate_stream(self, query: str, context_documents: List[Document], **kwargs):
        """
        Generate response using OpenAI with streaming

        Yields:
            Token strings as they are generated
        """
        try:
            # Format context
            context = self._format_context(context_documents)

            # Stream response
            for chunk in self.chain.stream({
                "context": context,
                "question": query
            }):
                if chunk:
                    yield chunk

        except Exception as e:
            # On error, yield error message
            yield f"\n\n❌ Error: {str(e)}"
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format context documents for the prompt"""
        if not documents:
            return "No context documents provided."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', f'Document {i}')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            formatted_doc = f"Source {i}: {title} ({source})\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for doc in documents:
            source_info = {
                'title': doc.metadata.get('title', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'url': doc.metadata.get('url', ''),
                'chunk_id': doc.metadata.get('chunk_id', '')
            }
            sources.append(source_info)
        return sources


class GeminiGenerator(LLMGenerator):
    """Google Gemini LLM generator using LangChain"""
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY
        )
        self.model_name = model
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=RAG_SYSTEM_PROMPT + "\n\nAnswer:",
            input_variables=["context", "question"]
        )
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    

    def generate_stream(self, query: str, context_documents: List[Document], **kwargs):
        """
        Generate response using Gemini with streaming

        Yields:
            Token strings as they are generated
        """
        try:
            # Format context
            context = self._format_context(context_documents)

            # Stream response
            for chunk in self.chain.stream({
                "context": context,
                "question": query
            }):
                if chunk:
                    yield chunk

        except Exception as e:
            # On error, yield error message
            yield f"\n\n❌ Error: {str(e)}"

    def _format_context(self, documents: List[Document]) -> str:
        """Format context documents for the prompt"""
        if not documents:
            return "No context documents provided."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', f'Document {i}')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            formatted_doc = f"Source {i}: {title} ({source})\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for doc in documents:
            source_info = {
                'title': doc.metadata.get('title', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'url': doc.metadata.get('url', ''),
                'chunk_id': doc.metadata.get('chunk_id', '')
            }
            sources.append(source_info)
        return sources


def get_llm_generator(provider: str = None, **kwargs) -> LLMGenerator:
    """Factory function to get the appropriate LLM generator"""
    
    if provider is None:
        provider = LLM_PROVIDER or "auto"
    
    if provider == "openai":
        try:
            return OpenAIGenerator(**kwargs)
        except ValueError as e:
            print(f"⚠️ OpenAI generator not available: {str(e)}")
    
    elif provider == "gemini":
        try:
            return GeminiGenerator(**kwargs)
        except ValueError as e:
            print(f"⚠️ Gemini generator not available: {str(e)}")
    
    elif provider == "auto":
        try:
            return OpenAIGenerator(**kwargs)
        except ValueError:
            pass
        try:
            return GeminiGenerator(**kwargs)
        except ValueError:
            pass
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def expand_query(query: str, provider: str = "auto") -> List[str]:
    """
    Expand a user query into multiple alternative versions using Multi-Query Expansion (MQE)

    Args:
        query: Original user query
        provider: LLM provider to use ("auto", "gemini", "openai")

    Returns:
        List of expanded queries (including the original)
    """
    try:
        # Create a simple LLM for query expansion
        if provider == "auto":
            # Try Gemini first
            if GEMINI_API_KEY:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.7,
                    google_api_key=GEMINI_API_KEY
                )
            elif OPENAI_API_KEY:
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    openai_api_key=OPENAI_API_KEY
                )
            else:
                print("⚠️ No LLM API key available for query expansion, using original query only")
                return [query]
        else:
            print(f"⚠️ Unknown provider '{provider}', using original query only")
            return [query]

        # Create chain for query expansion
        prompt = PromptTemplate(
            template=MQE_PROMPT,
            input_variables=["user_query"]
        )

        chain = prompt | llm | StrOutputParser()

        # Generate expanded queries
        expanded_text = chain.invoke({"user_query": query})

        # Parse the response - each query on a new line
        expanded_queries = [q.strip() for q in expanded_text.strip().split('\n') if q.strip()]

        # Filter out numbered queries if LLM didn't follow instructions
        cleaned_queries = []
        for q in expanded_queries:
            # Remove leading numbers like "1.", "1)", etc.
            import re
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', q)
            if cleaned and cleaned not in cleaned_queries:
                cleaned_queries.append(cleaned)

        # Always include the original query first
        all_queries = [query] + cleaned_queries

        for i, q in enumerate(all_queries, 1):
            print(f"{i}. {q}")

        return all_queries

    except Exception as e:
        print(f"⚠️ Query expansion failed: {str(e)}, using original query only")
        return [query]


def generate_response(query: str, context_documents: List[Document], provider: str = None, **kwargs):
    """Convenience function to generate streaming response"""
    generator = get_llm_generator(provider, **kwargs)
    return generator.generate_stream(query, context_documents, **kwargs) 