"""
RAG Application Entry Point
============================
Simple entry point to run the Streamlit frontend.

Usage:
    python -m streamlit run app.py
    OR
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the frontend
from src.rag_frontend import main

if __name__ == "__main__":
    main()
