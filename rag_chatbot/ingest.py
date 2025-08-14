"""Utilities for indexing documents into a Qdrant vector store."""

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

def ingest_documents(
    data_dir: str | Path,
    collection_name: str = "docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Qdrant:
    """Load text files, create embeddings, and store them in Qdrant.

    Args:
        data_dir: Directory containing ``.txt`` files to index.
        collection_name: Name of the Qdrant collection.
        chunk_size: Size of text chunks for splitting documents.
        chunk_overlap: Overlap between text chunks.

    Returns:
        An in-memory :class:`~langchain_community.vectorstores.Qdrant` instance
        containing the embedded documents.
    """
    loader = DirectoryLoader(str(data_dir), glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant.from_documents(
        splits,
        embeddings,
        location=":memory:",
        collection_name=collection_name,
    )
    return vector_store

