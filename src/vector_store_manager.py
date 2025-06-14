from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List
import shutil
import os

class VectorStoreManager:
    def __init__(self, embeddings_model: Embeddings, persist_directory: str = "vector_db"):
        self.embeddings_model = embeddings_model
        self.persist_directory = persist_directory
        self.vectorstore = None

    def create_or_load_vector_store(self, documents: List[Document] = None):
        """
        Creates a new Chroma vector store from documents or loads an existing one.
        If documents are provided, it will create a new store and persist it.
        Otherwise, it attempts to load an existing store.
        """
        if os.path.exists(self.persist_directory) and documents is None:
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings_model
            )
        elif documents is not None:
            if os.path.exists(self.persist_directory):
                print(f"Removing existing vector store at {self.persist_directory}")
                shutil.rmtree(self.persist_directory) # Clear old data

            print(f"Creating new vector store and persisting to {self.persist_directory}")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings_model,
                persist_directory=self.persist_directory
            )
            print("Vector store created successfully.")
        else:
            raise ValueError("No documents provided to create a new vector store, and no existing store found.")

    def get_retriever(self, k: int = 4):
        """Returns a retriever configured for the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_or_load_vector_store first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def delete_vector_store(self):
        """Deletes the persistent vector store directory."""
        if os.path.exists(self.persist_directory):
            print(f"Deleting vector store at {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
        else:
            print(f"No vector store found at {self.persist_directory} to delete.")

if __name__ == "__main__":
    from llm_config import get_embeddings_model
    from data_loader import load_and_split_pdf

    # Ensure you have an 'example.pdf'
    pdf_path = "PostgreSQL_(Postgres).pdf"
    if not os.path.exists(pdf_path):
        print(f"Please place an 'example.pdf' in the root directory for testing.")
        exit()

    embeddings = get_embeddings_model()
    manager = VectorStoreManager(embeddings_model=embeddings)

    # 1. Create a new vector store
    print("\n--- Creating a new vector store ---")
    documents = load_and_split_pdf(pdf_path)
    if documents:
        manager.create_or_load_vector_store(documents=documents)
        retriever = manager.get_retriever()
        query = "What is LangChain?"
        retrieved_docs = retriever.invoke(query)
        print(f"\nRetrieved {len(retrieved_docs)} documents for query: '{query}'")
        if retrieved_docs:
            print("First retrieved document content:")
            print(retrieved_docs[0].page_content[:200] + "...")

    # 2. Load an existing vector store
    print("\n--- Loading existing vector store ---")
    manager_load = VectorStoreManager(embeddings_model=embeddings) # New instance to simulate loading
    manager_load.create_or_load_vector_store() # No documents provided, so it tries to load
    retriever_load = manager_load.get_retriever()
    query_load = "What is a document?"
    retrieved_docs_load = retriever_load.invoke(query_load)
    print(f"\nRetrieved {len(retrieved_docs_load)} documents for query: '{query_load}' (from loaded store)")

    # 3. Clean up
    # manager.delete_vector_store()