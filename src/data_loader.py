from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def load_and_split_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF document and splits it into chunks.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split PDF into {len(splits)} chunks.")
        return splits
    except Exception as e:
        print(f"Error loading or splitting PDF: {e}")
        return []

if __name__ == "__main__":
    import os
    # Example usage - ensure 'example.pdf' exists for testing
    pdf_path = "PostgreSQL_(Postgres).pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Please create or provide a valid PDF.")
    else:
        docs = load_and_split_pdf(pdf_path)
        if docs:
            print("\nFirst chunk content:")
            print(docs[0].page_content[:500])
            print(f"Metadata: {docs[0].metadata}")