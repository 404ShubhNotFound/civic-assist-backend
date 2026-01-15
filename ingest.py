import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_engine import EmbeddingManager, VectorStore

def ingest_data(pdf_directory: str):
    print(f"Scanning {pdf_directory}...")
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    all_docs = []
    
    # 1. Load PDFs
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = pdf.name
            all_docs.extend(docs)
            print(f"Loaded {pdf.name}")
        except Exception as e:
            print(f"Error loading {pdf.name}: {e}")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate Embeddings
    print("Generating embeddings (this may take a moment)...")
    embedder = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.generate_embeddings(texts)

    # 4. Save to Vector Store
    print("Saving to ChromaDB...")
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)
    print("Ingestion Complete!")

if __name__ == "__main__":
    # Point this to your data folder
    ingest_data("./data")