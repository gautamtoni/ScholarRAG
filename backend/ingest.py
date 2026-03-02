import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import config


def load_pdfs(directory):
    docs = []

    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            path = os.path.join(directory, file)
            loader = PyPDFLoader(path)
            pages = loader.load()
            docs.extend(pages)

    return docs


def ingest():

    documents = load_pdfs(config.PDF_SOURCE_DIRECTORY)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )

    vectordb.persist()
    print("✅ Ingestion Completed")


if __name__ == "__main__":
    ingest()
