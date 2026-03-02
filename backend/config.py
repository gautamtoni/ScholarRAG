class Config:

    PDF_SOURCE_DIRECTORY = "../data"
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"

    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200


config = Config()
