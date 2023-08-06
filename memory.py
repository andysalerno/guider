from typing import Dict, List
import chromadb
from sentence_transformers import SentenceTransformer

client = None
collection = None


def get_client():
    global client

    if client is None:
        print("Creating client at: db/chromadb.db")
        client = chromadb.PersistentClient(path="db/chromadb.db")
        print("Client created.")

    return client


def get_collection(model: SentenceTransformer):
    global collection

    client = get_client()

    if collection is None:
        print("Creating collection my_collection")
        collection = client.get_or_create_collection(
            name="my_collection", embedding_function=model.encode
        )
        print("my_collection created")
    else:
        print(f"Loaded existing collection with document count: {collection.count()}")

    return collection


class Memory:
    def __init__(self, model: SentenceTransformer) -> None:
        self.collection = get_collection(model)

    # time-weighted chat history,
    # web_search cached history,
    # 'remember to do X' history
    def add_many(
        self, ids: List[str], documents: List[str], metadatas: List[Dict[str, str]]
    ):
        self.collection.add(ids, documents=documents, metadatas=metadatas)

    def add(self, ids: str, documents: str, metadatas: Dict[str, str]):
        self.collection.add(ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, n_results: int, where: Dict[str, str]):
        return self.collection.query(
            query_texts=query_text, n_results=n_results, where=where
        )
