from typing import Dict, List
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

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
        print("Getting or creating collection my_collection")
        collection = client.get_or_create_collection(
            name="my_collection", embedding_function=model.encode
        )

        print(f"my_collection loaded, it has document count: {collection.count()}")

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
        ids = [id if len(id) > 1 else str(uuid.uuid4()) for id in ids]
        # self.collection.add(ids, documents=documents, metadatas=metadatas)
        print(f"adding many: ids: {ids}\ndocuments: {documents}")
        self.collection.add(ids, documents=documents)

    def add(self, id: str, document: str, metadata: Dict[str, str]):
        self.add_many([id], [document], [metadata])

    def query(self, query_text: str, n_results: int, where: Dict[str, str]):
        result = self.collection.query(
            query_texts=query_text, n_results=n_results, where=where
        )

        result["ids"] = result["ids"][0]
        result["documents"] = result["documents"][0]
        result["metadatas"] = result["metadatas"][0]
        result["distances"] = result["distances"][0]

        return result
