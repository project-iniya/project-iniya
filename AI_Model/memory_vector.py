import chromadb
import ollama
import uuid

DB_PATH = "AI_Model/memory/vector_memory"
EMBED_MODEL = "mxbai-embed-large"

_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _client.get_or_create_collection(name="chat_memory")


def _embed(text: str):
    """
    Get embedding vector for a given text using Ollama.
    """
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]


def store_message(message: str):
    """
    Store a chunk of conversation or context into vector DB.
    """
    if not message.strip():
        return

    _collection.add(
        documents=[message],
        embeddings=[_embed(message)],
        ids=[str(uuid.uuid4())]
    )


def search_relevant(query: str, limit: int = 5):
    """
    Retrieve semantically similar past messages.
    Returns a list of strings.
    """
    if _collection.count() == 0:
        return []

    results = _collection.query(
        query_embeddings=[_embed(query)],
        n_results=limit
    )

    return results.get("documents", [[]])[0] if results.get("documents") else []

def delete_vector_memory(keyword: str):
    """
    Delete vector entries containing a matching keyword.
    """
    keyword = keyword.lower()
    results = _collection.get(include=["documents", "embeddings"])
    
    if not results or "documents" not in results:
        return "No memory stored."

    delete_ids = [
        id_ for id_, doc in zip(results["ids"], results["documents"])
        if keyword in doc.lower()
    ]

    if delete_ids:
        _collection.delete(ids=delete_ids)
        return f"Removed {len(delete_ids)} related memory entries."
    else:
        return "No matching memory found."
