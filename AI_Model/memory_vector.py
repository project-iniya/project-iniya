import chromadb
import ollama
import uuid
from .config import CURRENT_CHAT_ID

DB_PATH = f"AI_Model/memory/{CURRENT_CHAT_ID}/vector_memory"
EMBED_MODEL = "mxbai-embed-large"

# ---- SAFETY LIMITS ----
MAX_EMBED_CHARS = 2000        # Prevent context overflow
MAX_STORE_CHARS = 3000        # Prevent storing huge texts
EMBED_RETRIES = 2             # Retry embedding if Ollama fails

_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _client.get_or_create_collection(name="chat_memory")


# ---------------- SAFE EMBEDDING ----------------
def _embed(text: str):
    """
    Get embedding vector safely (never crash).
    Truncates large input and retries if Ollama errors.
    """
    if not text:
        return None

    text = text[:MAX_EMBED_CHARS]

    for _ in range(EMBED_RETRIES):
        try:
            res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return res["embedding"]
        except Exception as e:
            print(f"[VECTOR] Embedding error: {e}")

    # If embedding fails completely → skip storing/searching
    return None


# ---------------- STORE MESSAGE ----------------
def store_message(message: str):
    """
    Store a chunk of conversation or context into vector DB safely.
    Never crashes even if embedding fails.
    """
    if not message or not message.strip():
        return

    message = message[:MAX_STORE_CHARS]

    emb = _embed(message)
    if emb is None:
        print("[VECTOR] Skipped storing message (embedding failed)")
        return

    try:
        _collection.add(
            documents=[message],
            embeddings=[emb],
            ids=[str(uuid.uuid4())]
        )
    except Exception as e:
        print(f"[VECTOR] Store error: {e}")


# ---------------- SEARCH ----------------
def search_relevant(query: str, limit: int = 5):
    """
    Retrieve semantically similar past messages safely.
    Returns a list of strings.
    """
    if _collection.count() == 0:
        return []

    emb = _embed(query)
    if emb is None:
        return []

    try:
        results = _collection.query(
            query_embeddings=[emb],
            n_results=limit
        )
        return results.get("documents", [[]])[0] if results.get("documents") else []
    except Exception as e:
        print(f"[VECTOR] Search error: {e}")
        return []


# ---------------- DELETE ----------------
def delete_vector_memory(keyword: str):
    """
    Delete vector entries containing a matching keyword safely.
    """
    keyword = keyword.lower()

    try:
        results = _collection.get(include=["documents"])
    except Exception as e:
        return f"Error accessing memory: {e}"

    if not results or "documents" not in results:
        return "No memory stored."

    delete_ids = [
        id_ for id_, doc in zip(results["ids"], results["documents"])
        if keyword in doc.lower()
    ]

    if delete_ids:
        try:
            _collection.delete(ids=delete_ids)
            return f"Removed {len(delete_ids)} related memory entries."
        except Exception as e:
            return f"Delete failed: {e}"
    else:
        return "No matching memory found."
