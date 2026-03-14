import chromadb
import ollama
import uuid
import hashlib
from datetime import datetime


class VectorMemory:
    def __init__(
        self,
        chat_id: str,
        base_dir="AI_Model/memory_cache",
        embed_model="mxbai-embed-large",
        max_embed_chars=2000,
        max_store_chars=3000,
        embed_retries=2,
    ):
        self.chat_id = str(chat_id)
        self.db_path = f"{base_dir}/{self.chat_id}/vector_memory"

        self.embed_model = embed_model
        self.max_embed_chars = max_embed_chars
        self.max_store_chars = max_store_chars
        self.embed_retries = embed_retries

        self._client = chromadb.PersistentClient(path=self.db_path)
        self._collection = self._client.get_or_create_collection(name="chat_memory")

    # ================= EMBEDDING =================

    def _embed(self, text: str):
        if not text:
            return None

        text = text[: self.max_embed_chars]

        for _ in range(self.embed_retries):
            try:
                res = ollama.embeddings(model=self.embed_model, prompt=text)
                return res["embedding"]
            except Exception:
                pass

        return None

    def _hash(self, text: str):
        return hashlib.md5(text.encode()).hexdigest()

    # ================= STORE =================

    def store_message(self, message: str, role="memory"):
        if not message or not message.strip():
            return

        message = message[: self.max_store_chars]

        emb = self._embed(message)
        if emb is None:
            return

        doc_hash = self._hash(message)

        try:
            self._collection.add(
                documents=[message],
                embeddings=[emb],
                ids=[str(uuid.uuid4())],
                metadatas=[
                    {
                        "hash": doc_hash,
                        "role": role,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ],
            )
        except Exception:
            pass

    # ================= SEARCH =================

    def search_relevant(self, query: str, limit: int = 5):
        if self._collection.count() == 0:
            return []

        emb = self._embed(query)
        if emb is None:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[emb],
                n_results=limit,
            )
            return results.get("documents", [[]])[0]
        except Exception:
            return []

    # ================= DELETE =================

    def delete(self, keyword: str):
        keyword = keyword.lower()

        try:
            results = self._collection.get(include=["documents"])
        except Exception:
            return "Memory access error."

        if not results or "documents" not in results:
            return "No memory stored."

        delete_ids = [
            id_
            for id_, doc in zip(results["ids"], results["documents"])
            if keyword in doc.lower()
        ]

        if delete_ids:
            self._collection.delete(ids=delete_ids)
            return f"Removed {len(delete_ids)} entries."

        return "No matching memory."
