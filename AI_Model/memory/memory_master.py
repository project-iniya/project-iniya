"""
memory_master.py — Global cross-chat vector memory using mxbai-embed-large
Stored in memory_cache/_master/
"""

import json
import numpy as np
import ollama
from pathlib import Path
from AI_Model.log import log

MASTER_DIR = Path(__file__).resolve().parent.parent.parent / "AI_Model" /"memory_cache" / "_master"
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
TOP_K = 5
SIM_THRESHOLD = 0.45


def _embed(text: str) -> list[float] | None:
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        log(f"Master embed failed: {e}", "MASTER MEM")
        return None


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class MasterMemory:

    def __init__(self):
        MASTER_DIR.mkdir(parents=True, exist_ok=True)
        self.index_path = MASTER_DIR / "index.json"
        self.entries = self._load()
        log(f"MasterMemory loaded ({len(self.entries)} entries)", "MASTER MEM")

    def _load(self) -> list[dict]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self):
        self.index_path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def store(self, text: str, source_chat_id: str = "unknown"):
        """Embed and store a fact in master memory."""
        # deduplicate — don't store near-identical facts
        if self.entries:
            vec = _embed(text)
            if vec:
                for e in self.entries:
                    if _cosine(vec, e["embedding"]) > 0.92:
                        log(f"Master mem: skipped duplicate: {text[:60]}", "MASTER MEM")
                        return
        else:
            vec = _embed(text)

        if not vec:
            log(f"Master mem: embed failed for: {text[:60]}", "MASTER MEM")
            return

        entry = {
            "text": text,
            "embedding": vec,
            "source_chat": source_chat_id,
        }
        self.entries.append(entry)
        self._save()
        log(f"Master mem stored: {text[:60]}", "MASTER MEM")

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        """Return top-k relevant facts for a query."""
        if not self.entries:
            return []
        vec = _embed(query)
        if not vec:
            return []
        scored = [
            (_cosine(vec, e["embedding"]), e["text"])
            for e in self.entries
        ]
        scored.sort(reverse=True)
        return [
            text for score, text in scored
            if score >= SIM_THRESHOLD
        ][:top_k]

    def delete(self, text: str):
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["text"] != text]
        if len(self.entries) < before:
            self._save()
            log(f"Master mem deleted: {text[:60]}", "MASTER MEM")

    def all_facts(self) -> list[str]:
        return [e["text"] for e in self.entries]