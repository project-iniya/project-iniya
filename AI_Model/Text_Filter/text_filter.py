import ollama
import numpy as np
from detoxify import Detoxify
from transformers import pipeline
from filter_reference_text import JAILBREAK_PHRASES
from filter_reference_text import PROFANITY_PHRASES


class TextFilter:

    def __init__(self):

        # Toxicity model
        self.detox = Detoxify("original")

        # Hate / abuse classifier (replaces broken NSFW model)
        self.hate = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            return_all_scores=True
        )

        # Jailbreak vectors
        self.JB_VECTORS = [(p, self.embed(p)) for p in JAILBREAK_PHRASES]
        self.JB_STRONG = 0.65
        self.JB_WEAK = 0.50

        # Profanity vectors
        self.PROF_VECTORS = [(p, self.embed(p)) for p in PROFANITY_PHRASES]
        self.PROF_STRONG = 0.80
        self.PROF_WEAK = 0.65


    # ---- Embeddings (Ollama) ----
    def embed(self, text: str):
        r = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )
        return np.array(r["embedding"], dtype=np.float32)

    def cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


    # ---- Vector similarity ----
    def vector_score(self, text, vectors):
        v = self.embed(text)
        best = 0.0
        match = None

        for phrase, pv in vectors:
            s = self.cosine(v, pv)
            if s > best:
                best = s
                match = phrase

        return best, match


    # ---- Hate / abuse score ----
    def hate_score(self, text):
        scores = self.hate(text)[0]
        hate = 0
        abuse = 0

        for s in scores:
            if s["label"] == "hate":
                hate = s["score"]
            if s["label"] == "abusive":
                abuse = s["score"]

        return max(hate, abuse)


    # ---- Main moderation ----
    def moderate(self, text):
        result = {}

        # --- Toxicity ---
        tox = self.detox.predict(text)
        result["toxicity"] = tox

        # --- Hate & abuse ---
        ha = self.hate_score(text)
        result["hate_abuse"] = ha

        # --- Jailbreak ---
        jb_score, jb_match = self.vector_score(text, self.JB_VECTORS)
        result["jailbreak_score"] = jb_score
        result["jailbreak_match"] = jb_match

        # --- Profanity ---
        prof_score, prof_match = self.vector_score(text, self.PROF_VECTORS)
        result["profanity_score"] = prof_score
        result["profanity_match"] = prof_match

        # --- Decision ---
        blocked = False
        reasons = []

        if tox["toxicity"] > 0.8 or tox["severe_toxicity"] > 0.5:
            blocked = True
            reasons.append("toxicity")

        if ha > 0.75:
            blocked = True
            reasons.append("hate/abuse")

        if jb_score > self.JB_STRONG:
            blocked = True
            reasons.append("jailbreak")

        if prof_score > self.PROF_STRONG:
            blocked = True
            reasons.append("profanity")

        result["blocked"] = blocked
        result["reasons"] = reasons

        return result


if __name__ == "__main__":
    import json
    tf = TextFilter()
    with open("test.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()      # removes \n and spaces
            if line and line[0] =='#':
                print(line)
                continue
            if line:
                res = tf.moderate(line)
                print(f"{line} - {res['blocked']}")
            else:
                print()