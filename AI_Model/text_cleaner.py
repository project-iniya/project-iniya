import re
from html import unescape
from typing import Dict, List

BOLD = re.compile(r"\*\*(.*?)\*\*")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`([^`]+)`")
CODE_BLOCK = re.compile(r"```(\w+)?\n([\s\S]*?)```")

LIST_LINE = re.compile(r"^\s*(\d+\.|[\-\*])\s+(.*)")

def clean_text(text: str) -> Dict:
    """
    Turns LLM markdown into:
    - TTS-safe text
    - Extracted rich objects (bold, links, code, lists)
    """

    original = text

    # ---- Extract code blocks first ----
    code_blocks = []
    def _grab_code(m):
        lang = m.group(1) or "text"
        code = m.group(2)
        code_blocks.append({
            "language": lang.strip(),
            "code": code.strip()
        })
        return " "  # Remove from spoken text

    text = CODE_BLOCK.sub(_grab_code, text)

    # ---- Extract inline code ----
    inline_code = INLINE_CODE.findall(text)

    # ---- Extract bold & links ----
    bold = BOLD.findall(text)
    links = [{"text": m[0], "url": m[1]} for m in LINK.findall(text)]

    # ---- Strip markdown but keep content ----
    text = LINK.sub(r"\1", text)
    text = BOLD.sub(r"\1", text)
    text = INLINE_CODE.sub(r"\1", text)

    lines = text.splitlines()

    list_items = []
    spoken_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = LIST_LINE.match(line)
        if m:
            item = m.group(2)
            list_items.append(item)
            spoken_lines.append(item)
        else:
            spoken_lines.append(line)

    text = " ".join(spoken_lines)

    # ---- Kill markdown artifacts that TTS hates ----
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("•", " ")
    text = text.replace("###", " ")
    text = text.replace("##", " ")
    text = text.replace("#", " ")

    # ---- Normalize whitespace ----
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "tts_text": text,
        "bold": bold,
        "links": links,
        "inline_code": inline_code,
        "code_blocks": code_blocks,
        "list_items": list_items,
        "raw": original
    }
