"""
coding_tools.py — File & shell tools for BrainAgent
All operations are sandboxed to memory_cache/{chat_id}/workspace/
"""

import json
import subprocess
import re
import os
from pathlib import Path
from AI_Model.log import log

EXEC_TIMEOUT = 15

ALLOWED_RUNNERS = {
    ".py":   ["python"],
    ".js":   ["node"],
    ".ts":   ["npx", "ts-node"],
    ".sh":   ["bash"],
    ".rb":   ["ruby"],
}

BLOCKED_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+/[fFsS]",
    r"\bformat\b",
    r"\brmdir\s+/s\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bsudo\b",
    r"\bsu\s+-\b",
    r"\brunas\b",
    r"\bchmod\s+777\b",
    r"\bcurl\b.*\|\s*bash",
    r"\bwget\b.*\|\s*sh",
    r"\bregedit\b",
    r"\breg\s+(add|delete|import)\b",
    r"\bschtasks\b",
    r"\bat\s+\d",
    r"\.\./\.\./",
    r"~\/",
    r"%APPDATA%",
    r"%SYSTEMROOT%",
]

BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


class CodingTools:

    def __init__(self, chat_id: str):
        BASE = Path(__file__).resolve().parent.parent.parent
        self.workspace = BASE / "memory_cache" / str(chat_id) / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        log(f"CodingTools workspace: {self.workspace}", "CODING")

        proto_path = Path(__file__).parent / "coding_protocols.json"
        self.protocols = json.loads(proto_path.read_text(encoding="utf-8"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _safe_path(self, rel_path: str) -> Path | None:
        try:
            target = (self.workspace / rel_path).resolve()
            target.relative_to(self.workspace)  # raises if outside
            return target
        except Exception:
            return None

    def _is_blocked(self, cmd: str) -> tuple[bool, str]:
        for pattern in BLOCKED_RE:
            m = pattern.search(cmd)
            if m:
                return True, f"Blocked pattern: `{m.group()}`"
        return False, ""

    # ── Tools ─────────────────────────────────────────────────────────────────

    def file_read(self, inp: dict) -> dict:
        path = self._safe_path(inp.get("path", ""))
        if not path:
            return {"status": "error", "error": "Path escapes workspace."}
        if not path.exists():
            return {"status": "error", "error": f"File not found: {inp['path']}"}
        if not path.is_file():
            return {"status": "error", "error": "Not a file."}
        try:
            content = path.read_text(encoding="utf-8")
            log(f"file_read: {inp['path']} ({len(content.splitlines())} lines)", "CODING")
            return {"status": "ok", "path": inp["path"], "content": content, "lines": len(content.splitlines())}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def file_write(self, inp: dict) -> dict:
        path = self._safe_path(inp.get("path", ""))
        if not path:
            return {"status": "error", "error": "Path escapes workspace."}
        if path.exists() and not inp.get("overwrite", True):
            return {"status": "error", "error": "File exists and overwrite=False."}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = inp.get("content", "")
            path.write_text(content, encoding="utf-8")
            log(f"file_write: {inp['path']} ({len(content)} bytes)", "CODING")
            return {"status": "ok", "path": inp["path"], "bytes": len(content)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def file_list(self, inp: dict) -> dict:
        path = self._safe_path(inp.get("path", "."))
        if not path:
            return {"status": "error", "error": "Path escapes workspace."}
        if not path.exists():
            return {"status": "error", "error": "Directory not found."}
        try:
            entries = [
                {
                    "name": i.name,
                    "type": "dir" if i.is_dir() else "file",
                    "size": i.stat().st_size if i.is_file() else None
                }
                for i in sorted(path.iterdir())
            ]
            return {"status": "ok", "path": inp.get("path", "."), "entries": entries}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def file_delete(self, inp: dict) -> dict:
        path = self._safe_path(inp.get("path", ""))
        if not path:
            return {"status": "error", "error": "Path escapes workspace."}
        if not path.exists():
            return {"status": "error", "error": "Not found."}
        if path.is_dir():
            return {"status": "error", "error": "Cannot delete directories."}
        try:
            path.unlink()
            log(f"file_delete: {inp['path']}", "CODING")
            return {"status": "ok", "deleted": inp["path"]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def code_run(self, inp: dict) -> dict:
        path = self._safe_path(inp.get("path") or inp.get("script_path") or inp.get("file", ""))
        if not path:
            return {"status": "error", "error": "Path escapes workspace."}
        if not path.exists():
            return {"status": "error", "error": f"File not found: {inp['path']}"}
        runner = ALLOWED_RUNNERS.get(path.suffix.lower())
        if not runner:
            return {"status": "error", "error": f"No runner for '{path.suffix}'. Supported: {list(ALLOWED_RUNNERS)}"}

        # stdin: accept either "stdin" string or "inputs" list (one value per prompt)
        raw_inputs = inp.get("inputs") or []
        stdin_data = inp.get("stdin") or ("\n".join(str(x) for x in raw_inputs) + "\n" if raw_inputs else None)

        try:
            result = subprocess.run(
                runner + [str(path)],
                input=stdin_data,
                cwd=str(self.workspace),
                capture_output=True, text=True,
                timeout=EXEC_TIMEOUT, encoding="utf-8", errors="replace"
            )
            log(f"code_run: {inp['path']} → exit {result.returncode}", "CODING")
            return {
                "status": "ok",
                "exit_code": result.returncode,
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-1000:]
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Timed out after {EXEC_TIMEOUT}s"}
        except FileNotFoundError:
            return {"status": "error", "error": f"Runner not found: {runner[0]}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def shell_exec(self, inp: dict) -> dict:
        command = inp.get("command", "")
        blocked, reason = self._is_blocked(command)
        if blocked:
            log(f"shell_exec BLOCKED: {command!r} — {reason}", "CODING WARN")
            return {"status": "blocked", "reason": reason}
        cwd = self._safe_path(inp.get("cwd", "."))
        if not cwd:
            return {"status": "error", "error": "cwd escapes workspace."}
        
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                command, shell=True, cwd=str(cwd),
                capture_output=True, text=True,
                timeout=EXEC_TIMEOUT, encoding="utf-8", errors="replace",
                env=env
            )
            log(f"shell_exec: {command!r} → exit {result.returncode}", "CODING")
            return {
                "status": "ok",
                "exit_code": result.returncode,
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-1000:]
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Timed out after {EXEC_TIMEOUT}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Registry ──────────────────────────────────────────────────────────────

    def get_tools(self) -> dict:
        """Call this from ToolManager to get the bound tool functions."""
        return {
            "file_read":   self.file_read,
            "file_write":  self.file_write,
            "file_list":   self.file_list,
            "file_delete": self.file_delete,
            "code_run":    self.code_run,
            "shell_exec":  self.shell_exec,
        }