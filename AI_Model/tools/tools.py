import json
import re
import os
from tavily import TavilyClient
from dotenv import load_dotenv

from AI_Model.memory.memory_tasks import TaskMemory
from AI_Model.log import log
from .scraper import scrape_url
from AI_Model.tools.directives import protocol


class ToolManager:
    def __init__(self,chatID, tavily_api_key: str | None = None):
        load_dotenv()
        self.task_memory = TaskMemory(chatID)
        self.chatID = chatID
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=self.tavily_api_key)

        # Tool registry (per-instance)
        self.tools = {
            "web_search": self.web_search,
            "web_scrape": scrape_url,

            "task_add": self.task_memory.add_task,
            "task_list": self.task_memory.list_tasks,
            "task_complete": self.task_memory.complete_task,
            "task_delete": self.task_memory.delete_task,

            "execute_protocol": protocol.execute_protocol_by_code,
        }

    # ================= WEB SEARCH =================

    def web_search(self, query: str, max_results: int = 5) -> str:
        log(f"🔍 Tavily search: {query}")

        try:
            result = self.tavily.search(
                query=query,
                include_answer=True,
                max_results=max_results,
            )
        except Exception as e:
            log(e)
            return f"❌ Tavily Search Failed: {e}"

        # Direct answer from Tavily
        if result.get("answer"):
            log({result["answer"]})
            return f"📌 {result['answer']}"

        items = result.get("results", [])
        log(f"Found {len(items)} results.", "TOOLS")

        if not items:
            log("No results found.")
            return "No results found."

        formatted = [
            f"[{i+1}] {item['title']}\n{item['url']}"
            for i, item in enumerate(items[:max_results])
        ]

        return "\n".join(formatted)

    # ================= TOOL CALL PARSER =================

    @staticmethod
    def parse_tool_call(text: str):
        """
        Extract first JSON object from text.
        LLM may prepend natural language before JSON.
        """
        match = re.search(r"\{.*?\}", text.strip(), re.DOTALL)
        if not match:
            return None, None

        try:
            data = json.loads(match.group(0))
            return data.get("action"), data.get("input")
        except Exception:
            return None, None

    # ================= EXECUTE TOOL =================

    def execute(self, action: str, tool_input):
        tool = self.tools.get(action)

        if not tool:
            return f"Unknown tool: {action}"

        try:
            if tool_input is None:
                return tool()
            elif isinstance(tool_input, dict):
                return tool(**tool_input)
            else:
                return tool(tool_input)
        except Exception as e:
            log(f"Tool error [{action}]: {e}", "TOOLS")
            return f"Tool execution failed: {e}"
