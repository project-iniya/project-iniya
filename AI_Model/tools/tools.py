import json,re,os,sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from tavily import TavilyClient 
from AI_Model.memory_tasks import add_task, list_tasks, complete_task
from AI_Model.log import log
from scraper import scrape_url
from dotenv import load_dotenv 

load_dotenv()

# LLM may respond with natural text before JSON, so parser should catch the first valid JSON.

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
def web_search(query: str, max_results: int = 5) -> str:
    """
    Performs a focused web search using Tavily.
    The model can give natural language queries,
    no keyword formatting required.
    """
    log(f"🔍 Tavily search: {query}")

    try:
        result = tavily.search(
            query=query,
            include_answer=True,
            max_results=max_results
        )
    except Exception as e:
        log(e)
        return f"❌ Tavily Search Failed: {e}"

    # If Tavily gives a direct answer:
    if result.get("answer"):
        log({result['answer']})
        return f"📌 {result['answer']}"

    # Otherwise fall back to link list:
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



TOOLS = {
    "web_search": web_search,
    "web_scrape": scrape_url,
    "task_add": lambda i: add_task(i),
    "task_list": lambda i: list_tasks(),
    "task_complete": lambda i: complete_task(i)
}


def parse_tool_call(text):
    match = re.search(r'\{.*?\}', text.strip(), re.DOTALL)
    if not match:
        return None, None

    try:
        data = json.loads(match.group(0))
        return data.get("action"), data.get("input")
    except Exception:
        return None, None
