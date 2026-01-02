from .scraper import scrape_url
from .sub_llm.sub_agent import sub_parse_url
from .log import log


def get_content_from_url(text: str) -> str:
    """
    Downloads content from a URL found in the given text.
    """

    url = sub_parse_url(text)
    log(f"Extracted URL: {url}", "TOOLS")

    if url == "No URL found.":
        return "❌ No URL found in the provided text."

    content = scrape_url(url)
    return content