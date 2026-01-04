import requests
from bs4 import BeautifulSoup

def scrape_url(url: str) -> str:
    """
    Fetch webpage text and extract readable content.
    """
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        cleaned = "\n".join(line.strip() for line in text.splitlines() if len(line.strip()) > 60)

        return cleaned[:4000]  # prevent overload
    except Exception as e:
        return f"[SCRAPER ERROR] {e}"
