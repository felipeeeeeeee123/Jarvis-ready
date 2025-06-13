import requests
from bs4 import BeautifulSoup

def web_search(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        results = soup.find_all("a", class_="result__a", limit=3)
        return "\n".join([r.get_text() for r in results]) or "No results found."
    except Exception as e:
        return f"[Web search error: {e}]"
