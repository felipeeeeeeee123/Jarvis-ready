import random
import time
from pathlib import Path
from typing import List

import feedparser

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]


class TrendingTopics:
    def __init__(self, cache_file: str = "data/trending.json"):
        self.cache = Path(cache_file)
        self.topics: List[str] = []
        self.last_fetch = 0
        self.load()

    def load(self):
        if self.cache.exists():
            try:
                import json
                data = json.loads(self.cache.read_text())
                self.topics = data.get("topics", [])
                self.last_fetch = data.get("timestamp", 0)
            except Exception:
                pass

    def save(self):
        self.cache.parent.mkdir(exist_ok=True)
        import json
        self.cache.write_text(
            json.dumps({"topics": self.topics, "timestamp": self.last_fetch}, indent=2)
        )

    def fetch(self):
        if time.time() - self.last_fetch < 24 * 3600 and self.topics:
            return self.topics
        topics = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    topics.append(entry.title)
            except Exception:
                continue
        if topics:
            self.topics = topics[:20]
            self.last_fetch = time.time()
            self.save()
        return self.topics

    def random_topic(self) -> str:
        topics = self.fetch()
        if not topics:
            return random.choice([
                "technology",
                "science",
                "finance",
                "sports",
                "culture",
            ])
        return random.choice(topics)
