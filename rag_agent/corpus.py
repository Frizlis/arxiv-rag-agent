from pathlib import Path

from pydantic import TypeAdapter

from rag_agent.schemas import Article

ARTICLE_LIST_ADAPTER = TypeAdapter(list[Article])


def load_corpus(path: Path) -> list[Article]:
    """Load and validate articles from a corpus JSON file."""
    content = path.read_text(encoding="utf-8-sig")
    articles = ARTICLE_LIST_ADAPTER.validate_json(content)

    if not articles:
        raise ValueError("Corpus must not be empty")

    seen_urls: set[str] = set()

    for index, article in enumerate(articles):
        if article.url in seen_urls:
            raise ValueError(
                f"Duplicate article URL at index {index}: {article.url}"
            )
        seen_urls.add(article.url)

    return articles
