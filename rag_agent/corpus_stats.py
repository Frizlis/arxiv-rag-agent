import re

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rag_agent.config import Settings
from rag_agent.corpus import load_corpus

PARAGRAPH_SEPARATOR = re.compile(r"\r?\n[ \t]*\r?\n")
TOKEN_THRESHOLDS = (256, 512, 1024)


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count text tokens without truncation or special tokens."""
    token_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=False, verbose=False
    )
    return len(token_ids)


def print_statistics(name: str, values: list[int]) -> None:
    """Print distribution statistics for a nonempty sequence."""
    array = np.asarray(values, dtype=np.int64)

    print(f"\n{name}")
    print("Count:", array.size)
    print("Min:", int(array.min()))
    print("Mean:", round(float(array.mean()), 1))

    for percentile in [50, 90, 95, 99]:
        value = np.percentile(array, percentile)
        print(f"P{percentile}:", round(float(value), 1))

    print("Max:", int(array.max()))

    for threshold in TOKEN_THRESHOLDS:
        percentage = float((array > threshold).mean() * 100)
        print(f"Above {threshold}: {percentage:.2f}%")


def main() -> None:
    settings = Settings()
    articles = load_corpus(settings.corpus_path)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model,
        revision=settings.embedding_revision,
        cache_dir=str(settings.embedding_cache_dir),
        local_files_only=True,
    )

    article_lengths: list[int] = []
    paragraph_lengths: list[int] = []
    paragraphs_per_article: list[int] = []

    for article in articles:
        article_lengths.append(count_tokens(article.body, tokenizer))

        paragraphs = [
            part
            for part in PARAGRAPH_SEPARATOR.split(article.body)
            if part.strip()
        ]
        paragraphs_per_article.append(len(paragraphs))

        for paragraph in paragraphs:
            paragraph_lengths.append(count_tokens(paragraph, tokenizer))

    print("Model:", settings.embedding_model)
    print("Revision:", settings.embedding_revision)
    print("Special tokens included: False")
    print("Truncation enabled: False")

    print_statistics("Article tokens", article_lengths)
    print_statistics("Paragraph tokens", paragraph_lengths)

    counts = np.asarray(paragraphs_per_article)
    print("\nParagraphs per article")
    print("Min:", int(counts.min()))
    print("Median:", float(np.median(counts)))
    print("Max:", int(counts.max()))


if __name__ == "__main__":
    main()
