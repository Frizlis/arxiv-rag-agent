from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5


def create_document_id(url: str) -> UUID:
    """Create a stable document ID from the original URL."""
    return uuid5(NAMESPACE_URL, url)


def compute_content_hash(body: str) -> str:
    """Compute a SHA-256 digest of the original article body."""
    return sha256(body.encode("utf-8")).hexdigest()
