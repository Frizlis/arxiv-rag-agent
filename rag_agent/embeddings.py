from sentence_transformers import SentenceTransformer

from rag_agent.config import Settings


def create_embedding_model(settings: Settings) -> SentenceTransformer:
    """Load an embedding model using application settings."""
    if settings.embedding_provider != "sentence_transformers":
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding_provider}"
        )

    model = SentenceTransformer(
        model_name_or_path=settings.embedding_model,
        revision=settings.embedding_revision,
        device=settings.embedding_device,
        cache_folder=str(settings.embedding_cache_dir),
        local_files_only=settings.embedding_local_files_only,
    )

    dimension = model.get_embedding_dimension()
    if dimension != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected "
            f"{settings.embedding_dim}, got {dimension}"
        )

    return model
