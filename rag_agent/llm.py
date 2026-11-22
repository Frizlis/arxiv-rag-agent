from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from rag_agent.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create a chat model client using application settings."""
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=str(settings.ollama_base_url),
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_context_length,
            reasoning=settings.llm_thinking,
        )

    raise ValueError(
        f"Unsupported chat model provider: {settings.llm_provider}"
    )
