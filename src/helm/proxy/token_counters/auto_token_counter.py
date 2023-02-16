from typing import Dict, List

from helm.common.request import Request, Sequence
from .free_token_counter import FreeTokenCounter
from .token_counter import TokenCounter


class AutoTokenCounter(TokenCounter):
    """Automatically count tokens based on the organization."""

    def __init__(self, huggingface_client):
        return

    def get_token_counter(self, organization: str) -> TokenCounter:
        """Return a token counter based on the organization."""
        return FreeTokenCounter()

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts tokens based on the organization.
        """
        token_counter: TokenCounter = self.get_token_counter(request.model_organization)
        return token_counter.count_tokens(request, completions)
