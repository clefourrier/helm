from helm.proxy.models import get_model, get_model_names_with_tag, Model, WIDER_CONTEXT_WINDOW_TAG
from .geclm_window_service import GeclmWindowService
from .window_service import WindowService
from .tokenizer_service import TokenizerService


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """
        model: Model = get_model(model_name)
        organization: str = model.organization
        engine: str = model.engine

        window_service: WindowService

        if organization == "geclm":
            window_service = GeclmWindowService(service)
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
