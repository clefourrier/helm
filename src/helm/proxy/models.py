from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Different modalities
TEXT_MODEL_TAG: str = "text"
IMAGE_MODEL_TAG: str = "image"
CODE_MODEL_TAG: str = "code"
EMBEDDING_MODEL_TAG: str = "embedding"

# Some model APIs have limited functionalities
FULL_FUNCTIONALITY_TEXT_MODEL_TAG: str = "full_functionality_text"
LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG: str = "limited_functionality_text"

# For OpenAI models with wider context windows
WIDER_CONTEXT_WINDOW_TAG: str = "wider_context_window"

# To fetch models that use these tokenizers
GPT2_TOKENIZER_TAG: str = "gpt2_tokenizer"
AI21_TOKENIZER_TAG: str = "ai21_tokenizer"
COHERE_TOKENIZER_TAG: str = "cohere_tokenizer"
OPT_TOKENIZER_TAG: str = "opt_tokenizer"
GPTJ_TOKENIZER_TAG: str = "gptj_tokenizer"
GPTNEO_TOKENIZER_TAG: str = "gptneo_tokenizer"

# Models that are used for ablations and fine-grained analyses.
# These models are selected specifically because of their low marginal cost to evaluate.
ABLATION_MODEL_TAG: str = "ablation"

# Some models (e.g., T5) have stripped newlines.
# So we cannot use \n as a stop sequence for these models.
NO_NEWLINES_TAG: str = "no_newlines"

# Some models (e.g., UL2) require a prefix (e.g., [NLG]) in the
# prompts to indicate the mode before doing inference.
NLG_PREFIX_TAG: str = "nlg_prefix_tag"


@dataclass
class Model:
    """Represents a model that we can make requests to."""

    # Model group, used to filter for quotas (e.g. "huggingface")
    group: str

    # Name of the specific model (e.g. "huggingface/gpt-j-6b")
    name: str

    # Display name of the specific model (e.g. "GPT-J-6B")
    display_name: str

    # Organization that originally created the model (e.g. "EleutherAI")
    #   Note that this may be different from group or the prefix of the name
    #   ("huggingface" in "huggingface/gpt-j-6b") as the hosting organization
    #   may be different from the creator organization. We also capitalize
    #   this field properly to later display in the UI.
    creator_organization: str

    # Short description of the model
    description: str

    # Tags corresponding to the properties of the model
    tags: List[str] = field(default_factory=list)

    # Estimated training co2e cost of this model
    training_co2e_cost: Optional[float] = None

    @property
    def organization(self) -> str:
        """
        Extracts the organization from the model name.
        Example: 'ai21/j1-jumbo' => 'ai21'
        """
        return self.name.split("/")[0]

    @property
    def engine(self) -> str:
        """
        Extracts the model engine from the model name.
        Example: 'ai21/j1-jumbo' => 'j1-jumbo'
        """
        return self.name.split("/")[1]


# For the list of available models, see the following docs:
# Note that schema.yaml has much of this information now.
# Over time, we should add more information there.

ALL_MODELS = [
    # For debugging
    Model(
        group="simple",
        creator_organization="Simple",
        name="simple/model1",
        display_name="Simple Model 1",
        description="Copy last tokens (for debugging)",
    ),
    # Test
    Model(
        group="geclm",
        creator_organization="geclm",
        name="geclm/gpt2_local",
        display_name="GPT2 local",
        description="GPT2 local",
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, GPT2_TOKENIZER_TAG],
    ),
]

MODEL_NAME_TO_MODEL: Dict[str, Model] = {model.name: model for model in ALL_MODELS}


def get_model(model_name: str) -> Model:
    """Get the `Model` given the name."""
    if model_name not in MODEL_NAME_TO_MODEL:
        raise ValueError(f"No model with name: {model_name}")

    return MODEL_NAME_TO_MODEL[model_name]


def get_model_group(model_name: str) -> str:
    """Get the model's group given the name."""
    model: Model = get_model(model_name)
    return model.group


def get_all_models() -> List[str]:
    """Get all model names."""
    return list(MODEL_NAME_TO_MODEL.keys())


def get_models_by_organization(organization: str) -> List[str]:
    """
    Gets models by organization e.g., ai21 => ai21/j1-jumbo, ai21/j1-grande, ai21-large.
    """
    return [model.name for model in ALL_MODELS if model.organization == organization]


def get_model_names_with_tag(tag: str) -> List[str]:
    """Get all the name of the models with tag `tag`."""
    return [model.name for model in ALL_MODELS if tag in model.tags]


def get_all_text_models() -> List[str]:
    """Get all text model names."""
    return get_model_names_with_tag(TEXT_MODEL_TAG)


def get_all_code_models() -> List[str]:
    """Get all code model names."""
    return get_model_names_with_tag(CODE_MODEL_TAG)
