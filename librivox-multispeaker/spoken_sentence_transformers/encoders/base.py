from abc import abstractmethod

import numpy as np
from torch import Tensor

from sentence_transformers.models.Module import Module


class AcousticEncoder(Module):
    """Abstract base for ST-compatible acoustic encoders.

    Subclasses bridge a specific audio backend (HuggingFace transformers,
    EnCodec, …) into the SentenceTransformer pipeline:

        AcousticEncoder → Pooling → [MultiAxisProjection]

    Must implement: tokenize(), forward(), get_word_embedding_dimension().
    """

    @abstractmethod
    def tokenize(self, audio_list: list[np.ndarray | dict], **kwargs) -> dict[str, Tensor]: ...

    @abstractmethod
    def forward(self, features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]: ...

    @abstractmethod
    def get_word_embedding_dimension(self) -> int: ...
