# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma4CTC model configuration"""

from transformers import PreTrainedConfig


AUTO_MAP = {
    "AutoConfig": "configuration_gemma4_ctc.Gemma4CTCConfig",
}


class Gemma4CTCConfig(PreTrainedConfig):
    r"""
    Configuration class for Gemma4ForCTC.

    This config wraps the Gemma 4 audio encoder (USM-style conformer) for
    CTC-based ASR.  The audio encoder is loaded from a pre-trained Gemma 4
    multimodal checkpoint; its final ``output_proj`` linear (1024 → 1536) is
    replaced with ``nn.Identity()`` so the CTC head operates directly on the
    1024-dimensional conformer output.

    Args:
        vocab_size (`int`, *optional*, defaults to 32):
            Vocabulary size of the CTC head.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the conformer output (after replacing
            ``output_proj`` with ``nn.Identity()``).  Must match
            ``Gemma4AudioConfig.hidden_size`` of the chosen checkpoint.
        final_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied to the encoder output before the
            CTC linear head.
        ctc_loss_reduction (`str`, *optional*, defaults to ``"sum"``):
            Reduction passed to ``torch.nn.functional.ctc_loss``.
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite CTC losses and their gradients.
        pad_token_id (`int`, *optional*, defaults to 0):
            Token id used as the CTC blank label.
        gemma4_audio_model_id (`str`, *optional*, defaults to
            ``"google/gemma-4-e2b-it"``):
            Hub repo (or local path) of the Gemma 4 multimodal checkpoint
            from which the audio tower is extracted during ``__init__``.
    """

    model_type = "gemma4_ctc"

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 1024,
        final_dropout: float = 0.1,
        ctc_loss_reduction: str = "sum",
        ctc_zero_infinity: bool = False,
        pad_token_id: int = 0,
        gemma4_audio_model_id: str = "google/gemma-4-e2b-it",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.final_dropout = final_dropout
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.gemma4_audio_model_id = gemma4_audio_model_id


__all__ = ["Gemma4CTCConfig"]
