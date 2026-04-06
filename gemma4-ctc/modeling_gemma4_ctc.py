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
"""PyTorch Gemma4ForCTC model."""

import torch
from torch import nn

from transformers import __version__ as _transformers_version
from transformers import AutoModelForMultimodalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import logging

from .configuration_gemma4_ctc import Gemma4CTCConfig


assert tuple(int(x) for x in _transformers_version.split(".")[:2]) >= (5, 5), \
    "transformers >= 5.5.0 required for Gemma4 support"


AUTO_MAP = {
    "AutoConfig": "configuration_gemma4_ctc.Gemma4CTCConfig",
    "AutoModelForCTC": "modeling_gemma4_ctc.Gemma4ForCTC",
}


logger = logging.get_logger(__name__)


class Gemma4CTCPreTrainedModel(PreTrainedModel):
    config_class = Gemma4CTCConfig
    base_model_prefix = "gemma4_audio_encoder"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    # output_proj is replaced with nn.Identity() at init time; ignore any
    # stale weight keys that may appear in checkpoints saved before that swap.
    _keys_to_ignore_on_load_unexpected = [r"gemma4_audio_encoder\.output_proj\..*"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range") else 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class Gemma4ForCTC(Gemma4CTCPreTrainedModel):
    def __init__(self, config: Gemma4CTCConfig):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Gemma4ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        _full_model = AutoModelForMultimodalLM.from_pretrained(
            config.gemma4_audio_model_id,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.gemma4_audio_encoder = _full_model.model.audio_tower
        del _full_model

        # Disable the projection to LLM embedding space (1024 → 1536);
        # we want raw conformer output.
        self.gemma4_audio_encoder.output_proj = nn.Identity()

        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def freeze_audio_encoder(self):
        """Freeze all Gemma4 conformer parameters."""
        for param in self.gemma4_audio_encoder.parameters():
            param.requires_grad = False

    def freeze_audio_encoder_except_norm(self):
        """Freeze encoder but leave layer norm parameters (norm_out) trainable."""
        for name, param in self.gemma4_audio_encoder.named_parameters():
            if "norm_out" not in name:
                param.requires_grad = False

    def forward(
        self,
        input_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch, time, 128)`):
            Log-mel features from ``Gemma4AudioFeatureExtractor``.
        attention_mask (`torch.BoolTensor` of shape `(batch, time)`, *optional*):
            Mask over padding frames (True = valid).
        labels (`torch.LongTensor` of shape `(batch, target_length)`, *optional*):
            CTC target labels.  Indices in ``[-100, 0, ..., config.vocab_size - 1]``;
            positions set to ``-100`` are ignored.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.gemma4_audio_encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # outputs.last_hidden_state: (batch, time/4, hidden_size)
        # outputs.attention_mask:    (batch, time/4) bool, True = valid frame
        hidden_states = outputs.last_hidden_state
        output_mask = outputs.attention_mask

        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            input_lengths = output_mask.sum(dim=-1).long()

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(loss=loss, logits=logits)


__all__ = ["Gemma4CTCPreTrainedModel", "Gemma4ForCTC"]
