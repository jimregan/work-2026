"""Data collator for CTC training with Gemma4ForCTC."""

from dataclasses import dataclass

import torch
from transformers import Wav2Vec2CTCTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor


@dataclass
class DataCollatorCTCWithPadding:
    feature_extractor: Gemma4AudioFeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    padding: bool | str = "longest"

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch
