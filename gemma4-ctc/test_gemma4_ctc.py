"""Sanity check for Gemma4ForCTC loaded via trust_remote_code."""

import torch
from transformers import AutoConfig, AutoModelForCTC

config = AutoConfig.from_pretrained("./gemma4-ctc", trust_remote_code=True)
model = AutoModelForCTC.from_pretrained("./gemma4-ctc", trust_remote_code=True)

assert type(model.gemma4_audio_encoder.output_proj).__name__ == "Identity", (
    f"output_proj should be Identity, got {type(model.gemma4_audio_encoder.output_proj)}"
)

assert model.lm_head.in_features == 1024, (
    f"lm_head.in_features should be 1024, got {model.lm_head.in_features}"
)

batch, time, n_mels = 2, 400, 128
input_features = torch.zeros(batch, time, n_mels, dtype=torch.bfloat16)
attention_mask = torch.ones(batch, time, dtype=torch.bool)

model.eval()
with torch.no_grad():
    out = model(input_features=input_features, attention_mask=attention_mask)

expected_time = time // 4
assert out.logits.shape == (batch, expected_time, config.vocab_size), (
    f"logits shape {out.logits.shape} != expected ({batch}, {expected_time}, {config.vocab_size})"
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {total - trainable:,}")
print("All assertions passed.")
