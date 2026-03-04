"""Training script for MultiAxisProjection heads.

Encodes audio with a frozen encoder (default: openai/whisper-large-v3),
caches the embeddings, then trains lightweight per-axis projection heads
with contrastive (InfoNCE) losses.

Usage::

    python train_projection.py \
        --dataset_config datasets.json \
        --encoder_model openai/whisper-large-v3 \
        --cache_dir ./encoder_cache \
        --output_dir ./trained_projection \
        --axes content:256 speaker:256 accent:128 \
        --hidden_dim 0 \
        --batch_size 256 \
        --epochs 20 \
        --lr 1e-3 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from spoken_sentence_transformers import MultiAxisProjection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Column mapping for a single HuggingFace dataset."""

    name: str
    split: str = "train"
    audio_column: str = "audio"
    axis_columns: dict[str, str] = field(default_factory=dict)
    # Maps axis name → dataset column name.
    # e.g. {"content": "text", "speaker": "speaker_id", "accent": "accent"}
    include_values: dict[str, list[str]] = field(default_factory=dict)
    # Keep only rows where column value is in the list.
    # e.g. {"speaker_id": ["p225", "p226"]}
    exclude_values: dict[str, list[str]] = field(default_factory=dict)
    # Drop rows where column value is in the list.
    # e.g. {"speaker_id": ["bdl"]}
    audio_path_contains: str | None = None
    # Keep only rows whose audio path contains this substring.
    # e.g. "mic1" to select one microphone set in VCTK.


def load_dataset_configs(path: str) -> list[DatasetConfig]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    return [DatasetConfig(**entry) for entry in raw]


# ---------------------------------------------------------------------------
# Encoder + caching
# ---------------------------------------------------------------------------


def encode_and_cache(
    configs: list[DatasetConfig],
    cache_dir: str,
    device: str,
    encoder_model: str = "openai/whisper-large-v3",
    batch_size: int = 16,
) -> tuple[torch.Tensor, list[dict]]:
    """Encode audio with a frozen encoder, caching results.

    For seq2seq models (e.g. Whisper) only the encoder is used.
    For encoder-only models (e.g. wav2vec2, HuBERT) the full model is used.

    Returns:
        embeddings: Float tensor of shape [N, D] where D is the encoder
            hidden size.
        metadata: List of dicts keyed by axis name as defined by each
            dataset config's ``axis_columns``.
    """
    from datasets import load_dataset
    from transformers import AutoFeatureExtractor, AutoModel

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    emb_file = cache_path / "embeddings.pt"
    meta_file = cache_path / "metadata.json"

    if emb_file.exists() and meta_file.exists():
        logger.info("Loading cached embeddings from %s", cache_dir)
        embeddings = torch.load(emb_file, map_location="cpu", weights_only=True)
        with open(meta_file) as f:
            metadata = json.load(f)
        return embeddings, metadata

    # Load and normalize datasets
    all_rows: list[dict] = []
    for cfg in configs:
        logger.info("Loading dataset %s (split=%s)", cfg.name, cfg.split)
        ds = load_dataset(cfg.name, split=cfg.split)
        n_skipped = 0
        for row in ds:
            # Audio path filter (e.g. VCTK mic selection)
            if cfg.audio_path_contains is not None:
                audio = row[cfg.audio_column]
                path = audio.get("path", "") if isinstance(audio, dict) else ""
                if cfg.audio_path_contains not in path:
                    n_skipped += 1
                    continue

            # Column value filters
            skip = False
            for col, vals in cfg.include_values.items():
                if str(row.get(col, "")) not in vals:
                    skip = True
                    break
            if not skip:
                for col, vals in cfg.exclude_values.items():
                    if str(row.get(col, "")) in vals:
                        skip = True
                        break
            if skip:
                n_skipped += 1
                continue

            row_data: dict = {"audio": row[cfg.audio_column]}
            for axis_name, col in cfg.axis_columns.items():
                row_data[axis_name] = row.get(col, "")
            all_rows.append(row_data)

        if n_skipped:
            logger.info("Skipped %d rows from %s after filtering", n_skipped, cfg.name)

    logger.info("Encoding %d samples with %s", len(all_rows), encoder_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_model)
    full_model = AutoModel.from_pretrained(encoder_model)
    # For seq2seq models use encoder only; for encoder-only models use as-is.
    encoder = getattr(full_model, "encoder", full_model)
    if encoder is not full_model:
        del full_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    encoder = encoder.to(device).eval()

    all_embeddings = []
    metadata = []

    for i in tqdm(range(0, len(all_rows), batch_size), desc="Encoding"):
        batch_rows = all_rows[i : i + batch_size]
        audios = []
        for row in batch_rows:
            audio = row["audio"]
            if isinstance(audio, dict):
                waveform = audio["array"]
                sr = audio["sampling_rate"]
            else:
                waveform = audio
                sr = 16000
            audios.append(waveform)
            metadata.append({k: v for k, v in row.items() if k != "audio"})

        inputs = feature_extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)
            # Mean-pool over time dimension → [B, D]
            hidden = outputs.last_hidden_state
            emb = hidden.mean(dim=1).cpu()
            all_embeddings.append(emb)

    embeddings = torch.cat(all_embeddings, dim=0)

    # Save cache
    torch.save(embeddings, emb_file)
    with open(meta_file, "w") as f:
        json.dump(metadata, f)
    logger.info("Cached %d embeddings to %s", len(embeddings), cache_dir)

    return embeddings, metadata


# ---------------------------------------------------------------------------
# Axis-aware contrastive dataset
# ---------------------------------------------------------------------------


@dataclass
class AxisIndex:
    """Maps label values to sample indices for one axis."""

    name: str
    groups: dict[str, list[int]] = field(default_factory=dict)

    def build(self, metadata: list[dict], key: str) -> None:
        self.groups.clear()
        for idx, m in enumerate(metadata):
            label = m.get(key, "")
            if label:
                self.groups.setdefault(label, []).append(idx)
        # Remove singleton groups (no positive pair possible)
        self.groups = {k: v for k, v in self.groups.items() if len(v) > 1}

    @property
    def valid_indices(self) -> set[int]:
        """Indices that belong to at least one non-singleton group."""
        result = set()
        for indices in self.groups.values():
            result.update(indices)
        return result


def build_axis_indices(
    metadata: list[dict], axes: list[str]
) -> dict[str, AxisIndex]:
    """Build grouping indices for each axis."""
    result = {}
    for axis in axes:
        ai = AxisIndex(name=axis)
        ai.build(metadata, axis)
        if ai.groups:
            result[axis] = ai
            logger.info(
                "Axis %r: %d groups, %d samples",
                axis,
                len(ai.groups),
                len(ai.valid_indices),
            )
        else:
            logger.warning("Axis %r: no valid groups found", axis)
    return result


class ContrastiveAxisDataset(Dataset):
    """Yields (anchor_idx, {axis: positive_idx}) tuples."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        metadata: list[dict],
        axis_indices: dict[str, AxisIndex],
    ) -> None:
        self.embeddings = embeddings
        self.metadata = metadata
        self.axis_indices = axis_indices

        # Valid anchors: samples that appear in all axis groups
        valid = None
        for ai in axis_indices.values():
            if valid is None:
                valid = ai.valid_indices
            else:
                valid = valid & ai.valid_indices
        self.anchor_indices = sorted(valid) if valid else []

        # For speaker axis: build reverse map (index → label)
        self._index_to_label: dict[str, dict[int, str]] = {}
        for axis_name, ai in axis_indices.items():
            rev = {}
            for label, indices in ai.groups.items():
                for idx in indices:
                    rev[idx] = label
            self._index_to_label[axis_name] = rev

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, item: int) -> dict:
        anchor_idx = self.anchor_indices[item]
        result = {"anchor_idx": anchor_idx}

        for axis_name, ai in self.axis_indices.items():
            label = self._index_to_label[axis_name].get(anchor_idx)
            if label is None:
                # Shouldn't happen since we filtered, but be safe
                result[f"{axis_name}_pos_idx"] = anchor_idx
                continue

            candidates = ai.groups[label]

            if axis_name == "accent":
                # For accent: positive must be different speaker
                anchor_speaker = self.metadata[anchor_idx].get("speaker", "")
                diff_speaker = [
                    c
                    for c in candidates
                    if c != anchor_idx
                    and self.metadata[c].get("speaker", "") != anchor_speaker
                ]
                pool = diff_speaker if diff_speaker else [
                    c for c in candidates if c != anchor_idx
                ]
            else:
                # speaker/content: same group, different utterance
                pool = [c for c in candidates if c != anchor_idx]

            pos_idx = random.choice(pool) if pool else anchor_idx
            result[f"{axis_name}_pos_idx"] = pos_idx

        return result


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """In-batch InfoNCE loss.

    Args:
        anchor: [B, D] normalized embeddings.
        positive: [B, D] normalized embeddings, where positive[i]
            is the positive match for anchor[i].
        temperature: Scaling factor for logits.

    Returns:
        Scalar loss.
    """
    # Similarity matrix: [B, B]
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(len(anchor), device=anchor.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    embeddings: torch.Tensor,
    metadata: list[dict],
    axes: dict[str, int],
    hidden_dim: int | None,
    output_dir: str,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    temperature: float = 0.05,
    axis_weights: dict[str, float] | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> None:
    """Train MultiAxisProjection heads with contrastive losses."""
    random.seed(seed)
    torch.manual_seed(seed)

    axis_indices = build_axis_indices(metadata, list(axes.keys()))
    if not axis_indices:
        raise ValueError("No valid axes found. Check metadata labels.")

    dataset = ContrastiveAxisDataset(embeddings, metadata, axis_indices)
    logger.info("Training on %d anchor samples", len(dataset))

    if len(dataset) == 0:
        raise ValueError("No samples with valid groups across all axes.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    projection = MultiAxisProjection(
        in_features=embeddings.shape[1],
        axes=axes,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    if axis_weights is None:
        axis_weights = {a: 1.0 for a in axis_indices}

    embeddings = embeddings.to(device)
    loss_history: list[dict[str, float]] = []

    for epoch in range(epochs):
        projection.train()
        epoch_losses: dict[str, float] = defaultdict(float)
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            anchor_idx = batch["anchor_idx"]
            anchor_emb = embeddings[anchor_idx]

            features = {"sentence_embedding": anchor_emb}
            out = projection(features)

            total_loss = torch.tensor(0.0, device=device)

            for axis_name in axis_indices:
                pos_idx = batch[f"{axis_name}_pos_idx"]
                pos_emb = embeddings[pos_idx]

                pos_features = {"sentence_embedding": pos_emb}
                pos_out = projection(pos_features)

                anchor_proj = F.normalize(
                    out[f"embedding_{axis_name}"], dim=-1
                )
                pos_proj = F.normalize(
                    pos_out[f"embedding_{axis_name}"], dim=-1
                )

                loss = info_nce_loss(anchor_proj, pos_proj, temperature)
                weight = axis_weights.get(axis_name, 1.0)
                total_loss = total_loss + weight * loss
                epoch_losses[axis_name] += loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            n_batches += 1

            pbar.set_postfix(loss=total_loss.item())

        scheduler.step()

        avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        avg_losses["total"] = sum(avg_losses.values())
        loss_history.append(avg_losses)

        logger.info(
            "Epoch %d/%d — %s",
            epoch + 1,
            epochs,
            " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()),
        )

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    projection.eval()
    projection.cpu()
    projection.save(str(out_path))

    training_meta = {
        "axes": axes,
        "hidden_dim": hidden_dim,
        "in_features": embeddings.shape[1],
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "temperature": temperature,
        "axis_weights": axis_weights,
        "num_samples": len(dataset),
        "loss_history": loss_history,
    }
    with open(out_path / "training_meta.json", "w") as f:
        json.dump(training_meta, f, indent=2)

    logger.info("Saved trained projection to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_axes(axes_strs: list[str]) -> dict[str, int]:
    """Parse 'name:dim' axis specifications."""
    result = {}
    for s in axes_strs:
        name, dim = s.split(":")
        result[name.strip()] = int(dim.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MultiAxisProjection heads with contrastive loss."
    )
    parser.add_argument(
        "--dataset_config",
        required=True,
        help="JSON file with dataset column mappings.",
    )
    parser.add_argument(
        "--encoder_model",
        default="openai/whisper-large-v3",
        help="HuggingFace model name or path for the frozen encoder.",
    )
    parser.add_argument(
        "--cache_dir",
        default="./encoder_cache",
        help="Directory for cached encoder embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        default="./trained_projection",
        help="Directory to save trained projection.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["content:256", "speaker:256", "accent:128"],
        help="Axis specs as name:dim pairs.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dim for MLP heads. Set to 0 for single linear layer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=16,
        help="Batch size for encoder inference.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="InfoNCE temperature.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    configs = load_dataset_configs(args.dataset_config)
    axes = parse_axes(args.axes)
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else None

    embeddings, metadata = encode_and_cache(
        configs,
        cache_dir=args.cache_dir,
        device=args.device,
        encoder_model=args.encoder_model,
        batch_size=args.encode_batch_size,
    )

    train(
        embeddings=embeddings,
        metadata=metadata,
        axes=axes,
        hidden_dim=hidden_dim,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
