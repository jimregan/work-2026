"""MultiAxisProjectionTrainer — adapted from sentence_transformers/trainer.py.

Adapts SentenceTransformerTrainer for MultiAxisSentenceTransformer models
trained with per-axis contrastive losses.

Dataset column naming convention:

    ``anchor_{suffix}``       — anchor example
    ``{axis}_pos_{suffix}``   — positive example for each axis

where ``{suffix}`` is one of:

    ``input_ids``          text inputs (tokenised)
    ``input_features``     audio inputs (Whisper feature extractor output)
    ``sentence_embedding`` pre-cached pooled embeddings

Loss functions receive ``(named_features, labels)`` where ``named_features``
is a ``dict[str, dict[str, Tensor]]`` mapping role names to feature dicts:

    named_features["anchor"]       — anchor inputs
    named_features["content_pos"]  — positives for the content axis
    named_features["speaker_pos"]  — positives for the speaker axis
    ...

The loss function should compute InfoNCE **per axis**, using only that axis's
positive pool as in-batch negatives — never mixing positives from different
axes into the same comparison.  It should return either a scalar ``Tensor`` or
a ``dict[str, Tensor]`` mapping axis names to per-axis losses; a dict is summed
for backprop and each component is tracked and logged individually.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from packaging.version import parse as parse_version
from torch import nn
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, RandomSampler
from transformers import EvalPrediction, PreTrainedTokenizerBase, Trainer, TrainerCallback
from transformers import __version__ as transformers_version
from transformers.data.data_collator import DataCollator
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import EvalLoopOutput

from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from sentence_transformers.sampler import (
    DefaultBatchSampler,
    GroupByLabelBatchSampler,
    MultiDatasetDefaultBatchSampler,
    NoDuplicatesBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import disable_logging, is_datasets_available, is_training_available

from .sentence_transformer import MultiAxisSentenceTransformer

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value

logger = logging.getLogger(__name__)

try:
    from transformers.integrations import TrackioCallback
except ImportError:
    TrackioCallback = None


class MultiAxisProjectionTrainer(Trainer):
    """Training loop for :class:`MultiAxisSentenceTransformer` models.

    The class attribute :attr:`DEFAULT_FEATURE_SUFFIXES` lists the dataset
    column suffixes recognised by :meth:`collect_features`.  Override it in
    a subclass, or pass ``feature_suffixes`` to ``__init__``, to support
    additional encoder input types (e.g. ``"_codec_tokens"`` for a codec
    encoder).

    Wraps the HuggingFace :class:`~transformers.Trainer` with the same
    batch-sampler, dataloader, and evaluator hooks as
    ``SentenceTransformerTrainer``, but removes text-specific machinery
    (prompts, router, model card) and opens the data collator to handle
    audio feature tensors in addition to tokenised text.

    Args:
        model: The :class:`MultiAxisSentenceTransformer` to train.
        args: Training arguments.  Defaults to a basic
            :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`
            with ``output_dir="tmp_trainer"``.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        loss: Loss module (or dict of loss modules for multi-dataset training).
            Must accept ``(named_features, labels)`` where ``named_features``
            is a ``dict[str, dict[str, Tensor]]`` keyed by role
            (``"anchor"``, ``"{axis}_pos"``, …) and return a scalar ``Tensor``
            or a ``dict[str, Tensor]`` of per-axis losses.
            **Required** — no default loss is applied.
        evaluator: Optional :class:`~sentence_transformers.evaluation.SentenceEvaluator`
            (or list thereof) for evaluation metrics beyond eval loss.
        data_collator: Custom data collator.  If ``None``:

            * a :class:`~sentence_transformers.data_collator.SentenceTransformerDataCollator`
              is created when a tokenizer is available, or
            * the HuggingFace default collator is used otherwise (suitable for
              pre-cached ``sentence_embedding`` tensors).
        tokenizer: Tokenizer or feature extractor attached to the model.
            Inferred from ``model.tokenizer`` if not supplied.
        model_init: Callable returning an initialised model, for
            hyper-parameter search.
        callbacks: Additional :class:`~transformers.TrainerCallback` instances.
        optimizers: ``(optimizer, scheduler)`` tuple.
        feature_suffixes: Column suffixes that :meth:`collect_features` will
            recognise as encoder inputs.  Extends or replaces
            :attr:`DEFAULT_FEATURE_SUFFIXES`.  ``None`` uses the class default.
    """

    DEFAULT_FEATURE_SUFFIXES: tuple[str, ...] = (
        "_input_ids",
        "_input_features",
        "_sentence_embedding",
        "_pixel_values",
    )

    def __init__(
        self,
        model: MultiAxisSentenceTransformer | None = None,
        args: SentenceTransformerTrainingArguments | None = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        loss: (
            nn.Module
            | dict[str, nn.Module]
            | Callable[[MultiAxisSentenceTransformer], nn.Module]
            | dict[str, Callable[[MultiAxisSentenceTransformer], nn.Module]]
            | None
        ) = None,
        evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
        data_collator: DataCollator | None = None,
        tokenizer: PreTrainedTokenizerBase | Callable | None = None,
        model_init: Callable[[], MultiAxisSentenceTransformer] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        feature_suffixes: tuple[str, ...] | None = None,
    ) -> None:
        self.feature_suffixes = feature_suffixes if feature_suffixes is not None else self.DEFAULT_FEATURE_SUFFIXES

        if not is_training_available():
            raise RuntimeError(
                "To train a MultiAxisSentenceTransformer model, you need the `accelerate` and "
                "`datasets` packages:\n"
                "    pip install accelerate datasets"
            )

        if args is None:
            output_dir = "tmp_trainer"
            logger.info("No `SentenceTransformerTrainingArguments` passed, using `output_dir=%s`.", output_dir)
            args = SentenceTransformerTrainingArguments(output_dir=output_dir)
        elif not isinstance(args, SentenceTransformerTrainingArguments):
            raise ValueError(
                "Please use `SentenceTransformerTrainingArguments` imported from `sentence_transformers`."
            )

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                logger.warning(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. "
                    "`model_init` will overwrite your model when calling the `train` method."
                )
            self.model_init = model_init

        if compute_metrics is not None:
            logger.warning(
                "`compute_metrics` is not compatible with MultiAxisProjectionTrainer. "
                "Use the `evaluator` argument for evaluation metrics instead."
            )

        if loss is None:
            raise ValueError(
                "A `loss` function must be provided to `MultiAxisProjectionTrainer`. "
                "Provide a loss module that accepts (named_features, labels) where "
                "named_features is a dict[str, dict[str, Tensor]] keyed by role "
                "('anchor', '{axis}_pos', …), and returns a scalar Tensor or "
                "a dict[str, Tensor] of per-axis losses."
            )

        # Resolve tokenizer from model if not provided.
        if tokenizer is None and hasattr(model, "tokenizer") and isinstance(model.tokenizer, PreTrainedTokenizerBase):
            tokenizer = model.tokenizer

        if data_collator is None and tokenizer is not None:
            data_collator = SentenceTransformerDataCollator(
                tokenize_fn=model.tokenize,
                all_special_ids=set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set(),
            )
        # If data_collator is still None here, the HuggingFace default_data_collator
        # will be used, which handles pre-stacked tensors (e.g. cached sentence embeddings).

        for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
            if isinstance(dataset, IterableDataset) and dataset.column_names is None:
                sample = next(iter(dataset))
                naive_type_mapping = {str: "string", int: "int64", float: "float32", bool: "bool"}
                example_features = {
                    key: Value(naive_type_mapping.get(type(value), "null")) for key, value in sample.items()
                }
                raise ValueError(
                    f"The provided `{dataset_name}_dataset` must have Features. Specify them with e.g.:\n"
                    f"{dataset_name}_dataset = {dataset_name}_dataset.cast(Features({example_features}))\n"
                    "See the Datasets documentation for more information on dataset Features: "
                    "https://huggingface.co/docs/datasets/en/about_dataset_features"
                )

        if isinstance(train_dataset, dict) and not isinstance(train_dataset, DatasetDict):
            train_dataset = DatasetDict(train_dataset)
        if isinstance(eval_dataset, dict) and not isinstance(eval_dataset, DatasetDict):
            eval_dataset = DatasetDict(eval_dataset)

        super_kwargs = {
            "model": None if self.model_init else model,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset if eval_dataset is not None or evaluator is None else "dummy",
            "model_init": model_init,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }
        # transformers v4.46.0 renamed `tokenizer` to `processing_class`.
        if parse_version(transformers_version) >= parse_version("4.46.0"):
            super_kwargs["processing_class"] = tokenizer
        else:
            super_kwargs["tokenizer"] = tokenizer

        if eval_dataset is None and evaluator is None and args.eval_strategy != "no":
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy!r}, but you provided neither "
                "an `eval_dataset` nor an `evaluator`. "
                "Either provide one of these or set `args.eval_strategy='no'` to skip evaluation."
            )

        super().__init__(**super_kwargs)
        if self.eval_dataset == "dummy":
            self.eval_dataset = None

        # Accumulator for per-axis (dict) loss components, mirroring ST trainer.
        self.accum_loss_components: dict[str, dict] = {"train": {}, "eval": {}}

        # Every MultiAxisSentenceTransformer can always return a loss.
        self.can_return_loss = True

        self.model: MultiAxisSentenceTransformer
        self.args: SentenceTransformerTrainingArguments

        if isinstance(loss, dict):
            self.loss = {
                dataset_name: self.prepare_loss(loss_fn, model)
                for dataset_name, loss_fn in loss.items()
            }
            for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
                if dataset is None:
                    continue
                if not isinstance(dataset, dict):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then the `{dataset_name}_dataset` must be a `DatasetDict`."
                    )
                if missing := set(dataset.keys()) - set(loss.keys()):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then all keys from the `{dataset_name}_dataset` "
                        f"dictionary must occur in `loss` also. "
                        f"Currently, {sorted(missing)} occur{'s' if len(missing) == 1 else ''} in "
                        f"`{dataset_name}_dataset` but not in `loss`."
                    )
        else:
            self.loss = self.prepare_loss(loss, model)

        if evaluator is not None and not isinstance(evaluator, SentenceEvaluator):
            evaluator = SequentialEvaluator(evaluator)
        self.evaluator = evaluator

        if self.train_dataset is not None:
            self.train_dataset = self.preprocess_dataset(train_dataset, dataset_name="train")
        if self.eval_dataset is not None:
            self.eval_dataset = self.preprocess_dataset(eval_dataset, dataset_name="eval")

    # ------------------------------------------------------------------
    # Model / loss helpers (unchanged from SentenceTransformerTrainer)
    # ------------------------------------------------------------------

    def call_model_init(self, trial=None) -> MultiAxisSentenceTransformer:
        model = super().call_model_init(trial=trial)
        if not hasattr(self, "loss"):
            return model

        if isinstance(self.loss, dict):
            for key, loss_fn in self.loss.items():
                if not isinstance(loss_fn, torch.nn.Module):
                    self.loss[key] = loss_fn(model)
                elif hasattr(loss_fn, "model"):
                    self.loss = self.override_model_in_loss(self.loss, model)
        elif not isinstance(self.loss, torch.nn.Module):
            self.loss = self.loss(model)
        elif hasattr(self.loss, "model"):
            self.loss = self.override_model_in_loss(self.loss, model)
        return model

    def override_model_in_loss(
        self, loss: torch.nn.Module, model: MultiAxisSentenceTransformer
    ) -> torch.nn.Module:
        for name, child in loss.named_children():
            if name == "model" and isinstance(child, MultiAxisSentenceTransformer):
                loss.model = model
            elif isinstance(child, torch.nn.Module):
                setattr(loss, name, self.override_model_in_loss(child, model))
        return loss

    def prepare_loss(
        self,
        loss: Callable[[MultiAxisSentenceTransformer], torch.nn.Module] | torch.nn.Module,
        model: MultiAxisSentenceTransformer,
    ) -> torch.nn.Module:
        if isinstance(loss, torch.nn.Module):
            return loss.to(model.device)
        return loss(model).to(model.device)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: MultiAxisSentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        dataset_name = inputs.pop("dataset_name", None)
        named_features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        if (
            model == self.model_wrapped
            and hasattr(loss_fn, "model")
            and loss_fn.model != model
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)

        loss = loss_fn(named_features, labels)
        if isinstance(loss, dict):
            self.track_loss_components(loss)
            loss = torch.stack(list(loss.values())).sum()
        if return_outputs:
            return loss, {}
        return loss

    def track_loss_components(self, loss: dict[str, torch.Tensor]) -> None:
        training_type = "train" if self.model.training else "eval"
        for key, value in loss.items():
            if self.args.logging_nan_inf_filter and (torch.isnan(value) or torch.isinf(value)):
                if key not in self.accum_loss_components[training_type]:
                    value = torch.tensor(0.0, dtype=value.dtype, device=value.device)
                else:
                    value = self.accum_loss_components[training_type][key] / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )

            if key not in self.accum_loss_components[training_type]:
                self.accum_loss_components[training_type][key] = value
            else:
                self.accum_loss_components[training_type][key] = (
                    self.accum_loss_components[training_type][key] + value
                )

        if "steps" not in self.accum_loss_components[training_type]:
            self.accum_loss_components[training_type]["steps"] = torch.tensor(
                0, dtype=int, device=value.device
            )
        self.accum_loss_components[training_type]["steps"] += 1

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        training_type = None
        if "loss" in logs:
            training_type = "train"
        elif "eval_loss" in logs:
            training_type = "eval"

        if training_type:
            logs = logs.copy()
            accum_losses = self._nested_gather(self.accum_loss_components[training_type])
            if "steps" in accum_losses:
                steps = accum_losses.get("steps").sum().item()
                self.accum_loss_components[training_type]["steps"] *= 0

                for key, value in accum_losses.items():
                    if key == "steps":
                        continue
                    log_key = f"{training_type}_{key}" if training_type == "eval" else key
                    logs[log_key] = round((value.sum() / steps).item(), 4)
                    self.accum_loss_components[training_type][key] = torch.tensor(
                        0.0, dtype=value.dtype, device=value.device
                    )

        if start_time is not None:
            return super().log(logs, start_time)
        else:
            return super().log(logs)

    # ------------------------------------------------------------------
    # Feature collection
    # ------------------------------------------------------------------

    def collect_features(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor | None]:
        """Split a collated batch into named per-role feature dicts and labels.

        Returns a ``dict[role, feature_dict]`` rather than a flat list, so that
        loss functions can construct **axis-specific views** of the batch for
        each InfoNCE comparison.  For example:

        .. code-block:: python

            named["anchor"]        # {"sentence_embedding": Tensor[B, D]}
            named["content_pos"]   # {"sentence_embedding": Tensor[B, D]}
            named["speaker_pos"]   # {"sentence_embedding": Tensor[B, D]}

        This matters for multi-axis InfoNCE: for axis *k*, the in-batch
        negatives must come **only** from ``{axis_k}_pos``, never from
        ``{axis_j}_pos`` for *j ≠ k*.  Returning positives by name lets each
        axis comparison select its own pool, presenting a different view of
        the batch to each loss term.

        Recognised column suffixes are those in :attr:`feature_suffixes`
        (defaults to :attr:`DEFAULT_FEATURE_SUFFIXES`):

        * ``_input_ids``          — tokenised text
        * ``_input_features``     — audio (e.g. Whisper mel spectrogram)
        * ``_sentence_embedding`` — pre-cached pooled embedding
        * ``_pixel_values``       — image (CLIP-style)

        Additional suffixes can be registered by passing ``feature_suffixes``
        to the constructor or overriding :attr:`DEFAULT_FEATURE_SUFFIXES` in
        a subclass.
        """
        # Collect all role prefixes (e.g. "anchor", "content_pos", "speaker_pos")
        # preserving insertion order so anchor comes first when columns are ordered.
        roles: dict[str, dict[str, torch.Tensor]] = {}
        for column in inputs:
            for suffix in self.feature_suffixes:
                if column.endswith(suffix):
                    role = column[: -len(suffix)].rstrip("_")
                    if role not in roles:
                        roles[role] = {}
                    key = suffix.lstrip("_")  # strip leading underscore for feature dict key
                    roles[role][key] = inputs[column]
                    break

        labels = inputs.get("label", None)
        return roles, labels

    # ------------------------------------------------------------------
    # Evaluation (unchanged)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        if eval_dataset:
            eval_dataset = self.preprocess_dataset(eval_dataset, dataset_name="eval")
        else:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.evaluator is None:
            return output

        if self.is_in_train and isinstance(self.eval_dataset, dict) and metric_key_prefix.startswith("eval_"):
            if metric_key_prefix[5:] == list(self.eval_dataset.keys())[0]:
                metric_key_prefix = "eval"
            else:
                return output

        with nullcontext() if self.is_local_process_zero() else disable_logging(logging.INFO):
            output_path = self.args.output_dir
            if output_path is not None:
                output_path = os.path.join(output_path, "eval")
                os.makedirs(output_path, exist_ok=True)
            evaluator_metrics = self.evaluator(
                self.model,
                output_path=output_path,
                epoch=self.state.epoch,
                steps=self.state.global_step,
            )
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[f"{metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        output.metrics.update(evaluator_metrics)
        return output

    def _load_best_model(self) -> None:
        logger.info(
            "Loading best model from %s (score: %s).",
            self.state.best_model_checkpoint,
            self.state.best_metric,
        )
        try:
            self._load_from_checkpoint(self.state.best_model_checkpoint)
        except Exception as exc:
            logger.error(
                "Could not load the best model from %s. Error: %s",
                self.state.best_model_checkpoint,
                exc,
            )

    # ------------------------------------------------------------------
    # Batch samplers (unchanged)
    # ------------------------------------------------------------------

    def validate_column_names(self, dataset: Dataset, dataset_name: str | None = None) -> None:
        if isinstance(dataset, dict):
            for dataset_name, dataset in dataset.items():
                self.validate_column_names(dataset, dataset_name=dataset_name)
            return

        if overlap := set(dataset.column_names) & {"return_loss", "dataset_name"}:
            raise ValueError(
                f"The following column names are invalid in your "
                f"{dataset_name + ' ' if dataset_name else ''}dataset: {list(overlap)}. "
                "Avoid using these column names, as they are reserved for internal use."
            )

    def get_batch_sampler(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> BatchSampler | None:
        batch_sampler_kwargs = {
            "batch_size": batch_size,
            "drop_last": drop_last,
            "valid_label_columns": valid_label_columns,
            "generator": generator,
            "seed": seed,
        }
        if inspect.isclass(self.args.batch_sampler) and issubclass(
            self.args.batch_sampler, DefaultBatchSampler
        ):
            return self.args.batch_sampler(dataset, **batch_sampler_kwargs)

        if callable(self.args.batch_sampler):
            return self.args.batch_sampler(dataset, **batch_sampler_kwargs)

        if isinstance(dataset, IterableDataset):
            if self.args.batch_sampler != BatchSamplers.BATCH_SAMPLER:
                logger.warning("When using an IterableDataset, you cannot specify a batch sampler.")
            return None

        if self.args.batch_sampler == BatchSamplers.NO_DUPLICATES:
            return NoDuplicatesBatchSampler(dataset, **batch_sampler_kwargs)

        if self.args.batch_sampler == BatchSamplers.NO_DUPLICATES_HASHED:
            return NoDuplicatesBatchSampler(dataset, precompute_hashes=True, **batch_sampler_kwargs)

        if self.args.batch_sampler == BatchSamplers.GROUP_BY_LABEL:
            return GroupByLabelBatchSampler(dataset, **batch_sampler_kwargs)

        if self.args.batch_sampler == BatchSamplers.BATCH_SAMPLER:
            return DefaultBatchSampler(
                RandomSampler(dataset, generator=generator), **batch_sampler_kwargs
            )

    def get_multi_dataset_batch_sampler(
        self,
        dataset: ConcatDataset,
        batch_samplers: list[BatchSampler],
        generator: torch.Generator | None = None,
        seed: int | None = 0,
    ) -> BatchSampler:
        multi_batch_sampler_kwargs = {
            "batch_samplers": batch_samplers,
            "generator": generator,
            "seed": seed,
        }
        if inspect.isclass(self.args.multi_dataset_batch_sampler) and issubclass(
            self.args.multi_dataset_batch_sampler, MultiDatasetDefaultBatchSampler
        ):
            return self.args.multi_dataset_batch_sampler(dataset, **multi_batch_sampler_kwargs)

        if callable(self.args.multi_dataset_batch_sampler):
            return self.args.multi_dataset_batch_sampler(dataset, **multi_batch_sampler_kwargs)

        if self.args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.ROUND_ROBIN:
            return RoundRobinBatchSampler(dataset=dataset, **multi_batch_sampler_kwargs)

        if self.args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.PROPORTIONAL:
            return ProportionalBatchSampler(dataset=dataset, **multi_batch_sampler_kwargs)

    # ------------------------------------------------------------------
    # Dataloaders (unchanged)
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires specifying a train_dataset to MultiAxisProjectionTrainer.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(train_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.train_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )
            if self.args.batch_sampler != BatchSamplers.BATCH_SAMPLER:
                logger.warning("When using an IterableDataset, you cannot specify a batch sampler.")

        elif isinstance(train_dataset, IterableDatasetDict):
            raise ValueError(
                "MultiAxisProjectionTrainer is not compatible with IterableDatasetDict. "
                "Please use a DatasetDict instead."
            )

        elif isinstance(train_dataset, DatasetDict):
            for dataset in train_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "MultiAxisProjectionTrainer is not compatible with a DatasetDict "
                        "containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns
                    if hasattr(data_collator, "valid_label_columns")
                    else None,
                    generator=generator,
                )
                for dataset in train_dataset.values()
            ]

            train_dataset = ConcatDataset(train_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=train_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        elif isinstance(train_dataset, Dataset):
            batch_sampler = self.get_batch_sampler(
                train_dataset,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns
                if hasattr(data_collator, "valid_label_columns")
                else None,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler
        else:
            raise ValueError(
                "Unsupported `train_dataset` type. Use a Dataset, DatasetDict, or IterableDataset."
            )

        self.accelerator.even_batches = False
        self._train_dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return self._train_dataloader

    def get_eval_dataloader(
        self, eval_dataset: Dataset | DatasetDict | IterableDataset | None = None
    ) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            if self.evaluator is not None:
                return DataLoader([])
            raise ValueError(
                "Evaluation requires specifying an eval_dataset to MultiAxisProjectionTrainer."
            )

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(eval_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.eval_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )

        elif isinstance(eval_dataset, IterableDatasetDict):
            raise ValueError(
                "MultiAxisProjectionTrainer is not compatible with IterableDatasetDict. "
                "Please use a DatasetDict instead."
            )

        elif isinstance(eval_dataset, DatasetDict):
            for dataset in eval_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "MultiAxisProjectionTrainer is not compatible with a DatasetDict "
                        "containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns
                    if hasattr(data_collator, "valid_label_columns")
                    else None,
                    generator=generator,
                )
                for dataset in eval_dataset.values()
            ]

            eval_dataset = ConcatDataset(eval_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=eval_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        elif isinstance(eval_dataset, Dataset):
            batch_sampler = self.get_batch_sampler(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns
                if hasattr(data_collator, "valid_label_columns")
                else None,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        else:
            raise ValueError(
                "Unsupported `eval_dataset` type. Use a Dataset, DatasetDict, or IterableDataset."
            )

        self.accelerator.even_batches = True
        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_test_dataloader(self, test_dataset: Dataset | DatasetDict | IterableDataset) -> DataLoader:
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(test_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.eval_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )

        elif isinstance(test_dataset, IterableDatasetDict):
            raise ValueError(
                "MultiAxisProjectionTrainer is not compatible with IterableDatasetDict. "
                "Please use a DatasetDict instead."
            )

        elif isinstance(test_dataset, DatasetDict):
            for dataset in test_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "MultiAxisProjectionTrainer is not compatible with a DatasetDict "
                        "containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns
                    if hasattr(data_collator, "valid_label_columns")
                    else None,
                    generator=generator,
                )
                for dataset in test_dataset.values()
            ]

            test_dataset = ConcatDataset(test_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=test_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        elif isinstance(test_dataset, Dataset):
            batch_sampler = self.get_batch_sampler(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns
                if hasattr(data_collator, "valid_label_columns")
                else None,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        else:
            raise ValueError(
                "Unsupported `test_dataset` type. Use a Dataset, DatasetDict, or IterableDataset."
            )

        self.accelerator.even_batches = True
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if hasattr(self.args, "save_safetensors"):
            self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        else:
            self.model.save_pretrained(output_dir)

        if parse_version(transformers_version) >= parse_version("4.46.0"):
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        else:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        loaded_model = MultiAxisSentenceTransformer(checkpoint_path)
        self.model.load_state_dict(loaded_model.state_dict())

    # ------------------------------------------------------------------
    # Dataset preprocessing (prompt/router handling removed)
    # ------------------------------------------------------------------

    def preprocess_dataset(
        self,
        dataset: DatasetDict | Dataset | None = None,
        dataset_name: str | None = None,
    ) -> DatasetDict | Dataset | None:
        """Optionally add a dataset-name column for multi-dataset / multi-loss training."""
        if hasattr(dataset, "_multi_axis_preprocessed") or dataset is None:
            return dataset

        dataset = self.maybe_add_dataset_name_column(dataset, dataset_name=dataset_name)
        dataset._multi_axis_preprocessed = True
        return dataset

    def maybe_add_dataset_name_column(
        self,
        dataset: DatasetDict | Dataset | None,
        dataset_name: str | None = None,
    ) -> DatasetDict | Dataset | None:
        self.validate_column_names(dataset, dataset_name=dataset_name)

        if dataset is None or isinstance(dataset, (Dataset, IterableDataset)):
            return dataset

        # Only add dataset_name column when using per-dataset losses.
        if isinstance(self.loss, dict):
            dataset = self.add_dataset_name_column(dataset)
        return dataset

    def add_dataset_name_column(
        self,
        dataset: DatasetDict | IterableDatasetDict | Dataset | IterableDataset,
        dataset_name: str | None = None,
    ) -> DatasetDict | Dataset | None:
        if isinstance(dataset, (IterableDatasetDict, DatasetDict)):
            for dataset_name, inner_dataset in dataset.items():
                dataset[dataset_name] = self.add_dataset_name_column(
                    dataset=inner_dataset,
                    dataset_name=dataset_name,
                )
            return dataset

        if dataset_name is None:
            return dataset

        if isinstance(dataset, Dataset):
            dataset.set_transform(
                partial(self.add_dataset_name_transform, dataset_name=dataset_name, **dataset._format_kwargs)
            )
        elif isinstance(dataset, IterableDataset):
            features = dataset.features
            if dataset_name:
                features["dataset_name"] = Value("string")
            dataset = dataset.map(
                partial(self.add_dataset_name_transform, dataset_name=dataset_name),
                batched=True,
                features=features,
            )
        else:
            raise ValueError(
                "Unsupported `dataset` type. Use a Dataset, DatasetDict, IterableDataset, or IterableDatasetDict."
            )
        return dataset

    @staticmethod
    def add_dataset_name_transform(
        batch: dict[str, list[Any]],
        dataset_name: str | None = None,
        transform: Callable[[dict[str, list[Any]]], dict[str, list[Any]]] | None = None,
        **kwargs,
    ) -> dict[str, list[Any]]:
        if transform:
            batch = transform(batch)
        if not batch or not list(batch.values())[0] or dataset_name is None:
            return batch
        batch_size = len(list(batch.values())[0])
        batch["dataset_name"] = [dataset_name] * batch_size
        return batch

    # ------------------------------------------------------------------
    # Optimizer (unchanged from SentenceTransformerTrainer)
    # ------------------------------------------------------------------

    def get_optimizer_cls_and_kwargs(
        self,
        args: SentenceTransformerTrainingArguments,
        model: MultiAxisSentenceTransformer | None = None,
    ) -> tuple[Any, Any]:
        if isinstance(self.loss, dict):
            loss_model = nn.Sequential(OrderedDict(self.loss))
        else:
            loss_model = self.loss
        optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(args, loss_model)

        decay_parameters = self.get_decay_parameter_names(loss_model)
        if not {"params", "model", "optimizer_dict"} & set(optimizer_kwargs.keys()):
            optimizer_kwargs["optimizer_dict"] = [
                {
                    "params": [
                        p
                        for n, p in loss_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in loss_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        for parameter_pattern, learning_rate in args.learning_rate_mapping.items():
            optimizer_param_keys = set(optimizer_kwargs.keys()) & {"params", "model", "optimizer_dict"}
            optimizer_param_key = optimizer_param_keys.pop() if optimizer_param_keys else "optimizer_dict"

            matching_params = {
                n: p for n, p in loss_model.named_parameters() if re.search(parameter_pattern, n)
            }

            if matching_params:
                for group in optimizer_kwargs[optimizer_param_key]:
                    if "params" in group:
                        group["params"] = [
                            p
                            for p in group["params"]
                            if all(p is not param for param in matching_params.values())
                        ]
            else:
                raise ValueError(
                    f"No parameters found matching the pattern {parameter_pattern!r} in the model. "
                    "Please check the pattern and ensure it matches some of the model's parameters."
                )

            decay_parameters = self.get_decay_parameter_names(loss_model)
            matching_params_with_decay = {n: p for n, p in matching_params.items() if n in decay_parameters}
            matching_params_without_decay = {
                n: p for n, p in matching_params.items() if n not in decay_parameters
            }

            if matching_params_with_decay:
                optimizer_kwargs[optimizer_param_key].append(
                    {
                        "params": list(matching_params_with_decay.values()),
                        "lr": learning_rate,
                        "weight_decay": self.args.weight_decay,
                    }
                )

            if matching_params_without_decay:
                optimizer_kwargs[optimizer_param_key].append(
                    {
                        "params": list(matching_params_without_decay.values()),
                        "lr": learning_rate,
                        "weight_decay": 0.0,
                    }
                )

        return optimizer_cls, optimizer_kwargs
