"""Stage 2: CTC word-level alignment with wav2vec2."""

import numpy as np

DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLE_RATE = 16000


def load_model(model_name: str = None):
    """Load a wav2vec2 model and processor.

    Returns (model, processor).
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if model_name is None:
        model_name = DEFAULT_MODEL
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    return model, processor


def extract_audio_segment(
    audio_path: str, start: float, end: float
) -> np.ndarray:
    """Extract an audio segment and return as a 16kHz mono numpy array.

    Args:
        audio_path: path to audio file (mp3, wav, etc.)
        start: start time in seconds
        end: end time in seconds

    Returns:
        numpy array of float32 samples at 16kHz
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    segment = audio[int(start * 1000) : int(end * 1000)]
    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    # Normalise to [-1, 1]
    max_val = np.iinfo(np.int16).max
    samples = samples / max_val
    return samples


def align_with_transcript(
    audio_array: np.ndarray,
    words: list[str],
    model,
    processor,
) -> list[dict]:
    """Align words to audio using CTC segmentation.

    Args:
        audio_array: 16kHz mono float32 numpy array
        words: list of words to align (ground truth)
        model: wav2vec2 model
        processor: wav2vec2 processor

    Returns:
        list of dicts with keys: text, start, end, conf
    """
    import torch
    import ctc_segmentation

    if not words:
        return []

    inputs = processor(
        audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_np = log_probs.squeeze(0).cpu().numpy()

    vocab = processor.tokenizer.get_vocab()
    char_list = [None] * len(vocab)
    for char, idx in vocab.items():
        char_list[idx] = char

    config = ctc_segmentation.CtcSegmentationParameters(
        char_list=char_list,
        index_duration=audio_array.shape[0] / (log_probs_np.shape[0] * SAMPLE_RATE),
    )

    ground_truth = list(words)
    timings, char_probs, state = ctc_segmentation.ctc_segmentation(
        log_probs_np, ground_truth, config
    )

    result = []
    for i, word in enumerate(words):
        result.append({
            "text": word,
            "start": round(float(timings[i, 0]), 3),
            "end": round(float(timings[i, 1]), 3),
            "conf": round(float(char_probs[i]), 3),
        })
    return result


def get_word_timestamps(
    audio_array: np.ndarray,
    model,
    processor,
) -> list[dict]:
    """Get word timestamps using the model's own CTC decoding.

    Fallback for boilerplate/unmatched segments where we don't have
    ground truth text.

    Returns:
        list of dicts with keys: text, start, end, conf
    """
    import torch

    if len(audio_array) == 0:
        return []

    inputs = processor(
        audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)

    # Duration of each frame in seconds
    frame_duration = audio_array.shape[0] / (logits.shape[1] * SAMPLE_RATE)

    vocab = processor.tokenizer.get_vocab()
    id_to_char = {v: k for k, v in vocab.items()}
    blank_id = processor.tokenizer.pad_token_id

    # Group consecutive non-blank frames into characters, then words
    chars = []
    for i, pred_id in enumerate(predicted_ids.tolist()):
        if pred_id == blank_id:
            continue
        # Skip repeated characters
        if chars and chars[-1]["id"] == pred_id and chars[-1]["frame_end"] == i - 1:
            chars[-1]["frame_end"] = i
            continue
        char = id_to_char.get(pred_id, "")
        prob = float(log_probs[i, pred_id].exp())
        chars.append({
            "char": char,
            "id": pred_id,
            "frame_start": i,
            "frame_end": i,
            "prob": prob,
        })

    # Merge characters into words (split on space / word boundary token)
    words = []
    current_word = ""
    current_start = None
    current_probs = []

    for c in chars:
        ch = c["char"]
        if ch in ("|", " ", "▁"):
            if current_word:
                words.append({
                    "text": current_word,
                    "start": round(current_start * frame_duration, 3),
                    "end": round(c["frame_start"] * frame_duration, 3),
                    "conf": round(float(np.mean(current_probs)), 3),
                })
                current_word = ""
                current_start = None
                current_probs = []
        else:
            if current_start is None:
                current_start = c["frame_start"]
            current_word += ch
            current_probs.append(c["prob"])

    if current_word:
        last_frame = chars[-1]["frame_end"] if chars else 0
        words.append({
            "text": current_word,
            "start": round(current_start * frame_duration, 3),
            "end": round(last_frame * frame_duration, 3),
            "conf": round(float(np.mean(current_probs)), 3),
        })

    return words
