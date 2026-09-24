from transformers import Wav2Vec2CTCTokenizer

from ctc_vocab import build_ctc_vocab, collect_ctc_units


def test_collect_ctc_units_from_string_examples():
    examples = [{"text": "ab cd"}]
    assert collect_ctc_units(examples, "text") == ["a", "b", "c", "d", "|"]


def test_collect_ctc_units_from_token_lists():
    examples = [{"phonemes": ["ɑː", "b", "|", "<ha>"]}]
    assert collect_ctc_units(examples, "phonemes") == ["<ha>", "b", "|", "ɑː"]


def test_build_ctc_vocab_keeps_special_tokens_first():
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(".")
    vocab = build_ctc_vocab(["ɑː", "b", "|"], tokenizer)

    assert vocab["<pad>"] == 0
    assert vocab["<s>"] == 1
    assert vocab["</s>"] == 2
    assert vocab["<unk>"] == 3
    assert set(vocab) == {"<pad>", "<s>", "</s>", "<unk>", "ɑː", "b", "|"}
