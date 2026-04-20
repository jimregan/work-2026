from sacremoses import MosesSentenceSplitter


def split_sentences(text: str, lang: str = "en") -> list[str]:
    splitter = MosesSentenceSplitter(lang)
    sentences = splitter.split(text.splitlines())
    return [s.strip() for s in sentences if s.strip()]
