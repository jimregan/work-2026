digit_or_space = pynini.closure(NEMO_DIGIT | pynini.accep(" ")
        cardinal_format = (NEMO_DIGIT - "0") + pynini.closure(digit_or_space + NEMO_DIGIT, 0, 1)