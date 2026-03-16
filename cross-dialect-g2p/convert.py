"""ARPABET → IPA converter for CMU dict (General American).

Stress digits on vowels: 1 = primary (ˈ), 2 = secondary (ˌ), 0 = unstressed.
Stress marks are prepended to the vowel they modify.

GA note: AH maps to ə regardless of stress — GA has no phonemic /ʌ/.
"""

STRESS_MARK = {"1": "ˈ", "2": "ˌ", "0": ""}

# Base (unstressed) phone → IPA, no stress mark included
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ",
    "AH": "ə",
    "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
    "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "iː",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m",
    "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ",
    "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ",
    "UH": "ʊ", "UW": "uː",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Rhoticised vowels differ by stress in GA
_ER_BY_STRESS = {"0": "ɚ", "1": "ɝ", "2": "ɝ", "": "ɝ"}


def arpabet_to_ipa(phones: list[str]) -> str:
    """Convert list of ARPABET phones (with stress digits) to IPA string."""
    result = []
    for phone in phones:
        if phone[-1] in "012":
            stress_digit = phone[-1]
            bare = phone[:-1]
        else:
            stress_digit = ""
            bare = phone

        stress_mark = STRESS_MARK.get(stress_digit, "")

        if bare == "ER":
            result.append(stress_mark + _ER_BY_STRESS[stress_digit])
        elif bare in ARPABET_TO_IPA:
            result.append(stress_mark + ARPABET_TO_IPA[bare])
        else:
            result.append(f"[{phone}]")

    return "".join(result)
