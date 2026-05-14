---
subject: "Re: LRE §4 — what does the acoustic evidence look like?"
from: Jim O'Regan
date: 2026-04-26 16:16
message_id: AC9FC6A4-752A-4907-B3E5-FFB7DC08A4E6@localhost
---

← [[2026-04-26 15-47 [claude] LRE §4 — what does the acoustic evidence look like]]
→ [[2026-04-26 16-19 [jim] LRE §4 — what does the acoustic evidence look like]]

Ok… on this topic, I’ll need to cite and describe this paper: https://www.isca-archive.org/interspeech_2025/guo25d_interspeech.html

I also need to get a reference for the Levenshtein automaton from https://en.wikipedia.org/wiki/Levenshtein_automaton

I’m heavily paraphrasing the Guo paper, based on what I **think** is really happening; I need to get this double checked/read for critical feedback.

This text is somewhat incoherent, lacks sufficient detail, and should trigger multiple followup questions aimed at fixing that!

```
@inproceedings{guo25d_interspeech,
  title     = {{Dysfluent WFST: A Framework for Zero-Shot Speech Dysfluency Transcription and Detection}},
  author    = {Chenxu Guo and Jiachen Lian and Xuanru Zhou and Jinming Zhang and Shuhe Li and Zongli Ye and Peter Park and Anaisha Das and Zoe Ezzes and Jet Vonk and Brittany Morin and Rian Bogley and Lisa Wauters and Zachary Miller and Maria Gorno-Tempini and Gopala Anumanchipalli},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {2205--2209},
  doi       = {10.21437/Interspeech.2025-2446},
  issn      = {2958-1796},
}
```

---

The Dysfluent WFST framework [@guo25d_interspeech] generates an alignment based on two inputs: the speech signal, and the reference text. From the speech signal, it generates an emission matrix from a speech encoder, such as wav2vec 2 or WavLM, to represent the log posterior probabilities of phone classes over frames of speech. From the reference text, a weighted finite-state transducer (WFST) is created by composing a weighted finite-state automaton (WFSA) with a Levenshtein transducer: that is, a transducer that encapsulates the edit operations used in the Levenshtein distance.

[I need to come back to this]

We extend the Dysfluent WFST framework with an additional WFST, which incorporates known phonological rules that are given a higher weight than general edit-distance transductions: for example, the phonological rule n > m /_ # [+labial] — that is, the phone “n” becomes “m” at the end of a word if the beginning of the next word starts with a phone with the “labial” feature (e.g., f, v, b, p).

[This needs more work]

We use our own implementation of dysfluent WFST, both to allow the introduction of this additional WFST that more directly represents phonological rules, but also so that the construction and composition of the WFST(s) is done using OpenFST, which has a richer set of user-facing operations, and existing tool support. The resulting WFSTs are imported into K2, so the rest of the decoding process happens in a similar manner to Dysfluent WFST.


> On 26 Apr 2026, at 17:47, claude@localhost wrote:
> 
> Context
> -------
> Building §4 of the LRE paper. The rule format (before/match/after/output, YAML-based) is now described. Moving to the acoustic validation step.
> 
> Question
> --------
> So the rule generates a candidate variant — say, "n" surfaces as "m" before a labial. You now have a phonological hypothesis.
> 
> How does the system check whether that variant actually appears in the recordings? What does the evidence look like — is it a score, a lattice, a count of tokens? And what threshold or criterion decides "this variant is real"?
> 
> Reply to this email to answer. Claude Code is waiting.
> 
> [claudemail:cde8bc10-de16-4422-b767-8c39344f2094]
> Sent at: 4/26/2026, 5:47:42 PM
> 


