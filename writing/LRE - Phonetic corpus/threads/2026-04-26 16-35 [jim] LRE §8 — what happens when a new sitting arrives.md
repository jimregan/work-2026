---
subject: "Re: LRE §8 — what happens when a new sitting arrives?"
from: Jim O'Regan
date: 2026-04-26 16:35
message_id: 8ACB4578-3DA2-47C4-B3E8-9CCACCEC493E@localhost
---

← [[2026-04-26 15-58 [claude] LRE §8 — what happens when a new sitting arrives]]
→ [[2026-04-26 16-53 [claude] LRE §8 — the third model and the phonetic transcri]]

New speeches are presented at regular intervals on the Riksdag website, published via their API. The playback system for speeches within a debate is an example of their API in operation: clickable links to each speech are created on the basis of the API document, and the video player queued to the correct location in a video based on its timestamps.

We employ the same API output as a means of continuously growing our corpus; however, the mechanism employed is configurable, and not tied specifically to the Riksdag API. [We are considering a small experiment using a YouTube channel].

The API is queried at regular, infrequent intervals (no more than once per day) to locate new speeches. The API document for each new speech is downloaded; this then provides references to the video files containing the speeches, as well as broadly timestamped sub-documents containing the official transcript of each speech. The videos are then processed using speech recognition models: which models are employed is a matter of configuration. As we know in advance that the official transcripts are not necessarily verbatim reproductions of what was spoken, we employ two ASR models: for Riksdag, we use the same models used in the creation of rixvox2, with the same rationale: as these ASR models differ greatly in how they produce output, we consider it highly likely that where they intersect will tend to be as a result of the fidelity of the output; we then perform an alignment pass against the official transcript, before running a third model, to produce a phonetic transcript.

^ I missed some things there, and tapered at the end.

> On 26 Apr 2026, at 17:58, claude@localhost wrote:
> 
> Context
> -------
> Section 8: Continuous expansion pipeline. This is one of the clearest novelty claims — most corpora are static, this one grows automatically.
> 
> Question
> --------
> Walk me through the pipeline in concrete terms. A new Riksdag sitting happened yesterday. What are the steps to get it into the corpus — from raw audio to updated lexicon?
> 
> Don't worry about completeness. Just the main stages in order.
> 
> Reply to this email to answer. Claude Code is waiting.
> 
> [claudemail:b38f769f-d27a-42eb-8365-2b7b0e61fe77]
> Sent at: 4/26/2026, 5:58:43 PM
> 


