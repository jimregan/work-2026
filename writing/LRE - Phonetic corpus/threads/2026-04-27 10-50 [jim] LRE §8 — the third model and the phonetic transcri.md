---
subject: "Re: LRE §8 — the third model and the phonetic transcript"
from: Jim O'Regan
date: 2026-04-27 10:50
message_id: D64B1746-625F-48FC-B5EB-0C5391935DE5@localhost
---

← [[2026-04-26 16-53 [claude] LRE §8 — the third model and the phonetic transcri]]
→ [[2026-04-27 13-30 [claude] LRE §8 — not forced alignment, so what]]

The third model is the one trained on the Waxholm data. It is a wav2vec2 model that outputs phonetic “word" transcripts. As with other wav2vec2 models, it’s possible to get relatively accurate word timings with DTW, but phone timings need to come from an alignment model. One aspect of this paper is specifically not using **forced** alignment.

> On 26 Apr 2026, at 18:53, claude@localhost wrote:
> 
> Context
> -------
> §8: Continuous expansion pipeline. The two-ASR-model intersection step is clear. Following up on the phonetic transcript step you said you'd tapered on.
> 
> Question
> --------
> You described two ASR models for word-level alignment, then mentioned a third model for the phonetic transcript. What is that third model, and what does its output look like — phone boundaries, posteriors, something else?
> 
> Also: you said the pipeline is configurable and not tied to the Riksdag API. What would it take to point it at a different parliament or broadcaster?
> 
> Reply to this email to answer. Claude Code is waiting.
> 
> [claudemail:512b0308-22d0-4f38-8c60-b88ed0f88ad2]
> Sent at: 4/26/2026, 6:53:35 PM
> 


