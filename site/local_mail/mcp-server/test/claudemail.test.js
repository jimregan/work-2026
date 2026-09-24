import test from "node:test";
import assert from "node:assert/strict";

import {
  buildQuestionEmail,
  extractReplyText,
  getThreadReferences,
  isReplyMatch,
  subjectMatches,
} from "../lib/claudemail.js";

test("buildQuestionEmail includes context and conversation token", () => {
  const email = buildQuestionEmail({
    body: "Which draft should I continue?",
    context: "You are helping me build a writing ritual.",
    conversationId: "1234",
    sentAt: 0,
  });

  assert.match(email, /Context/);
  assert.match(email, /\[claudemail:1234\]/);
});

test("extractReplyText removes quoted original message", () => {
  const reply = extractReplyText(`Use the second idea.

On Fri, Apr 24, 2026 at 10:00 AM Claude <claude@localhost> wrote:
> Question
> --------
> Which draft should I continue?
>
> [claudemail:1234]`);

  assert.equal(reply, "Use the second idea.");
});

test("subjectMatches tolerates common reply prefixes", () => {
  assert.equal(subjectMatches("Draft choice", "Re: Draft choice"), true);
  assert.equal(subjectMatches("Draft choice", "SV: Draft choice"), true);
  assert.equal(subjectMatches("Draft choice", "Different subject"), false);
});

test("getThreadReferences combines references and in-reply-to", () => {
  const refs = getThreadReferences({
    Headers: {
      References: ["<a@local> <b@local>"],
      "In-Reply-To": ["<c@local>"],
    },
  });

  assert.deepEqual(refs, ["<a@local>", "<b@local>", "<c@local>"]);
});

test("isReplyMatch prefers thread metadata but falls back to token", () => {
  const result = isReplyMatch({
    message: {
      Created: "2026-04-24T10:05:00Z",
      Subject: "Re: Draft choice",
      To: [{ Address: "claude@localhost" }],
    },
    fullMessage: {
      Created: "2026-04-24T10:05:00Z",
      To: [{ Address: "claude@localhost" }],
      Headers: {
        "In-Reply-To": ["<orig@local>"],
      },
      Text: "Use the second idea.",
    },
    expectedSubject: "Draft choice",
    sentAt: Date.parse("2026-04-24T10:00:00Z"),
    replyAddress: "claude@localhost",
    conversationId: "1234",
    originalMessageId: "<orig@local>",
  });

  assert.equal(result, true);
});
