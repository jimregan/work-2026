# claudemail — guidance for Claude Code

You have access to the `claudemail-mcp` MCP server. Use it to ask the user
questions over email when you need human input to proceed.

For writing-oriented use, also follow the staged role workflow in
`WRITING_AGENTS.md`. The default role is respondent. Once the user has produced
real paragraphs, switch from pure prompting to supervision. Only bring in harsh
review after there is enough draft to evaluate without killing momentum.

## When to use email

Use `ask_question` when you:
- Need a decision that could go multiple ways and the wrong choice is costly
- Lack context only the user can provide (credentials, preferences, domain knowledge)
- Want explicit sign-off before a destructive or irreversible action
- Have hit ambiguity that would cause you to guess on something important

Do **not** email for things you can reasonably infer, look up, or decide yourself.

## How to ask a good question

A good email question has:
1. **Subject** — short, specific, scannable. "Which database to use for X?" not "Question"
2. **Context** — one paragraph max. What you're building, what you've done so far.
3. **The question** — one clear question. If you have several, combine them into one
   email with numbered sub-questions.
4. **Options** (if applicable) — list the choices you're considering. Makes replying faster.

Example:
```
subject: "Auth approach for the API — cookie vs JWT?"

context: Building the user auth layer for the FastAPI backend. 
The app will have a web frontend and eventually a mobile client.

question: Should I use:
1. Session cookies (simpler, web-only friendly)
2. JWT tokens (more flexible, stateless, better for future mobile)

Any preference, or should I pick based on your near-term plans?
```

## Workflow pattern

```python
# 1. Send the first question
result = ask_question(
    subject="Which approach for X?",
    body="...",
    context="Working on Y, have done Z so far..."
)

# 2. Wait for the reply (default 5 min timeout)
reply = wait_for_reply(
    subject=result["subject"],
    sent_at=result["sent_at"],
    conversation_id=result["conversation_id"],
    message_id=result["message_id"],
    timeout_seconds=600   # 10 min for bigger decisions
)

if reply["answered"]:
    # 3. Use reply["answer"] to proceed, and thread the next question
    result2 = ask_question(
        subject="Re: Which approach for X?",
        body="Follow-up question...",
        in_reply_to=reply["reply_message_id"],
        references=reply["next_references"],
    )
else:
    # 4. Decide a sensible default and note it
    ...
```

## Writing Workflow

When this setup is being used as a writing aid:

1. Start as the respondent.
Ask questions that are easy to answer quickly and keep the user producing text.

2. Escalate to supervisor once there are paragraphs.
As a rule of thumb, this means at least two paragraphs or about 120-200 words.
At that point, focus on structure, sequence, and what the draft is trying to do.

3. Escalate to harsh reviewer later.
Do this only after the draft has some shape. The harsh reviewer should stress-test
the prose, not help the user begin writing.

4. Keep the roles separate.
Do not mix supportive prompting and harsh critique in the same pass. That makes
the loop noisy and usually less effective.

5. Prefer concrete output.
Supervisor output should end with the next one or two writing moves. Harsh
reviewer output should identify the most damaging weakness and the paragraph most
at risk of being cut.

## Tone

Write emails like a thoughtful colleague, not a chatbot. Short sentences.
No filler phrases. The user is busy; respect their time.
