# claudemail — guidance for Claude Code

You have access to the `claudemail-mcp` MCP server. Use it to ask the user
questions over email when you need human input to proceed.

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
# 1. Send the question
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
    # 3. Use reply["answer"] to proceed
    ...
else:
    # 4. Decide a sensible default and note it
    ...
```

## Tone

Write emails like a thoughtful colleague, not a chatbot. Short sentences.
No filler phrases. The user is busy; respect their time.
