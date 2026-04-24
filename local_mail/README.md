# claudemail

> Claude Code asks you questions by email. You reply. It continues.

Exploits the fact that replying to an email feels easy and natural,
while staring at a blank terminal prompt does not.

## How it works

```
Claude Code → ask_question() → SMTP → Mailpit → POP3 → Apple Mail
                                                              ↓
Claude Code ← wait_for_reply() ← Mailpit REST API ← your reply
```

- **Mailpit** runs in Docker: a tiny (~10 MB) local mail server with SMTP, POP3, and a web UI
- **claudemail-mcp** is a Node.js MCP server that gives Claude Code two tools:
  - `ask_question` — sends you an email
  - `wait_for_reply` — polls Mailpit until you reply (or timeout)
  - `check_inbox` — debug tool to see what's in the inbox

On top of the mail transport, this repo now includes a staged writing workflow:
- **Respondent** for getting words onto the page
- **Supervisor** once you have paragraphs and need structure
- **Harsh reviewer** once there is enough draft to critique hard

Replies are matched by normal mail thread metadata first (`Message-ID`,
`In-Reply-To`, `References`), with a conversation token in the body as a
fallback. That makes the loop much more reliable than matching on subject alone.

## Quick start

```bash
# 1. Start everything
./setup.sh

# 2. Add the MCP server to Claude Code (path will be printed by setup.sh)
claude mcp add claudemail -e CLAUDEMAIL_TO=you@localhost ... -- node .../index.js

# 3. Test it
node scripts/test-send.js
```

When Claude uses the tools, it should pass the full `ask_question` result into
`wait_for_reply`, especially `subject`, `sent_at`, `conversation_id`, and
`message_id`.

## Apple Mail config

| Setting   | Value         |
|-----------|---------------|
| Protocol  | POP           |
| Host      | 127.0.0.1     |
| Port      | 1110          |
| Username  | you           |
| Password  | anything      |
| TLS       | OFF           |
| SMTP host | 127.0.0.1     |
| SMTP port | 1025          |

## Mailpit web UI

http://localhost:8025 — see all messages, great for debugging.

## Environment variables

| Variable                  | Default                  | Description              |
|---------------------------|--------------------------|--------------------------|
| `CLAUDEMAIL_TO`           | `you@localhost`          | Your email address       |
| `CLAUDEMAIL_FROM`         | `claude@localhost`       | Claude's "from" address  |
| `CLAUDEMAIL_SMTP_HOST`    | `localhost`              | SMTP host                |
| `CLAUDEMAIL_SMTP_PORT`    | `1025`                   | SMTP port                |
| `CLAUDEMAIL_MAILPIT_URL`  | `http://localhost:8025`  | Mailpit REST API base    |

## Files

```
claudemail/
├── index.js               # thin wrapper to the MCP server
├── docker-compose.yml     # Mailpit container
├── setup.sh               # One-shot setup
├── CLAUDE.md              # Prompt guidance for Claude Code
├── WRITING_AGENTS.md      # staged respondent/supervisor/reviewer workflow
├── mcp-server/
│   ├── package.json
│   ├── index.js           # MCP server entrypoint
│   ├── lib/claudemail.js  # reply matching and parsing
│   └── test/              # node:test coverage for reply parsing
└── scripts/
    └── test-send.js       # Smoke test
```

## Suggested Writing Use

Use the respondent role until you have at least a couple of paragraphs. Then
switch to the supervisor role to diagnose shape and decide the next moves.
Only after that should you bring in a harsh reviewer to attack vagueness,
padding, and weak paragraphs. The full role definitions live in
`WRITING_AGENTS.md`.
