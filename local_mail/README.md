# claudemail

> Claude Code asks you questions by email. You reply. It continues.

Exploits the fact that replying to an email feels easy and natural,
while staring at a blank terminal prompt does not.

## How it works

```
Claude Code → ask_question() → SMTP → Mailpit → IMAP → Apple Mail
                                                              ↓
Claude Code ← wait_for_reply() ← Mailpit REST API ← your reply
```

- **Mailpit** runs in Docker: a tiny (~10 MB) local mail server with SMTP, IMAP, and a web UI
- **claudemail-mcp** is a Node.js MCP server that gives Claude Code two tools:
  - `ask_question` — sends you an email
  - `wait_for_reply` — polls Mailpit until you reply (or timeout)
  - `check_inbox` — debug tool to see what's in the inbox

## Quick start

```bash
# 1. Start everything
./setup.sh

# 2. Add the MCP server to Claude Code (path will be printed by setup.sh)
claude mcp add claudemail -e CLAUDEMAIL_TO=you@localhost ... -- node .../index.js

# 3. Test it
node scripts/test-send.js
```

## Apple Mail config

| Setting   | Value         |
|-----------|---------------|
| Protocol  | IMAP          |
| Host      | localhost     |
| Port      | 1143          |
| Username  | you@localhost |
| Password  | anything      |
| TLS       | OFF           |
| SMTP host | localhost     |
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
| `CLAUDEMAIL_IMAP_HOST`    | `localhost`              | IMAP host                |
| `CLAUDEMAIL_IMAP_PORT`    | `1143`                   | IMAP port                |
| `CLAUDEMAIL_MAILPIT_URL`  | `http://localhost:8025`  | Mailpit REST API base    |

## Files

```
claudemail/
├── docker-compose.yml     # Mailpit container
├── setup.sh               # One-shot setup
├── CLAUDE.md              # Prompt guidance for Claude Code
├── mcp-server/
│   ├── package.json
│   └── index.js           # MCP server (ask_question, wait_for_reply)
└── scripts/
    └── test-send.js       # Smoke test
```
