#!/usr/bin/env bash
# claudemail-setup.sh — one-shot setup script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_DIR="$SCRIPT_DIR/mcp-server"
CLAUDE_CONFIG="$HOME/.claude.json"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  claudemail setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Check prerequisites ────────────────────────────────
echo ""
echo "▶ Checking prerequisites..."
for cmd in docker node npm; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "  ✗ $cmd not found. Please install it first."
    exit 1
  fi
  echo "  ✓ $cmd found"
done

# ── 2. Start Mailpit ──────────────────────────────────────
echo ""
echo "▶ Starting Mailpit in Docker..."
cd "$SCRIPT_DIR"

# Try 'docker compose' (v2 plugin), fall back to direct docker run
if docker compose version &>/dev/null 2>&1; then
  docker compose up -d
else
  echo "  (docker compose v2 not found, using docker run directly)"
  docker rm -f claudemail 2>/dev/null || true
  docker run -d \
    --name claudemail \
    --restart unless-stopped \
    -p 1025:1025 \
    -p 1143:1143 \
    -p 8025:8025 \
    -e MP_SMTP_AUTH_ACCEPT_ANY=true \
    -e MP_SMTP_AUTH_ALLOW_INSECURE=true \
    -e MP_IMAP_AUTH_ACCEPT_ANY=true \
    -e MP_IMAP_AUTH_ALLOW_INSECURE=true \
    -e MP_MAX_MESSAGES=500 \
    -v mailpit_data:/data \
    axllent/mailpit:latest
fi
echo "  ✓ Mailpit running"
echo "  ✓ Web UI: http://localhost:8025"
echo "  ✓ SMTP:   localhost:1025"
echo "  ✓ IMAP:   localhost:1143"

# ── 3. Install MCP server deps ────────────────────────────
echo ""
echo "▶ Installing MCP server dependencies..."
cd "$MCP_DIR"
npm install --silent
echo "  ✓ npm packages installed"

# ── 4. Make MCP server executable ────────────────────────
chmod +x "$MCP_DIR/index.js"

# ── 5. Configure Apple Mail ──────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Apple Mail setup (IMAP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Add this account in Mail → Settings → Accounts → +"
echo ""
echo "  IMAP (incoming):"
echo "    Server:    localhost"
echo "    Port:      1143"
echo "    Username:  you@localhost"
echo "    Password:  anything"
echo "    TLS:       OFF (plain)"
echo ""
echo "  SMTP (outgoing — for replies):"
echo "    Server:    localhost"
echo "    Port:      1025"
echo "    Username:  you@localhost"
echo "    Password:  anything"
echo "    TLS:       OFF (plain)"
echo ""

# ── 6. Print Claude Code MCP config ──────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Claude Code MCP config"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Run this to add claudemail to Claude Code:"
echo ""
echo "  claude mcp add claudemail \\"
echo "    -e CLAUDEMAIL_TO=you@localhost \\"
echo "    -e CLAUDEMAIL_FROM=claude@localhost \\"
echo "    -e CLAUDEMAIL_MAILPIT_URL=http://localhost:8025 \\"
echo "    -- node $MCP_DIR/index.js"
echo ""
echo "  Or add manually to ~/.claude.json mcpServers:"
cat <<JSON
  {
    "claudemail": {
      "command": "node",
      "args": ["$MCP_DIR/index.js"],
      "env": {
        "CLAUDEMAIL_TO": "you@localhost",
        "CLAUDEMAIL_FROM": "claude@localhost",
        "CLAUDEMAIL_MAILPIT_URL": "http://localhost:8025"
      }
    }
  }
JSON

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Test it with:"
echo "    cd $SCRIPT_DIR && node scripts/test-send.js"
echo ""
