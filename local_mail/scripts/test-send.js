#!/usr/bin/env node
/**
 * test-send.js — smoke test for the claudemail setup
 * Run: node scripts/test-send.js
 */
import { createRequire } from "node:module";

const requireFromMcp = createRequire(
  new URL("../mcp-server/package.json", import.meta.url)
);
const nodemailer = requireFromMcp("nodemailer");

const transport = nodemailer.createTransport({
  host: "localhost",
  port: 1025,
  secure: false,
  auth: { user: "claude", pass: "claude" },
  tls: { rejectUnauthorized: false },
});

const info = await transport.sendMail({
  from:    "claude@localhost",
  to:      "you@localhost",
  subject: "Test: claudemail is working 🎉",
  text:    [
    "This is a test message from the claudemail MCP setup.",
    "",
    "If you see this in Apple Mail, your IMAP connection is working.",
    "Reply to this message — Claude Code will be able to receive your answer.",
    "",
    "❓ Question",
    "──────────",
    "What are you currently working on? Describe it in one sentence.",
    "",
    "Reply to this email to answer.",
  ].join("\n"),
  replyTo: "claude@localhost",
});

console.log("✓ Test email sent:", info.messageId);
console.log("  Check http://localhost:8025 to see it in the web UI");
console.log("  Or check Apple Mail if IMAP is configured");
