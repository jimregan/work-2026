#!/usr/bin/env node
/**
 * claudemail-mcp
 * MCP server that lets Claude Code ask you questions via email.
 * You reply in your mail client → Claude Code polls for the answer.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import nodemailer from "nodemailer";

// ── Config (override with env vars) ──────────────────────────────────────────
const SMTP_HOST  = process.env.CLAUDEMAIL_SMTP_HOST  || "localhost";
const SMTP_PORT  = parseInt(process.env.CLAUDEMAIL_SMTP_PORT  || "1025");
const IMAP_HOST  = process.env.CLAUDEMAIL_IMAP_HOST  || "localhost";
const IMAP_PORT  = parseInt(process.env.CLAUDEMAIL_IMAP_PORT  || "1143");
const MAILPIT_URL= process.env.CLAUDEMAIL_MAILPIT_URL || "http://localhost:8025";
const YOUR_EMAIL = process.env.CLAUDEMAIL_TO          || "you@localhost";
const FROM_EMAIL = process.env.CLAUDEMAIL_FROM        || "claude@localhost";

// ── SMTP transport ────────────────────────────────────────────────────────────
const transport = nodemailer.createTransport({
  host: SMTP_HOST,
  port: SMTP_PORT,
  secure: false,
  auth: { user: "claude", pass: "claude" },
  tls: { rejectUnauthorized: false },
});

// ── Mailpit REST API helpers ──────────────────────────────────────────────────
async function listMessages() {
  const res = await fetch(`${MAILPIT_URL}/api/v1/messages`);
  if (!res.ok) throw new Error(`Mailpit API error: ${res.status}`);
  return res.json();
}

async function getMessage(id) {
  const res = await fetch(`${MAILPIT_URL}/api/v1/message/${id}`);
  if (!res.ok) throw new Error(`Mailpit API error: ${res.status}`);
  return res.json();
}

async function deleteMessage(id) {
  await fetch(`${MAILPIT_URL}/api/v1/message/${id}`, { method: "DELETE" });
}

// ── Find a reply to a sent message by subject ─────────────────────────────────
async function findReply(subject, sentAt) {
  const data = await listMessages();
  const messages = data.messages || [];
  const reSubject = `Re: ${subject}`;

  for (const msg of messages) {
    const msgTime = new Date(msg.Created).getTime();
    const isReply = msg.Subject === reSubject || msg.Subject.includes(subject);
    const isNewer = msgTime > sentAt;
    const toClaudeAddress = (msg.To || []).some(
      (t) => t.Address === FROM_EMAIL
    );

    if (isReply && isNewer && toClaudeAddress) {
      const full = await getMessage(msg.ID);
      return {
        id:      msg.ID,
        subject: msg.Subject,
        from:    msg.From?.Address,
        body:    full.Text || full.HTML || "(empty reply)",
        receivedAt: msg.Created,
      };
    }
  }
  return null;
}

// ── Tool: ask_question ────────────────────────────────────────────────────────
async function askQuestion({ subject, body, context }) {
  const now    = Date.now();
  const fullSubject = subject || "Claude needs your input";

  const emailBody = [
    context ? `📋 Context\n──────────\n${context}\n` : null,
    `❓ Question\n──────────\n${body}`,
    `\nReply to this email to answer. Claude Code is waiting.`,
    `\n─\nSent at: ${new Date(now).toLocaleString()}`,
  ]
    .filter(Boolean)
    .join("\n");

  await transport.sendMail({
    from:    FROM_EMAIL,
    to:      YOUR_EMAIL,
    subject: fullSubject,
    text:    emailBody,
    replyTo: FROM_EMAIL,
  });

  return {
    sentAt:  now,
    subject: fullSubject,
    message: `Question sent to ${YOUR_EMAIL}. Use wait_for_reply to poll for your answer.`,
  };
}

// ── Tool: wait_for_reply ──────────────────────────────────────────────────────
async function waitForReply({ subject, sent_at, timeout_seconds = 300, poll_interval_seconds = 10 }) {
  const deadline = Date.now() + timeout_seconds * 1000;
  const interval = poll_interval_seconds * 1000;
  const sentAt   = typeof sent_at === "number" ? sent_at : Date.now() - 60_000;

  while (Date.now() < deadline) {
    const reply = await findReply(subject, sentAt);
    if (reply) {
      await deleteMessage(reply.id); // clean up after reading
      return {
        answered: true,
        answer:   reply.body.trim(),
        from:     reply.from,
        received_at: reply.receivedAt,
      };
    }
    await new Promise((r) => setTimeout(r, interval));
  }

  return {
    answered: false,
    answer:   null,
    message:  `No reply received within ${timeout_seconds}s. Ask again or proceed without input.`,
  };
}

// ── Tool: check_inbox ─────────────────────────────────────────────────────────
async function checkInbox() {
  const data = await listMessages();
  const messages = (data.messages || []).slice(0, 10).map((m) => ({
    id:      m.ID,
    subject: m.Subject,
    from:    m.From?.Address,
    to:      (m.To || []).map((t) => t.Address),
    date:    m.Created,
  }));
  return { count: data.total || 0, recent: messages };
}

// ── MCP server ────────────────────────────────────────────────────────────────
const server = new Server(
  { name: "claudemail-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "ask_question",
      description:
        "Send an email to the user asking a question. Returns metadata needed to poll for the reply. Use this when you need human input to continue.",
      inputSchema: {
        type: "object",
        properties: {
          subject: {
            type: "string",
            description: "Email subject line (short, descriptive)",
          },
          body: {
            type: "string",
            description: "The question to ask. Be specific about what you need.",
          },
          context: {
            type: "string",
            description:
              "Optional: brief context about what you're working on, shown above the question.",
          },
        },
        required: ["body"],
      },
    },
    {
      name: "wait_for_reply",
      description:
        "Poll Mailpit for a reply to a question previously sent with ask_question. Blocks (with polling) until a reply arrives or timeout is reached.",
      inputSchema: {
        type: "object",
        properties: {
          subject: {
            type: "string",
            description: "The subject of the question you sent",
          },
          sent_at: {
            type: "number",
            description: "Epoch ms timestamp from ask_question result (sent_at field)",
          },
          timeout_seconds: {
            type: "number",
            description: "How long to wait for a reply (default: 300 = 5 min)",
          },
          poll_interval_seconds: {
            type: "number",
            description: "How often to check (default: 10s)",
          },
        },
        required: ["subject", "sent_at"],
      },
    },
    {
      name: "check_inbox",
      description:
        "List recent messages in the Mailpit inbox. Useful for debugging or checking if your email setup is working.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    let result;
    if      (name === "ask_question")  result = await askQuestion(args);
    else if (name === "wait_for_reply") result = await waitForReply(args);
    else if (name === "check_inbox")   result = await checkInbox();
    else throw new Error(`Unknown tool: ${name}`);

    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  } catch (err) {
    return {
      content: [{ type: "text", text: `Error: ${err.message}` }],
      isError: true,
    };
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
const transport2 = new StdioServerTransport();
await server.connect(transport2);
