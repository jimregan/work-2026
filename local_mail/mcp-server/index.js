#!/usr/bin/env node
/**
 * claudemail-mcp
 * MCP server that lets Claude Code ask you questions via email.
 * You reply in your mail client and Claude Code polls for the answer.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import nodemailer from "nodemailer";
import {
  buildQuestionEmail,
  extractModelNotes,
  extractReplyText,
  generateConversationId,
  getHeaderValues,
  getMessageId,
  getThreadReferences,
  isReplyMatch,
} from "./lib/claudemail.js";

const SMTP_HOST = process.env.CLAUDEMAIL_SMTP_HOST || "localhost";
const SMTP_PORT = parseInt(process.env.CLAUDEMAIL_SMTP_PORT || "1025", 10);
const MAILPIT_URL = process.env.CLAUDEMAIL_MAILPIT_URL || "http://localhost:8025";
const YOUR_EMAIL = process.env.CLAUDEMAIL_TO || "you@localhost";
const FROM_EMAIL = process.env.CLAUDEMAIL_FROM || "claude@localhost";

const transport = nodemailer.createTransport({
  host: SMTP_HOST,
  port: SMTP_PORT,
  secure: false,
  auth: { user: "claude", pass: "claude" },
  tls: { rejectUnauthorized: false },
});

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
  const res = await fetch(`${MAILPIT_URL}/api/v1/message/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Mailpit API error: ${res.status}`);
}

async function findReply({ subject, sentAt, conversationId, originalMessageId }) {
  const data = await listMessages();
  const messages = data.messages || [];

  for (const message of messages) {
    let fullMessage;
    try {
      fullMessage = await getMessage(message.ID);
    } catch {
      continue;
    }
    if (
      isReplyMatch({
        message,
        fullMessage,
        expectedSubject: subject,
        sentAt,
        replyAddress: FROM_EMAIL,
        conversationId,
        originalMessageId,
      })
    ) {
      const replyMessageId = getMessageId(fullMessage);
      const existingRefs = getThreadReferences(fullMessage);
      const nextReferences = [...new Set([...existingRefs, replyMessageId])].filter(Boolean).join(" ");
      const rawText = fullMessage.Text || fullMessage.HTML || "";
      const { notes: modelNotes, cleaned } = extractModelNotes(rawText);
      return {
        id: message.ID,
        subject: message.Subject,
        from: message.From?.Address,
        body: extractReplyText(cleaned),
        receivedAt: message.Created,
        messageId: replyMessageId,
        nextReferences,
        modelNotes,
      };
    }
  }

  return null;
}

async function askQuestion({ subject, body, context, in_reply_to, references }) {
  const sentAt = Date.now();
  const fullSubject = subject || "Claude needs your input";
  const conversationId = generateConversationId();
  const messageId = `<${conversationId}@claudemail.local>`;
  const emailBody = buildQuestionEmail({ body, context, conversationId, sentAt });

  const mail = {
    from: FROM_EMAIL,
    to: YOUR_EMAIL,
    subject: fullSubject,
    text: emailBody,
    replyTo: FROM_EMAIL,
    messageId,
    headers: {
      "X-ClaudeMail-Conversation": conversationId,
      "X-Auto-Response-Suppress": "All",
    },
  };
  if (in_reply_to) mail.inReplyTo = in_reply_to;
  if (references) mail.references = references;

  const info = await transport.sendMail(mail);

  return {
    sentAt,
    sent_at: sentAt,
    subject: fullSubject,
    conversation_id: conversationId,
    message_id: info.messageId || messageId,
    message: `Question sent to ${YOUR_EMAIL}. Use wait_for_reply to poll for your answer.`,
  };
}

async function waitForReply({
  subject,
  sent_at,
  sentAt,
  conversation_id,
  conversationId,
  message_id,
  messageId,
  timeout_seconds = 300,
  poll_interval_seconds = 10,
}) {
  const timeoutSeconds = Number(timeout_seconds);
  const pollIntervalSeconds = Number(poll_interval_seconds);

  if (!subject) throw new Error("wait_for_reply requires a subject");
  if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) {
    throw new Error("timeout_seconds must be a positive number");
  }
  if (!Number.isFinite(pollIntervalSeconds) || pollIntervalSeconds <= 0) {
    throw new Error("poll_interval_seconds must be a positive number");
  }

  const sentTimestamp = Number.isFinite(Number(sent_at))
    ? Number(sent_at)
    : Number.isFinite(Number(sentAt))
      ? Number(sentAt)
      : Date.now() - 60_000;
  const expectedConversationId = conversation_id || conversationId || null;
  const expectedMessageId = message_id || messageId || null;
  const deadline = Date.now() + timeoutSeconds * 1000;
  const interval = pollIntervalSeconds * 1000;

  while (Date.now() < deadline) {
    let reply = null;
    try {
      reply = await findReply({
        subject,
        sentAt: sentTimestamp,
        conversationId: expectedConversationId,
        originalMessageId: expectedMessageId,
      });
    } catch {
      await new Promise((resolve) => setTimeout(resolve, interval));
      continue;
    }

    if (reply) {
      await deleteMessage(reply.id);
      return {
        answered: true,
        answer: reply.body.trim(),
        from: reply.from,
        received_at: reply.receivedAt,
        subject: reply.subject,
        reply_message_id: reply.messageId,
        next_references: reply.nextReferences,
      };
    }

    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  return {
    answered: false,
    answer: null,
    message: `No reply received within ${timeoutSeconds}s. Ask again or proceed without input.`,
  };
}

async function checkInbox() {
  const data = await listMessages();
  const messages = await Promise.all(
    (data.messages || []).slice(0, 10).map(async (message) => {
      const fullMessage = await getMessage(message.ID);
      const preview = extractReplyText(fullMessage.Text || fullMessage.HTML || "").slice(0, 160);

      return {
        id: message.ID,
        subject: message.Subject,
        from: message.From?.Address,
        to: (message.To || []).map((recipient) => recipient.Address),
        date: message.Created,
        message_id: getMessageId(fullMessage),
        preview,
      };
    })
  );

  return { count: data.total || 0, recent: messages };
}

const server = new Server(
  { name: "claudemail-mcp", version: "1.2.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "ask_question",
      description:
        "Send an email to the user asking a question. Returns the subject, send timestamp, and thread metadata needed to poll for the reply.",
      inputSchema: {
        type: "object",
        properties: {
          subject: {
            type: "string",
            description: "Email subject line.",
          },
          body: {
            type: "string",
            description: "The question to ask. Be specific about what you need.",
          },
          context: {
            type: "string",
            description: "Optional context shown above the question.",
          },
          in_reply_to: {
            type: "string",
            description: "Message-ID to thread against. Use reply_message_id from wait_for_reply.",
          },
          references: {
            type: "string",
            description: "Space-separated reference chain for threading. Use next_references from wait_for_reply.",
          },
        },
        required: ["body"],
      },
    },
    {
      name: "wait_for_reply",
      description:
        "Poll Mailpit for a reply to a previous question. Prefer passing subject, sent_at, conversation_id, and message_id from ask_question for reliable matching.",
      inputSchema: {
        type: "object",
        properties: {
          subject: {
            type: "string",
            description: "The subject returned by ask_question.",
          },
          sent_at: {
            type: "number",
            description: "Epoch ms timestamp from ask_question.",
          },
          conversation_id: {
            type: "string",
            description: "Conversation token returned by ask_question.",
          },
          message_id: {
            type: "string",
            description: "Message-ID returned by ask_question.",
          },
          timeout_seconds: {
            type: "number",
            description: "How long to wait for a reply. Default: 300.",
          },
          poll_interval_seconds: {
            type: "number",
            description: "How often to poll for new messages. Default: 10.",
          },
        },
        required: ["subject", "sent_at"],
      },
    },
    {
      name: "check_inbox",
      description: "List recent Mailpit messages with a short preview for debugging.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  try {
    let result;
    if (name === "ask_question") result = await askQuestion(args);
    else if (name === "wait_for_reply") result = await waitForReply(args);
    else if (name === "check_inbox") result = await checkInbox();
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

const stdioTransport = new StdioServerTransport();
await server.connect(stdioTransport);
