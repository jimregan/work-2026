import crypto from "node:crypto";

function normalizeHeaderValues(value) {
  if (Array.isArray(value)) return value.map((item) => `${item}`.trim()).filter(Boolean);
  if (typeof value === "string") return [value.trim()].filter(Boolean);
  return [];
}

export function getHeaderValues(message, name) {
  const headers = message?.Headers || message?.headers;
  if (!headers || typeof headers !== "object") return [];

  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === name.toLowerCase()) {
      return normalizeHeaderValues(value);
    }
  }

  return [];
}

export function getMessageId(message) {
  return (
    message?.MessageID ||
    message?.MessageId ||
    getHeaderValues(message, "Message-ID")[0] ||
    null
  );
}

export function getThreadReferences(message) {
  return ["References", "In-Reply-To"]
    .flatMap((headerName) => getHeaderValues(message, headerName))
    .flatMap((value) => value.split(/\s+/))
    .map((value) => value.trim())
    .filter(Boolean);
}

export function normalizeAddress(address) {
  return `${address || ""}`.trim().toLowerCase();
}

export function extractAddresses(entries) {
  return (entries || [])
    .map((entry) => normalizeAddress(entry?.Address || entry?.address || entry))
    .filter(Boolean);
}

export function generateConversationId() {
  return crypto.randomUUID();
}

export function buildQuestionEmail({ body, context, conversationId, sentAt }) {
  return [
    context ? `Context\n-------\n${context}\n` : null,
    `Question\n--------\n${body}`,
    "\nReply to this email to answer. Claude Code is waiting.",
    `\n[claudemail:${conversationId}]`,
    `Sent at: ${new Date(sentAt).toLocaleString()}`,
  ]
    .filter(Boolean)
    .join("\n");
}

export function normalizeSubject(subject) {
  return `${subject || ""}`
    .trim()
    .toLowerCase()
    .replace(/^(re|aw|sv|fwd|fw):\s*/gi, "")
    .replace(/\s+/g, " ");
}

export function subjectMatches(expectedSubject, actualSubject) {
  const expected = normalizeSubject(expectedSubject);
  const actual = normalizeSubject(actualSubject);

  if (!expected || !actual) return false;
  return actual === expected || actual.includes(expected);
}

function trimQuotedReplyLines(text) {
  const lines = `${text || ""}`.replace(/\r\n/g, "\n").split("\n");
  const stopPatterns = [
    /^>/,
    /^On .+wrote:$/i,
    /^From:\s/i,
    /^Sent:\s/i,
    /^To:\s/i,
    /^Subject:\s/i,
    /^-{2,}\s*Original Message\s*-{2,}$/i,
    /^\[claudemail:[a-f0-9-]+\]$/i,
  ];

  const kept = [];
  for (const line of lines) {
    if (stopPatterns.some((pattern) => pattern.test(line.trim()))) break;
    kept.push(line);
  }

  return kept.join("\n").trim();
}

export function extractReplyText(messageBody) {
  const trimmed = `${messageBody || ""}`.trim();
  if (!trimmed) return "";

  const withoutTokenLine = trimmed
    .split(/\r?\n/)
    .filter((line) => !/^\[claudemail:[a-f0-9-]+\]$/i.test(line.trim()))
    .join("\n")
    .trim();

  return trimQuotedReplyLines(withoutTokenLine) || withoutTokenLine;
}

export function isReplyMatch({
  message,
  fullMessage,
  expectedSubject,
  sentAt,
  replyAddress,
  conversationId,
  originalMessageId,
}) {
  const createdAt = new Date(message?.Created || fullMessage?.Created || 0).getTime();
  if (!Number.isFinite(createdAt) || createdAt <= sentAt) return false;

  const recipients = extractAddresses(fullMessage?.To || message?.To);
  if (replyAddress && !recipients.includes(normalizeAddress(replyAddress))) {
    return false;
  }

  const references = new Set(getThreadReferences(fullMessage));
  const messageText = `${fullMessage?.Text || fullMessage?.HTML || ""}`;
  const hasConversationToken = conversationId
    ? messageText.includes(`[claudemail:${conversationId}]`)
    : false;
  const referencesOriginal = originalMessageId ? references.has(originalMessageId) : false;

  return (
    referencesOriginal ||
    hasConversationToken ||
    subjectMatches(expectedSubject, message?.Subject || fullMessage?.Subject)
  );
}
