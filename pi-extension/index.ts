/**
 * hexus — Vector memory extension for pi
 *
 * Integrates hexus (Postgres + pgvector memory) into pi agent.
 * Features:
 * 1. Embeds user's prompt → calls hexus /api/recall → injects relevant memories
 * 2. Stores conversation turns in hexus for future recall
 * 3. Session reflection when:
 *    - Session exceeds token threshold (default 8000)
 *    - 10+ turns since last reflection
 *    - User idle for 10+ seconds
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { complete, getModel } from "@earendil-works/pi-ai/compat";
import { getConfig, initConfig } from "./config";
import { getClient, type MemoryResult } from "./http-client";

// ---------------------------------------------------------------------------
// Reflection Config
// ---------------------------------------------------------------------------

interface ReflectionConfig {
  enabled: boolean;
  tokenThreshold: number;
  minTurnsBetweenReflections: number;
  idleSeconds: number;
  modelProvider: string;
  modelId: string;
}

function getReflectionConfig(): ReflectionConfig {
  const enabled = process.env["HEXUS_REFLECTION_ENABLED"] !== "false";
  const tokenThreshold = parseInt(process.env["HEXUS_REFLECTION_TOKEN_THRESHOLD"] ?? "", 10);
  const minTurns = parseInt(process.env["HEXUS_REFLECTION_MIN_TURNS"] ?? "", 10);
  const idleSeconds = parseInt(process.env["HEXUS_REFLECTION_IDLE_SECONDS"] ?? "", 10);

  if (isNaN(tokenThreshold)) console.warn("hexus: HEXUS_REFLECTION_TOKEN_THRESHOLD not numeric, using 8000");
  if (isNaN(minTurns)) console.warn("hexus: HEXUS_REFLECTION_MIN_TURNS not numeric, using 10");
  if (isNaN(idleSeconds)) console.warn("hexus: HEXUS_REFLECTION_IDLE_SECONDS not numeric, using 10");

  return {
    enabled,
    tokenThreshold: isNaN(tokenThreshold) ? 8000 : tokenThreshold,
    minTurnsBetweenReflections: isNaN(minTurns) ? 10 : minTurns,
    idleSeconds: isNaN(idleSeconds) ? 10 : idleSeconds,
    modelProvider: process.env["HEXUS_REFLECTION_MODEL"] ?? "headroom",
    modelId: process.env["HEXUS_REFLECTION_MODEL_ID"] ?? "tobiTradez/minimax-m2.7-highspeed",
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Simple hash of a string — used for session IDs to avoid leaking filepath. */
function hashString(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
  }
  // Return hex string (no negative values)
  return Math.abs(hash >>> 0).toString(16).padStart(8, "0");
}

/** Truncate text at word boundary, not mid-word. */
function truncateAtWord(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  const truncated = text.slice(0, maxLen);
  const lastSpace = truncated.lastIndexOf(" ");
  return (lastSpace > maxLen * 0.7 ? truncated.slice(0, lastSpace) : truncated) + "…";
}

function formatMemoryContext(results: MemoryResult[], agentIdentity: string): string {
  if (!results.length) return "";

  const lines = [`## Relevant Memory (hexus, ${agentIdentity})`, ""];
  for (const r of results) {
    const score = ((r.score || 0) * 100).toFixed(0);
    const content = truncateAtWord((r.content || "").replace(/\n+/g, " ").trim(), 400);
    lines.push(`- **[${score}%]** (${r.target || "memory"}) ${content}`);
  }
  lines.push("", "*Use `hexus_recall` to search, or `hexus_retain` to save.*");
  return lines.join("\n");
}

function extractText(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];
  return content
    .filter((b): b is { type: "text"; text: string } => (b as any)?.type === "text")
    .map((b) => b.text)
    .filter((t): t is string => typeof t === "string");
}

function buildConversationText(entries: any[]): string {
  const sections: string[] = [];
  for (const e of entries) {
    if (e.type !== "message" || !["user", "assistant"].includes(e.message?.role)) continue;
    const texts = extractText(e.message.content);
    if (texts.length) {
      sections.push(`${e.message.role === "user" ? "User" : "Assistant"}: ${texts.join("\n")}`);
    }
  }
  return sections.join("\n\n");
}

const REFLECTION_PROMPT = `Extract 1-5 durable facts from this conversation that should be remembered.

Rules:
- New info about the user (preferences, projects, people, events)
- Key decisions or agreements made
- Important context for future sessions
- Things the user explicitly asked to remember

Format as JSON array:
[{"content": "fact here", "target": "memory|user"}, ...]

Empty array if nothing worth remembering. Only output JSON.`;

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function hexus(pi: ExtensionAPI) {
  // Bootstrap config asynchronously — initConfig() runs once and caches.
  // getConfig() returns defaults until the promise resolves, then returns real config.
  initConfig().catch((err) => console.warn("hexus: config load failed:", err));
  const config = getConfig();
  const reflConfig = getReflectionConfig();

  // State
  let sessionId: string | undefined;
  let turnsSinceReflection = 0;
  let lastReflectionTurn = 0;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let isReflecting = false;
  let recallInFlight = false; // Dedupe concurrent recall calls

  // -------------------------------------------------------------------------
  // Reflection
  // -------------------------------------------------------------------------

  async function runReflection(ctx: ExtensionAPI): Promise<void> {
    if (isReflecting) return;
    isReflecting = true;

    const client = getClient();
    const branch = ctx.sessionManager.getBranch();
    const convText = buildConversationText(branch);

    if (!convText.trim()) { isReflecting = false; return; }

    ctx.ui.setStatus("hexus", "hexus: reflecting...");
    ctx.ui.notify("Running session reflection...", "info");

    try {
      const model = getModel(reflConfig.modelProvider, reflConfig.modelId);
      if (!model) {
        console.warn(`hexus: model ${reflConfig.modelProvider}/${reflConfig.modelId} not found`);
        isReflecting = false;
        return;
      }

      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
      if (!auth.ok || !auth.apiKey) { isReflecting = false; return; }

      const resp = await complete(model,
        { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: `Extract facts:\n\n${convText.slice(-4000)}` }], timestamp: Date.now() }], systemPrompt: REFLECTION_PROMPT },
        { apiKey: auth.apiKey, headers: auth.headers, env: auth.env }
      );

      const respText = resp.content.filter((c): c is { type: "text"; text: string } => c.type === "text").map(c => c.text).join("\n");

      let facts: Array<{ content: string; target: string }> = [];
      try {
        const m = respText.match(/\[[\s\S]*\]/);
        if (m) facts = JSON.parse(m[0]);
      } catch {}

      if (facts.length > 0) {
        const result = await client.retain({
          contents: facts.map(f => f.content),
          target: "memory",
          metadata: facts.map(f => ({ source: "reflection", target: f.target, reflection_turns: turnsSinceReflection })),
          agent_identity: config.agentIdentity,
        });
        ctx.ui.notify(`Reflection: saved ${result.inserted} fact${result.inserted !== 1 ? "s" : ""}`, "info");
      }

      lastReflectionTurn = turnsSinceReflection;
      turnsSinceReflection = 0;
    } catch (err) {
      console.error("hexus reflection error:", err);
    } finally {
      isReflecting = false;
      const health = await client.health(true).catch(() => null);
      ctx.ui.setStatus("hexus", `hexus: ${health?.row_counts.memory_entries ?? "?"} memories`);
      if (reflConfig.enabled) scheduleReflection(ctx);
    }
  }

  function scheduleReflection(ctx: ExtensionAPI): void {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(async () => {
      if (!reflConfig.enabled || isReflecting) return;
      const usage = ctx.getContextUsage();
      const tokens = usage?.tokens ?? 0;
      const sinceLast = turnsSinceReflection - lastReflectionTurn;
      if (tokens >= reflConfig.tokenThreshold && sinceLast >= reflConfig.minTurnsBetweenReflections) {
        await runReflection(ctx);
      } else {
        scheduleReflection(ctx);
      }
    }, reflConfig.idleSeconds * 1000);
  }

  // -------------------------------------------------------------------------
  // Events
  // -------------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    turnsSinceReflection = 0;
    lastReflectionTurn = 0;
    const client = getClient();

    const health = await client.health(true).catch(() => null);
    if (health?.status === "ok") {
      ctx.ui.notify(`hexus connected (${health.row_counts.memory_entries} memories)`, "info");
      ctx.ui.setStatus("hexus", `hexus: ${health.row_counts.memory_entries} memories`);
      if (reflConfig.enabled) scheduleReflection(ctx);
    } else {
      ctx.ui.notify("hexus: connection failed", "warning");
      ctx.ui.setStatus("hexus", "hexus: offline");
    }
  });

  // FIX: Turn counting moved here from input event.
  // input fires per keystroke, turn_end fires once per completed turn.
  // We also store the assistant message in the same handler.
  pi.on("turn_end", async (event, ctx) => {
    turnsSinceReflection++;

    // Reset idle timer after a completed turn
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (reflConfig.enabled) scheduleReflection(ctx);

    // Store assistant message in hexus
    if (config.storeTurns && sessionId) {
      const client = getClient();
      const text = (event.message?.content || [])
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map(c => c.text).join("\n");

      if (text.length >= 20) {
        client.appendTurn({
          session_id: sessionId,
          role: "assistant",
          content: text.slice(0, 4000),
          agent_identity: config.agentIdentity,
        }).catch(() => {});
      }
    }
  });

  pi.on("input", async (event, ctx) => {
    // Only reset idle timer on new user input — don't schedule reflection here
    // (turn_end handles that to avoid double-scheduling)
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (reflConfig.enabled && event.text?.trim()) scheduleReflection(ctx);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    const client = getClient();

    if (!sessionId) {
      const sf = ctx.sessionManager.getSessionFile();
      // FIX: Hash the filepath rather than base64-encoding it (avoids leaking paths)
      sessionId = sf ? `pi:${hashString(sf)}` : `pi:${hashString(String(Date.now()))}`;
    }

    const prompt = event.prompt?.trim();
    if (!prompt) return;

    // FIX: Dedupe concurrent recall calls within the same turn.
    if (recallInFlight) return;
    recallInFlight = true;

    try {
      const health = await client.health().catch(() => null);
      if (!health?.ok) { ctx.ui.setStatus("hexus", "hexus: offline"); return; }
      ctx.ui.setStatus("hexus", `hexus: ${health.row_counts.memory_entries} memories`);

      const recall = await client.recall({
        query: prompt,
        top_k: config.recallLimit,
        agent_identity: config.agentIdentity,
        min_similarity: config.minSimilarity,
      });

      if (recall.results.length > 0) {
        return { message: { customType: "hexus-memory", content: formatMemoryContext(recall.results, config.agentIdentity), display: false } };
      }
    } catch (err) {
      ctx.ui.setStatus("hexus", `hexus: error (${err instanceof Error ? err.message : "?"})`);
    } finally {
      recallInFlight = false;
    }
  });
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "hexus_recall",
    label: "Recall Memory",
    description: "Search hexus vector memory for relevant past entries.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
      scope: Type.Optional(Type.String({ description: "'current' or 'all'" })),
    }),
    async execute(_toolCallId, params, _signal, onUpdate) {
      const client = getClient();
      onUpdate?.({ content: [{ type: "text", text: "Searching memory..." }] });

      const [recall, health] = await Promise.all([
        client.recall({
          query: params.query,
          top_k: Math.min(params.limit ?? 5, 20),
          agent_identity: params.scope === "all" ? undefined : config.agentIdentity,
        }),
        client.health().catch(() => null),
      ]);

      if (!recall.results.length) {
        return {
          content: [{ type: "text", text: `No memories found for: \"${params.query}\"` }],
          details: { count: 0, memoryCount: health?.row_counts.memory_entries ?? null },
        };
      }

      const lines = [`Found ${recall.count} memories:\n`];
      for (const r of recall.results) {
        lines.push(`- [${((r.score || 0) * 100).toFixed(0)}%] ${truncateAtWord(r.content, 200)}`);
      }
      if (health?.status === "ok") {
        lines.push(`\n_hexus: ${health.row_counts.memory_entries} total memories_`);
      }
      return { content: [{ type: "text", text: lines.join("\n") }], details: { count: recall.count, memoryCount: health?.row_counts.memory_entries ?? null } };
    },
  });

  pi.registerTool({
    name: "hexus_retain",
    label: "Retain Memory",
    description: "Save important info to hexus memory.",
    parameters: Type.Object({
      content: Type.String({ description: "What to remember" }),
      target: Type.Optional(Type.String({ description: "'memory' or 'user'" })),
    }),
    async execute(_toolCallId, params, _signal, onUpdate) {
      const client = getClient();
      onUpdate?.({ content: [{ type: "text", text: "Saving..." }] });

      const [result, health] = await Promise.all([
        client.retain({
          contents: [params.content],
          target: params.target ?? "memory",
          agent_identity: config.agentIdentity,
        }),
        client.health(true).catch(() => null), // Refresh health after write
      ]);

      const status = result.inserted > 0
        ? `Saved (${result.inserted} new, ${result.duplicates} dupes)`
        : "Already exists";
      const healthLine = health?.status === "ok" ? ` | hexus: ${health.row_counts.memory_entries} memories` : "";

      return {
        content: [{ type: "text", text: `${status}${healthLine}` }],
        details: result,
      };
    },
  });

  pi.on("session_shutdown", () => {
    if (idleTimer) clearTimeout(idleTimer);
  });
}
