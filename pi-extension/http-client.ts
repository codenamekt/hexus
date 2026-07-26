/**
 * hexus HTTP client — calls the hexus REST API
 */

import { getConfig } from "./config";

export interface MemoryResult {
  id: number;
  agent_identity: string;
  target: string;
  content: string;
  score: number;
  created_at: string;
  metadata?: Record<string, unknown>;
}

export interface RecallResponse {
  query: string;
  count: number;
  results: MemoryResult[];
}

export interface HealthResponse {
  status: string;
  schema_ok: boolean;
  embedder: {
    model: string;
    dim: number;
    eager_loaded: boolean;
  };
  row_counts: {
    memory_entries: number;
  };
}

export interface RetainResponse {
  inserted: number;
  duplicates: number;
  errors: string[];
}

export interface AppendTurnResponse {
  id: number;
  session_id: string;
  role: string;
}

class HexusClient {
  private baseUrl: string;
  private healthCache: { data: HealthResponse | null; expiry: number } = { data: null, expiry: 0 };
  private offlineUntil = 0; // Unix ms; skip requests when offline

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); // Remove trailing slash
  }

  /** Returns cached health if fresh; updates cache on miss or expiry. */
  async health(forceRefresh = false): Promise<HealthResponse | null> {
    const now = Date.now();
    if (!forceRefresh && this.healthCache.data && now < this.healthCache.expiry) {
      return this.healthCache.data;
    }
    try {
      const data = await this.request<HealthResponse>("/api/health", {}, 3000);
      this.healthCache = { data, expiry: now + 30_000 };
      this.offlineUntil = 0;
      return data;
    } catch {
      // Mark offline for 5s to avoid hammering a dead server
      this.offlineUntil = now + 5_000;
      return this.healthCache.data; // return stale cache if available
    }
  }

  private async request<T>(
    path: string,
    options: RequestInit = {},
    timeoutMs = 5000
  ): Promise<T> {
    const now = Date.now();
    if (this.offlineUntil && now < this.offlineUntil) {
      throw new Error("hexus offline");
    }

    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });
      clearTimeout(timeout);

      if (!response.ok) {
        const text = await response.text().catch(() => "Unknown error");
        throw new Error(
          `hexus API error: ${response.status} ${response.statusText} - ${text}`
        );
      }

      return response.json() as Promise<T>;
    } catch (err) {
      clearTimeout(timeout);
      if (err instanceof Error && err.name === "AbortError") {
        throw new Error(`hexus request timed out after ${timeoutMs}ms`);
      }
      throw err;
    }
  }

  // Deprecated: use health() directly — kept for callers that ignore null
  async healthSync(): Promise<HealthResponse> {
    return (await this.health(true)) ?? { status: "offline", schema_ok: false, embedder: { model: "", dim: 0, eager_loaded: false }, row_counts: { memory_entries: 0 } };
  }

  async recall(params: {
    query: string;
    top_k?: number;
    agent_identity?: string;
    target?: string;
    min_similarity?: number;
  }): Promise<RecallResponse> {
    return this.request<RecallResponse>("/api/recall", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  async retain(params: {
    /** List of memory content strings (not objects). */
    contents: string[];
    /** Target applied to all items. */
    target?: string;
    /** Per-item or shared metadata. */
    metadata?: Record<string, unknown> | Record<string, unknown>[];
    agent_identity?: string;
  }): Promise<RetainResponse> {
    return this.request<RetainResponse>("/api/retain", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  async appendTurn(params: {
    session_id: string;
    role: string;
    content: string;
    agent_identity?: string;
    metadata?: Record<string, unknown>;
  }): Promise<AppendTurnResponse> {
    return this.request<AppendTurnResponse>("/api/append-turn", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }
}

let _client: HexusClient | undefined;

export function getClient(): HexusClient {
  if (!_client) {
    const config = getConfig();
    _client = new HexusClient(config.apiUrl);
  }
  return _client;
}

export function resetClient(): void {
  _client = undefined;
}
