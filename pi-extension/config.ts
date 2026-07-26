/**
 * hexus config — loaded from ~/.pi/agent/extensions/hexus/config.json
 * or environment variables.
 *
 * Uses dynamic import() to avoid sync fs I/O at module load time
 * in ESM contexts. Falls back to defaults if config file is missing.
 */

export interface HexusConfig {
  /** hexus API base URL */
  apiUrl: string;
  /** Default agent_identity for memory operations */
  agentIdentity: string;
  /** Number of memories to recall per turn */
  recallLimit: number;
  /** Minimum similarity score (0-1) */
  minSimilarity: number;
  /** Whether to inject memory into system prompt */
  enabled: boolean;
  /** Whether to store turns in hexus */
  storeTurns: boolean;
}

const CONFIG_PATH = "hexus/config.json";

// Default config — used when no config file exists
const DEFAULTS: HexusConfig = {
  apiUrl: "http://localhost:8000",
  agentIdentity: "pi",
  recallLimit: 5,
  minSimilarity: 0.3,
  enabled: true,
  storeTurns: true,
};

function applyEnvOverrides(cfg: HexusConfig): HexusConfig {
  const envUrl = process.env["HEXUS_API_URL"];
  const envIdentity = process.env["HEXUS_AGENT_IDENTITY"];
  const envEnabled = process.env["HEXUS_ENABLED"];
  const envStoreTurns = process.env["HEXUS_STORE_TURNS"];

  return {
    apiUrl: envUrl ?? cfg.apiUrl,
    agentIdentity: envIdentity ?? cfg.agentIdentity,
    recallLimit: cfg.recallLimit,
    minSimilarity: cfg.minSimilarity,
    enabled: envEnabled !== undefined ? envEnabled !== "false" : cfg.enabled,
    storeTurns: envStoreTurns !== undefined ? envStoreTurns !== "false" : cfg.storeTurns,
  };
}

async function loadConfigAsync(): Promise<HexusConfig> {
  try {
    // Dynamically import to avoid sync I/O at module load
    const { readFileSync, existsSync } = await import("node:fs");
    const { join } = await import("node:path");

    // Get CONFIG_DIR_NAME from pi-coding-agent at runtime
    let configDir = ".pi";
    try {
      const piModule = await import("@earendil-works/pi-coding-agent");
      configDir = piModule.CONFIG_DIR_NAME ?? ".pi";
    } catch {
      // Module import failed, use default
    }

    const home = process.env["HOME"] ?? "/home/codenamekt";

    // Try project-local first, then global
    const projectConfig = join(process.cwd(), configDir, CONFIG_PATH);
    const globalConfig = join(home, configDir, "agent", "extensions", CONFIG_PATH);

    let configPath: string | undefined;
    if (existsSync(projectConfig)) {
      configPath = projectConfig;
    } else if (existsSync(globalConfig)) {
      configPath = globalConfig;
    }

    if (configPath) {
      const content = readFileSync(configPath, "utf-8");
      const fileConfig = JSON.parse(content);
      return applyEnvOverrides({
        apiUrl: fileConfig.apiUrl ?? DEFAULTS.apiUrl,
        agentIdentity: fileConfig.agentIdentity ?? DEFAULTS.agentIdentity,
        recallLimit: fileConfig.recallLimit ?? DEFAULTS.recallLimit,
        minSimilarity: fileConfig.minSimilarity ?? DEFAULTS.minSimilarity,
        enabled: fileConfig.enabled ?? DEFAULTS.enabled,
        storeTurns: fileConfig.storeTurns ?? DEFAULTS.storeTurns,
      });
    }
  } catch {
    // Config file not found or parse error, fall through to defaults
  }

  return applyEnvOverrides(DEFAULTS);
}

let _config: HexusConfig | undefined;
let _loadPromise: Promise<HexusConfig> | undefined;

export function getConfig(): HexusConfig {
  // After initial async load, this returns the cached config.
  // Before async load completes, it returns defaults (avoids blocking).
  return _config ?? DEFAULTS;
}

/** Kick off async config load. Call once at extension init. */
export function initConfig(): Promise<HexusConfig> {
  if (_config) return Promise.resolve(_config);
  if (!_loadPromise) {
    _loadPromise = loadConfigAsync().then((cfg) => {
      _config = cfg;
      return cfg;
    });
  }
  return _loadPromise;
}
