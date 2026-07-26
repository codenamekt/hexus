# hexus-pi — Vector Memory Extension for pi Agent

Integrates hexus (Postgres + pgvector memory) into [pi](https://github.com/earendil-works/pi) agent harness.

## Features

1. **Memory Recall** — Embeds user prompts and searches hexus for relevant past memories, injecting them into the system prompt
2. **Turn Capture** — Stores conversation turns in hexus for future recall
3. **Session Reflection** — Automatically extracts durable facts from long/idle sessions and saves them to memory

## Installation

### Option 1: Symlink (Development)

```bash
ln -s /path/to/hexus/pi-extension ~/.pi/agent/extensions/hexus
```

### Option 2: Copy

```bash
cp -r /path/to/hexus/pi-extension ~/.pi/agent/extensions/hexus
```

### Option 3: Publish as pi Package

(WIP) Publish to npm and install via `pi install hexus`

## Configuration

Edit `~/.pi/agent/extensions/hexus/config.json`:

```json
{
  "apiUrl": "http://localhost:8000",
  "agentIdentity": "pi",
  "recallLimit": 5,
  "minSimilarity": 0.3,
  "enabled": true,
  "storeTurns": true
}
```

Or via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXUS_API_URL` | `http://localhost:8000` | hexus API base URL |
| `HEXUS_AGENT_IDENTITY` | `pi` | Memory namespace |
| `HEXUS_ENABLED` | `true` | Enable/disable extension |
| `HEXUS_STORE_TURNS` | `true` | Store conversation turns |
| `HEXUS_REFLECTION_ENABLED` | `true` | Enable session reflection |
| `HEXUS_REFLECTION_TOKEN_THRESHOLD` | `8000` | Min tokens before reflection |
| `HEXUS_REFLECTION_MIN_TURNS` | `10` | Turns between reflections |
| `HEXUS_REFLECTION_IDLE_SECONDS` | `10` | Idle time before reflection |
| `HEXUS_REFLECTION_MODEL` | `tobiTradez` | Model provider |
| `HEXUS_REFLECTION_MODEL_ID` | `minimax-m2.7-highspeed` | Model ID |

## hexus API Endpoints Required

This extension requires the REST API endpoints added to hexus:

- `GET /api/health` — Health check
- `POST /api/recall` — Vector search
- `POST /api/retain` — Store memory entries
- `POST /api/append-turn` — Store conversation turns

These are available in hexus v0.9.2+.

## Tools

The extension registers two tools for explicit memory operations:

- `hexus_recall` — Search memory for past entries
- `hexus_retain` — Save important info to memory

## Reflection

Session reflection automatically extracts durable facts from conversations when:

1. Session exceeds token threshold (default: 8000 tokens)
2. 10+ turns since last reflection
3. User idle for 10+ seconds

The reflection model analyzes the conversation and saves facts like:
- User preferences and projects
- Key decisions made
- Important context for future sessions

## Multi-Harness Strategy

This extension is designed to work with pi, but the same hexus backend can be used by other harnesses:

- **Hermes** — Native hexus plugin (`pip install hexus`)
- **pi** — This extension
- **Claude Desktop / Cursor** — MCP server (`hexus-mcp serve --transport stdio`)
- **Others** — HTTP REST API

See main [hexus README](../README.md) for full architecture.

## License

BSD-3-Clause
