"""mcp_server.import_cli — Bulk import CLI for Hexus memory.
Allows importing memory entries from existing installations (like Holographic, Honcho, Mem0, or raw MD/JSON files).
"""

import argparse
import json
import os
import sys
from typing import List, Optional
from hexus.store import MemoryStore
from hexus.embed import embed


def import_mem0(
    store: MemoryStore, file_path: str, agent_identity: str, target: str
) -> None:
    """Import memories from a Mem0 JSON file.
    Expected JSON format: a list of objects, each containing:
    - 'memory': the memory text content
    - 'metadata': optional metadata dictionary
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"ERROR: Failed to parse JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        # Allow single object import
        data = [data]

    print(f"Found {len(data)} items to import from Mem0 format...")
    success = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(data):
        content = item.get("memory") or item.get("content") or item.get("text")
        if not content:
            print(f"Skipping item {idx}: no 'memory' or 'content' field found.")
            skipped += 1
            continue

        meta = item.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {"raw_metadata": meta}
        meta["imported_from"] = "mem0"

        try:
            vec = embed(content)
        except Exception as exc:
            print(
                f"Warning: Failed to embed '{content[:40]}...': {exc}. Inserting without embedding."
            )
            vec = None

        try:
            row_id = store.add(
                agent_identity=agent_identity,
                target=target,
                content=content,
                embedding=vec,
                metadata=meta,
            )
            if row_id is not None:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"ERROR: Failed to insert item {idx}: {exc}", file=sys.stderr)
            errors += 1

    print(
        f"Import complete: {success} inserted, {skipped} skipped/duplicates, {errors} errors."
    )


def import_honcho(
    store: MemoryStore, file_path: str, agent_identity: str, target: str
) -> None:
    """Import memories from a Honcho export.
    Expected JSON format: list of messages or documents.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"ERROR: Failed to parse JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        data = [data]

    print(f"Found {len(data)} items to import from Honcho format...")
    success = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(data):
        content = item.get("content") or item.get("text") or item.get("value")
        if not content:
            print(f"Skipping item {idx}: no content field found.")
            skipped += 1
            continue

        meta = {
            "imported_from": "honcho",
            "honcho_id": item.get("id"),
        }
        if "metadata" in item and isinstance(item["metadata"], dict):
            meta.update(item["metadata"])

        try:
            vec = embed(content)
        except Exception as exc:
            print(
                f"Warning: Failed to embed '{content[:40]}...': {exc}. Inserting without embedding."
            )
            vec = None

        try:
            row_id = store.add(
                agent_identity=agent_identity,
                target=target,
                content=content,
                embedding=vec,
                metadata=meta,
            )
            if row_id is not None:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"ERROR: Failed to insert item {idx}: {exc}", file=sys.stderr)
            errors += 1

    print(
        f"Import complete: {success} inserted, {skipped} skipped/duplicates, {errors} errors."
    )


def import_holographic(
    store: MemoryStore, file_path: str, agent_identity: str, target: str
) -> None:
    """Import memories from a Holographic JSON export."""
    # Similar structure to Mem0/Honcho
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"ERROR: Failed to parse JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        data = [data]

    print(f"Found {len(data)} items to import from Holographic format...")
    success = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(data):
        content = item.get("fact") or item.get("statement") or item.get("content")
        if not content:
            print(f"Skipping item {idx}: no content/fact field found.")
            skipped += 1
            continue

        meta = {
            "imported_from": "holographic",
            "confidence": item.get("confidence", 1.0),
        }
        if "metadata" in item and isinstance(item["metadata"], dict):
            meta.update(item["metadata"])

        try:
            vec = embed(content)
        except Exception as exc:
            print(
                f"Warning: Failed to embed '{content[:40]}...': {exc}. Inserting without embedding."
            )
            vec = None

        try:
            row_id = store.add(
                agent_identity=agent_identity,
                target=target,
                content=content,
                embedding=vec,
                metadata=meta,
            )
            if row_id is not None:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"ERROR: Failed to insert item {idx}: {exc}", file=sys.stderr)
            errors += 1

    print(
        f"Import complete: {success} inserted, {skipped} skipped/duplicates, {errors} errors."
    )


def import_markdown(
    store: MemoryStore, file_path: str, agent_identity: str, target: str
) -> None:
    """Import memories from a raw markdown file."""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Bulk importing markdown file {file_path} to {agent_identity} ({target})...")
    try:
        res = store.bulk_upsert_md(
            agent_identity=agent_identity,
            target=target,
            file_path=file_path,
            embed_fn=embed,
        )
        print(
            f"Import complete: {res['inserted']} inserted, {res['skipped']} skipped/duplicates (parsed {res['parsed']} total)."
        )
    except Exception as exc:
        print(f"ERROR: Bulk import failed: {exc}", file=sys.stderr)
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hexus-import",
        description="Bulk import tool for Hexus memory store.",
    )
    # Define common arguments on the parent parser so they come before the subcommand
    parser.add_argument(
        "--dsn",
        default=os.environ.get("HEXUS_DSN", ""),
        help="Postgres DSN. Defaults to HEXUS_DSN env var.",
    )
    parser.add_argument(
        "--agent-identity",
        default="default",
        help="Target agent_identity (theme scope) for the imported memories. Defaults to 'default'.",
    )
    parser.add_argument(
        "--target",
        default="memory",
        choices=("memory", "user"),
        help="Target store ('memory' or 'user'). Defaults to 'memory'.",
    )

    sub = parser.add_subparsers(dest="format", required=True)

    # mem0 subparser
    p_mem0 = sub.add_parser("mem0", help="Import from Mem0 JSON export.")
    p_mem0.add_argument("file", help="Path to Mem0 export JSON file.")
    p_mem0.set_defaults(
        func=lambda store, args: import_mem0(
            store, args.file, args.agent_identity, args.target
        )
    )

    # honcho subparser
    p_honcho = sub.add_parser("honcho", help="Import from Honcho export JSON.")
    p_honcho.add_argument("file", help="Path to Honcho export JSON file.")
    p_honcho.set_defaults(
        func=lambda store, args: import_honcho(
            store, args.file, args.agent_identity, args.target
        )
    )

    # holographic subparser
    p_holo = sub.add_parser("holographic", help="Import from Holographic JSON.")
    p_holo.add_argument("file", help="Path to Holographic export JSON file.")
    p_holo.set_defaults(
        func=lambda store, args: import_holographic(
            store, args.file, args.agent_identity, args.target
        )
    )

    # markdown subparser
    p_md = sub.add_parser("markdown", help="Import from raw MEMORY.md / USER.md files.")
    p_md.add_argument("file", help="Path to markdown file.")
    p_md.set_defaults(
        func=lambda store, args: import_markdown(
            store, args.file, args.agent_identity, args.target
        )
    )

    args = parser.parse_args(argv)

    if not args.dsn:
        print("ERROR: --dsn is required (or set HEXUS_DSN env var).", file=sys.stderr)
        return 2

    store = MemoryStore(args.dsn)
    try:
        args.func(store, args)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
