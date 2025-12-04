from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import toml

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Pythons
    import tomli as tomllib  # type: ignore


@dataclass
class InstallTarget:
    name: str
    path: Path
    kind: str  # "json" or "toml"
    create_if_parent_exists: bool = False


def _resolve_command(explicit: str | None) -> str:
    """Find the command path we should write into client configs."""
    if explicit:
        return explicit
    found = shutil.which("gdb-mcp")
    if found:
        return found
    if sys.argv and sys.argv[0]:
        return os.path.abspath(sys.argv[0])
    raise RuntimeError("Unable to determine gdb-mcp command path.")


def _update_json_mcp_config(path: Path, command: str, args: List[str], create: bool) -> bool:
    """Update a JSON config file that uses the mcpServers/mcp_servers shape."""
    data: Dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Leave the file untouched if it's not valid JSON.
                raise RuntimeError(f"Config is not valid JSON: {path}")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(path)

    key = "mcpServers"
    if "mcp_servers" in data and "mcpServers" not in data:
        key = "mcp_servers"

    servers = data.get(key) or {}
    desired = {"command": command, "args": args, "env": {}}
    changed = servers.get("gdb-mcp") != desired
    servers["gdb-mcp"] = desired
    data[key] = servers

    if changed or not path.exists():
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    return changed


def _update_codex_config(path: Path, command: str, args: List[str]) -> bool:
    """Update Codex CLI config.toml to add the MCP server."""
    data: Dict = {}
    if path.exists():
        with path.open("rb") as f:
            try:
                data = tomllib.load(f)
            except Exception:
                raise RuntimeError(f"Config is not valid TOML: {path}")

    servers = data.get("mcp_servers") or {}
    desired = {"command": command, "args": args}
    changed = servers.get("gdb-mcp") != desired
    servers["gdb-mcp"] = desired
    data["mcp_servers"] = servers

    if changed or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            toml.dump(data, f)
    return changed


def _targets() -> List[InstallTarget]:
    home = Path.home()
    return [
        InstallTarget(
            name="Codex CLI",
            path=home / ".codex" / "config.toml",
            kind="toml",
            create_if_parent_exists=True,
        ),
        InstallTarget(
            name="Claude Desktop (Linux)",
            path=home / ".config" / "anthropic" / "claude_desktop_config.json",
            kind="json",
            create_if_parent_exists=True,
        ),
        InstallTarget(
            name="Claude Desktop (Alt path)",
            path=home / ".config" / "Claude" / "claude_desktop_config.json",
            kind="json",
            create_if_parent_exists=True,
        ),
        InstallTarget(
            name="Claude Desktop (macOS)",
            path=home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
            kind="json",
            create_if_parent_exists=True,
        ),
        InstallTarget(
            name="Cursor",
            path=home / ".cursor" / "mcp.json",
            kind="json",
            create_if_parent_exists=True,
        ),
        InstallTarget(
            name="Cursor (macOS)",
            path=home / "Library" / "Application Support" / "Cursor" / "mcp.json",
            kind="json",
            create_if_parent_exists=True,
        ),
        InstallTarget(
            name="Windsurf",
            path=home / ".config" / "Windsurf" / "mcp.json",
            kind="json",
            create_if_parent_exists=True,
        ),
    ]


def install(command: str | None = None, args: List[str] | None = None) -> List[Dict[str, str]]:
    """Attempt to configure all detected MCP-aware clients.

    Returns a list of dicts describing the outcome for each target.
    """
    command_path = _resolve_command(command)
    args = args or []

    results: List[Dict[str, str]] = []
    seen_paths: set[Path] = set()

    for target in _targets():
        if target.path in seen_paths:
            continue
        seen_paths.add(target.path)

        if not target.path.exists() and not target.create_if_parent_exists:
            results.append(
                {
                    "target": target.name,
                    "path": str(target.path),
                    "status": "skipped",
                    "reason": "config file not found",
                }
            )
            continue

        if target.create_if_parent_exists and not target.path.exists():
            # Only create if the parent directory is present (indicates the app is installed).
            if not target.path.parent.exists():
                results.append(
                    {
                        "target": target.name,
                        "path": str(target.path),
                        "status": "skipped",
                        "reason": "client directory not found",
                    }
                )
                continue

        try:
            if target.kind == "toml":
                changed = _update_codex_config(target.path, command_path, args)
            else:
                changed = _update_json_mcp_config(
                    target.path, command_path, args, create=target.create_if_parent_exists
                )
            results.append(
                {
                    "target": target.name,
                    "path": str(target.path),
                    "status": "updated" if changed else "unchanged",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                {
                    "target": target.name,
                    "path": str(target.path),
                    "status": "error",
                    "reason": str(exc),
                }
            )

    return results
