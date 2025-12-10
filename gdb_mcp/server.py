import argparse
import logging
from typing import Dict, List, Optional

from .gdb_session import DEFAULT_PROMPTS, GdbNotStarted, GdbSession
from .installer import install as run_install

try:
    # The Model Context Protocol python package is expected to be installed.
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "The `mcp` package is required. Install it with `pip install mcp`."
    ) from exc

log = logging.getLogger(__name__)

server = FastMCP("gdb-mcp")
_sessions: Dict[str, GdbSession] = {}


def _get_session(session_id: str) -> GdbSession:
    try:
        return _sessions[session_id]
    except KeyError as exc:
        raise GdbNotStarted(f"No gdb session with id '{session_id}' exists.") from exc


@server.tool(
    description="Start a new gdb session for a binary and return the session id.",
)
async def start_binary(
    binary_path: str,
    args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    load_init: bool = True,
    start_timeout: float = 30.0,
    prompt: Optional[str] = None,
    force_prompt: bool = True,
) -> dict:
    """
    Launch gdb against the provided binary.

    :param binary_path: Path to the target binary.
    :param args: Optional list of arguments passed to the target binary.
    :param cwd: Optional working directory for the debuggee.
    :returns: Session metadata including the initial gdb banner.
    """
    prompt_list = [prompt] if prompt else DEFAULT_PROMPTS.copy()
    # Deduplicate while preserving order
    seen = set()
    prompt_list = [p for p in prompt_list if not (p in seen or seen.add(p))]
    session = GdbSession(
        binary_path,
        args or [],
        cwd=cwd,
        load_init=load_init,
        start_timeout=start_timeout,
        prompts=prompt_list,
        force_prompt=force_prompt,
        prompt_string=prompt if prompt else None,
    )
    banner = await session.start()
    _sessions[session.session_id] = session
    return {
        "session_id": session.session_id,
        "banner": banner,
        "session": session.describe(),
    }


@server.tool(
    description="Start gdb and attach to an existing process id.",
)
async def attach_to_pid(
    pid: int,
    cwd: Optional[str] = None,
    load_init: bool = True,
    start_timeout: float = 30.0,
    prompt: Optional[str] = None,
    force_prompt: bool = True,
) -> dict:
    """
    Start gdb without a target binary and attach to a running PID.

    :param pid: Process identifier to attach to.
    :param cwd: Optional working directory used by gdb.
    :returns: Session metadata including output from gdb when attaching.
    """
    prompt_list = [prompt] if prompt else DEFAULT_PROMPTS.copy()
    seen = set()
    prompt_list = [p for p in prompt_list if not (p in seen or seen.add(p))]
    session = GdbSession(
        target=None,
        cwd=cwd,
        load_init=load_init,
        start_timeout=start_timeout,
        prompts=prompt_list,
        force_prompt=force_prompt,
        prompt_string=prompt if prompt else None,
    )
    banner = await session.start()
    attach_output = await session.attach(pid)
    _sessions[session.session_id] = session
    return {
        "session_id": session.session_id,
        "banner": banner,
        "attach_output": attach_output,
        "session": session.describe(),
    }


@server.tool(
    description="Execute a raw gdb command within a session and return the output.",
)
async def gdb_command(
    session_id: str, command: str, timeout_seconds: Optional[float] = 15.0
) -> dict:
    """
    Run a gdb command and capture the textual output.

    :param session_id: Active session id.
    :param command: Command string to execute (e.g., 'break main', 'run').
    :param timeout_seconds: How long to wait for the gdb prompt before timing out.
    :returns: The plain text output from gdb.
    """
    session = _get_session(session_id)
    output = await session.run_command(command, timeout=timeout_seconds)
    return {"output": output}


@server.tool(description="List active gdb sessions.")
async def list_sessions() -> dict:
    """Return summary information for all active sessions."""
    return {"sessions": [session.describe() for session in _sessions.values()]}


@server.tool(description="Shut down a gdb session and remove it from the registry.")
async def stop_session(session_id: str) -> dict:
    """Terminate a gdb session."""
    session = _get_session(session_id)
    await session.shutdown()
    del _sessions[session_id]
    return {"stopped": True}


@server.tool(description="Send multiple gdb commands in sequence.")
async def batch_commands(
    session_id: str,
    commands: List[str],
    timeout_seconds: Optional[float] = 15.0,
) -> dict:
    """
    Execute a list of commands sequentially.

    :param session_id: Active session id.
    :param commands: Commands to run in order.
    :param timeout_seconds: Timeout applied to each command individually.
    :returns: List of outputs aligned with the input commands.
    """
    session = _get_session(session_id)
    outputs: list[dict[str, str]] = []
    for command in commands:
        out = await session.run_command(command, timeout=timeout_seconds)
        outputs.append({"command": command, "output": out})
    return {"outputs": outputs}


@server.tool(
    description="Report whether the given session still has a live gdb process.",
)
async def session_status(session_id: str) -> dict:
    session = _get_session(session_id)
    return {"alive": session.is_alive(), "session": session.describe()}


def main() -> None:
    """Entry point used by `python -m gdb_mcp.server` or the console script."""
    parser = argparse.ArgumentParser(description="GDB MCP server")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Configure detected MCP-aware clients to use this server.",
    )
    parser.add_argument(
        "--install-command",
        help="Optional command path to record in client configs (defaults to this executable).",
    )
    parser.add_argument(
        "--install-args",
        nargs="*",
        default=None,
        help="Optional arguments to record alongside the command.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.install:
        results = run_install(command=args.install_command, args=args.install_args)
        for res in results:
            line = f"[{res['target']}] {res['status']}"
            reason = res.get("reason")
            if reason:
                line += f" - {reason}"
            line += f" ({res['path']})"
            print(line)
        return

    server.run()


if __name__ == "__main__":
    main()
