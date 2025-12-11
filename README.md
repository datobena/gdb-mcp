# gdb-mcp

Model Context Protocol server that wraps `gdb` so an LLM can drive live debugging
sessions. The server exposes tools for starting a session on a binary, attaching
to an existing PID, running ad-hoc commands, batching commands, checking status,
and shutting sessions down.

> Requires `gdb` on the host and the `mcp` python package (v1+).

## Setup

You can install either inside a virtualenv (recommended) or directly:

```bash
# optional but recommended
python -m venv .venv
. .venv/bin/activate

pip install -e .
# or: pip install .
```

## Running the MCP server

You can start the server directly:

```bash
gdb-mcp
# or
python -m gdb_mcp.server
```

The server uses stdio for transport (the default for most MCP clients). Point
your MCP client configuration at the command above.

### One-shot client configuration

Run `gdb-mcp --install` to add the server to any detected MCP-aware clients
without editing config files by hand. The installer currently knows how to
update:

- Codex CLI (`~/.codex/config.toml`)
- Claude Desktop configs (common Linux/macOS paths)
- Cursor / Windsurf MCP config files (if present)

Use `--install-command /custom/path/to/gdb-mcp` or `--install-args ...` if you
need to override what gets written.

## Exposed tools

- `start_binary(binary_path, args=None, cwd=None, load_init=True, start_timeout=30.0, prompt=None, force_prompt=True)` – launch gdb
  against a target binary (with optional args) and return a session id and the
  initial banner. Set `load_init=False` to skip user gdbinit (default is to load
  it so helpers like GEF/pwndbg remain enabled). Increase `start_timeout` if
  loading extensions takes longer to reach the first prompt. Use `prompt` to
  force a specific prompt string (defaults cover `(gdb)`, `(pwndbg)`, and
  `gef>`). `force_prompt` sets the prompt to the chosen value after startup to
  avoid timeouts from customized prompts (e.g., pwndbg); set `force_prompt=False`
  if you need to keep your custom prompt unchanged.
- `attach_to_pid(pid, cwd=None, load_init=True, start_timeout=30.0, prompt=None, force_prompt=True)` – start gdb and attach to a
  running process (same `load_init`/timeout/prompt behavior as above).
- `gdb_command(session_id, command, timeout_seconds=15.0)` – execute a single
  gdb command in the given session and return the output.
- `batch_commands(session_id, commands, timeout_seconds=15.0)` – run a list of
  commands sequentially.
- `list_sessions()` – get a snapshot of all active sessions.
- `session_status(session_id)` – check if the gdb process is still alive.
- `stop_session(session_id)` – shut down the gdb process and remove it.

Each tool returns simple JSON so it is easy to route back into your LLM prompt.

## Notes and tips

- The server automatically disables pagination and confirmation prompts and
  enables pending breakpoints to keep interactions non-blocking.
- `timeout_seconds` applies per command. If you expect a program to run for a
  long time, pass a larger timeout or `None`.
- Output is captured from gdb stdout/stderr until the next `(gdb)` prompt. If
  you spawn a program that never returns to the prompt (e.g., it blocks on
  input), the call will time out.
- Common debugger prompts are detected automatically (`(gdb)`, `(pwndbg)`,
  `gef>`), even with ANSI colors. You can still pass a custom `prompt` if you
  use something nonstandard. By default the server forces the prompt to a stable
  value (`force_prompt=True`) so gdbinit customizations (like pwndbg) don’t
  prevent the initial prompt from being detected.

## Demo

Auto-playing preview:

![Demo](docs/media/demo.gif)

Source video: `docs/media/demo.mp4`

## Credits

- This project was built for my CSE 598 class, which emphasized using AI/LLMs in
  our workflow. I leaned on AI to write the entire project—including this
  README.
- Inspired by https://github.com/mrexodia/ida-pro-mcp, and their server logic
  was used as a starting point.
