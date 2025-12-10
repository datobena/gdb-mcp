import asyncio
import os
import re
import shutil
import signal
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


class GdbError(Exception):
    """Base error for GDB control failures."""


class GdbNotFound(GdbError):
    """Raised when gdb is not available on PATH."""


class GdbNotStarted(GdbError):
    """Raised when a command is issued before a session is running."""


class GdbCrashed(GdbError):
    """Raised when the underlying gdb process dies unexpectedly."""


class GdbTimeout(GdbError):
    """Raised when gdb does not produce a prompt within the expected time."""


ANSI_RE = re.compile(rb"\x1B\[[0-?]*[ -/]*[@-~]")
DEFAULT_PROMPTS = ["(gdb)", "(pwndbg)", "gef>"]


@dataclass
class GdbSession:
    """Manage a single gdb instance over stdio."""

    target: Optional[str]
    args: List[str] = field(default_factory=list)
    cwd: Optional[str] = None
    load_init: bool = True
    start_timeout: float = 30.0
    prompts: List[str] = field(default_factory=lambda: DEFAULT_PROMPTS.copy())
    force_prompt: bool = True
    prompt_string: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    process: Optional[asyncio.subprocess.Process] = field(init=False, default=None)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if not self.prompts:
            self.prompts = DEFAULT_PROMPTS.copy()
        if self.prompt_string and self.prompt_string not in self.prompts:
            self.prompts = [self.prompt_string] + self.prompts

    async def start(self) -> str:
        """Start gdb and return the initial banner/output."""
        if shutil.which("gdb") is None:
            raise GdbNotFound("gdb is not on PATH; install it to debug binaries.")
        if self.target and not os.path.exists(self.target):
            raise GdbError(f"Target binary does not exist: {self.target}")

        cmd: list[str] = ["gdb", "--quiet"]
        if not self.load_init:
            cmd.append("--nx")
        prompt_to_set = self.prompt_string or (self.prompts[0] if self.prompts else "(gdb)")
        if self.force_prompt and prompt_to_set:
            cmd.extend(["-ex", f"set prompt {prompt_to_set}"])
        if self.target:
            cmd.append(self.target)
        if self.args:
            cmd.append("--args")
            cmd.extend(self.args)

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.cwd,
        )

        banner = await self._read_until_prompt(timeout=self.start_timeout)
        await self.run_command("set pagination off", timeout=3.0)
        await self.run_command("set confirm off", timeout=3.0)
        await self.run_command("set breakpoint pending on", timeout=3.0)
        return banner

    def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def run_command(self, command: str, timeout: Optional[float] = 15.0) -> str:
        """Send a raw gdb command and return the captured output (without the prompt)."""
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise GdbNotStarted("gdb session has not been started yet.")
        if self.process.returncode is not None:
            raise GdbCrashed(f"gdb exited with code {self.process.returncode}")

        async with self._lock:
            self.process.stdin.write(command.encode() + b"\n")
            await self.process.stdin.drain()
            return await self._read_until_prompt(timeout=timeout)

    async def attach(self, pid: int, timeout: Optional[float] = 10.0) -> str:
        """Attach to a running PID."""
        return await self.run_command(f"attach {pid}", timeout=timeout)

    async def shutdown(self) -> None:
        """Politely close the session and clean up the process."""
        if self.process is None:
            return
        if self.process.stdin and self.process.returncode is None:
            try:
                self.process.stdin.write(b"quit\n")
                await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self.process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        elif self.process.returncode is None:
            await self.process.wait()

    async def _read_until_prompt(self, timeout: Optional[float]) -> str:
        """Read stdout until we see a gdb prompt."""
        if self.process is None or self.process.stdout is None:
            raise GdbNotStarted("gdb session is not active.")

        prompt_bytes = [p.encode(errors="replace") for p in self.prompts if p]
        chunks: list[bytes] = []

        async def _reader() -> str:
            while True:
                chunk = await self.process.stdout.read(1024)
                if not chunk:
                    break
                chunks.append(chunk)
                joined = b"".join(chunks)
                sanitized = ANSI_RE.sub(b"", joined)
                matches = [
                    (sanitized.find(p), p)
                    for p in prompt_bytes
                    if sanitized.find(p) != -1
                ]
                if matches:
                    idx, found = min(matches, key=lambda x: x[0])
                    before = sanitized[:idx]
                    after = sanitized[idx + len(found) :]
                    pieces: list[str] = []
                    if before:
                        pieces.append(before.decode(errors="replace"))
                    if after.strip():
                        pieces.append(after.decode(errors="replace"))
                    return "".join(pieces).strip()
            joined = b"".join(chunks)
            sanitized = ANSI_RE.sub(b"", joined)
            return sanitized.decode(errors="replace").strip()

        try:
            return await asyncio.wait_for(_reader(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise GdbTimeout(
                f"Timed out waiting for gdb prompt after {timeout} seconds."
            ) from exc

    def describe(self) -> dict:
        """Return a serializable description of the session."""
        return {
            "id": self.session_id,
            "target": self.target,
            "args": self.args,
            "cwd": self.cwd or os.getcwd(),
            "alive": self.is_alive(),
            "load_init": self.load_init,
            "start_timeout": self.start_timeout,
            "prompts": self.prompts,
            "force_prompt": self.force_prompt,
            "prompt_string": self.prompt_string,
        }
