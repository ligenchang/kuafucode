"""
MCP (Model Context Protocol) client — stdio transport.

Manages one or more external MCP servers launched as child processes.
Communicates via newline-delimited JSON-RPC 2.0 over stdin/stdout.

Protocol flow
─────────────
  client → server:  initialize  (capabilities negotiation)
  server → client:  initialize response
  client → server:  notifications/initialized
  client → server:  tools/list  → list of tool descriptors
  client → server:  tools/call  → tool result

Tool naming
───────────
  MCP tool names are namespaced as  mcp__{server_name}__{tool_name}
  to avoid collision with nvagent's built-in tools.  The MCP client
  transparently strips the prefix before forwarding to the subprocess.

Public API
──────────
  McpServerConfig                    — per-server configuration dataclass
  McpClient(servers)                 — manages multiple MCP servers
  await McpClient.start()            — launch all server subprocesses
  await McpClient.stop()             — terminate all subprocesses
  McpClient.tool_schemas             — OpenAI-format schemas for all MCP tools
  await McpClient.call_tool(name, args) → str  — call a namespaced tool

Usage
─────
  client = McpClient(config.mcp.servers)
  await client.start()
  schemas = client.tool_schemas           # merge with TOOL_SCHEMAS for LLM
  result  = await client.call_tool("mcp__filesystem__read_file", {"path": "..."})
  await client.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MCP protocol version we advertise during initialization
_MCP_PROTOCOL_VERSION = "2024-11-05"

# Prefix applied to every MCP tool name when registered with the agent
_MCP_TOOL_PREFIX = "mcp__"

# Seconds to wait for the subprocess to reply to initialize / tools/list
_INIT_TIMEOUT = 15.0
# Seconds to wait for a tool call result (long-running tools may need more)
_CALL_TIMEOUT = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class McpServerConfig:
    """Configuration for a single stdio-based MCP server."""
    name: str                             # logical name, used for tool namespacing
    command: str                          # executable to launch
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # If true, the server is skipped if the command is not found on PATH
    optional: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# JSON-RPC helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_request(id: int, method: str, params: Optional[dict] = None) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


def _make_notification(method: str, params: Optional[dict] = None) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


# ─────────────────────────────────────────────────────────────────────────────
# Single MCP server process
# ─────────────────────────────────────────────────────────────────────────────

class _McpServerProcess:
    """
    Manages a single stdio MCP server subprocess.

    Each request/response is matched by integer id.  A background reader task
    routes incoming lines to waiting ``asyncio.Future`` objects keyed on id.
    Notifications (no id) are discarded — we don't need server-push in Phase 1.
    """

    def __init__(self, cfg: McpServerConfig) -> None:
        self.cfg = cfg
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._next_id: int = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()          # serialise writes to stdin
        self._tools: list[dict] = []         # raw MCP tool descriptors
        self._started = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Launch subprocess and exchange initialize handshake.

        Returns True if ready, False if startup failed (caller marks optional).
        """
        env = {**os.environ, **self.cfg.env}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.cfg.command,
                *self.cfg.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
        except (FileNotFoundError, PermissionError) as exc:
            logger.debug("MCP server %r failed to start: %s", self.cfg.name, exc)
            return False

        self._reader_task = asyncio.ensure_future(self._reader_loop())
        ok = await self._initialize()
        if not ok:
            await self.stop()
            return False
        self._tools = await self._list_tools()
        self._started = True
        logger.debug(
            "MCP server %r ready — %d tools", self.cfg.name, len(self._tools)
        )
        return True

    async def stop(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except Exception:
                with_kill = asyncio.create_task(self._proc.kill())
                try:
                    await asyncio.wait_for(with_kill, timeout=2.0)
                except Exception:
                    pass
        self._started = False

    # ── background reader ─────────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        """Read newline-delimited JSON from stdout and dispatch to waiting futures."""
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        fut.set_result(msg)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("MCP reader loop for %r exited: %s", self.cfg.name, exc)
        finally:
            # Fail all pending futures so callers don't wait forever
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("MCP server process ended"))
            self._pending.clear()

    # ── JSON-RPC send/receive ─────────────────────────────────────────────────

    async def _send_request(
        self, method: str, params: Optional[dict], timeout: float
    ) -> dict:
        assert self._proc and self._proc.stdin
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        payload = _make_request(req_id, method, params)
        async with self._lock:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _send_notification(self, method: str, params: Optional[dict] = None) -> None:
        assert self._proc and self._proc.stdin
        payload = _make_notification(method, params)
        async with self._lock:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()

    # ── MCP protocol methods ──────────────────────────────────────────────────

    async def _initialize(self) -> bool:
        try:
            resp = await self._send_request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "nvagent", "version": "1.0"},
                },
                timeout=_INIT_TIMEOUT,
            )
        except Exception as exc:
            logger.debug("MCP initialize failed for %r: %s", self.cfg.name, exc)
            return False

        if "error" in resp:
            logger.debug(
                "MCP initialize error for %r: %s", self.cfg.name, resp["error"]
            )
            return False

        # Acknowledge initialization
        await self._send_notification("notifications/initialized")
        return True

    async def _list_tools(self) -> list[dict]:
        try:
            resp = await self._send_request(
                "tools/list", None, timeout=_INIT_TIMEOUT
            )
        except Exception as exc:
            logger.debug("MCP tools/list failed for %r: %s", self.cfg.name, exc)
            return []
        result = resp.get("result", {})
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on this server. Returns text result or error string."""
        if not self._started:
            return f"[MCP] Server '{self.cfg.name}' is not running."
        try:
            resp = await self._send_request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
                timeout=_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return f"[MCP] Tool '{tool_name}' timed out after {_CALL_TIMEOUT:.0f}s."
        except Exception as exc:
            return f"[MCP] Tool '{tool_name}' error: {exc}"

        if "error" in resp:
            err = resp["error"]
            return f"[MCP] Tool error: {err.get('message', str(err))}"

        result = resp.get("result", {})
        is_error = result.get("isError", False)
        content = result.get("content", [])

        # Flatten content blocks to a single string
        parts: list[str] = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "resource":
                resource = block.get("resource", {})
                uri = resource.get("uri", "")
                text = resource.get("text") or resource.get("blob", "")
                parts.append(f"[resource: {uri}]\n{text}" if uri else str(text))
            else:
                # Unknown content type — serialize to JSON
                parts.append(json.dumps(block))

        text = "\n".join(parts)
        if is_error:
            return f"[MCP error] {text}"
        return text or "(empty result)"

    # ── schema conversion ─────────────────────────────────────────────────────

    def openai_schemas(self) -> list[dict]:
        """Return MCP tool descriptors converted to OpenAI function-calling format."""
        schemas = []
        for tool in self._tools:
            raw_name = tool.get("name", "")
            if not raw_name:
                continue
            # Namespace: mcp__{server_name}__{tool_name}
            namespaced = f"{_MCP_TOOL_PREFIX}{self.cfg.name}__{raw_name}"
            input_schema = tool.get("inputSchema", {})
            schemas.append({
                "type": "function",
                "function": {
                    "name": namespaced,
                    "description": (
                        f"[MCP/{self.cfg.name}] "
                        + tool.get("description", "")
                    ),
                    "parameters": {
                        "type": input_schema.get("type", "object"),
                        "properties": input_schema.get("properties", {}),
                        "required": input_schema.get("required", []),
                    },
                },
            })
        return schemas


# ─────────────────────────────────────────────────────────────────────────────
# McpClient — multi-server manager
# ─────────────────────────────────────────────────────────────────────────────

class McpClient:
    """
    Manages a collection of stdio MCP server subprocesses.

    Typical lifecycle
    -----------------
      client = McpClient(config.mcp.servers)
      await client.start()                    # called once in Agent.__init__ or start()
      schemas = client.tool_schemas           # list[dict] in OpenAI format
      result  = await client.call_tool(name, args)
      await client.stop()                     # called on agent shutdown
    """

    def __init__(self, server_configs: list[McpServerConfig]) -> None:
        self._configs = server_configs
        self._servers: dict[str, _McpServerProcess] = {}

    async def start(self) -> None:
        """Start all configured servers concurrently. Skips optional servers that fail."""
        tasks = []
        for cfg in self._configs:
            proc = _McpServerProcess(cfg)
            self._servers[cfg.name] = proc
            tasks.append(proc.start())

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = []
        for cfg, result in zip(self._configs, results):
            ok = result if isinstance(result, bool) else False
            if not ok:
                logger.debug(
                    "MCP server %r failed to start%s",
                    cfg.name,
                    " (optional — continuing)" if cfg.optional else "",
                )
                if not cfg.optional:
                    raise RuntimeError(
                        f"Required MCP server '{cfg.name}' failed to start."
                    )
                failed.append(cfg.name)

        for name in failed:
            del self._servers[name]

    async def stop(self) -> None:
        """Terminate all running server subprocesses."""
        await asyncio.gather(
            *(proc.stop() for proc in self._servers.values()),
            return_exceptions=True,
        )
        self._servers.clear()

    @property
    def tool_schemas(self) -> list[dict]:
        """OpenAI-format tool schemas for every tool across all running servers."""
        schemas: list[dict] = []
        for proc in self._servers.values():
            schemas.extend(proc.openai_schemas())
        return schemas

    def is_mcp_tool(self, name: str) -> bool:
        """Return True if *name* is a namespaced MCP tool."""
        return name.startswith(_MCP_TOOL_PREFIX)

    async def call_tool(self, namespaced_name: str, args: dict) -> str:
        """
        Dispatch a namespaced MCP tool call to the correct server.

        namespaced_name format: ``mcp__{server_name}__{tool_name}``
        """
        if not namespaced_name.startswith(_MCP_TOOL_PREFIX):
            return f"[MCP] '{namespaced_name}' is not a namespaced MCP tool."

        remainder = namespaced_name[len(_MCP_TOOL_PREFIX):]  # "server_name__tool_name"
        sep = remainder.find("__")
        if sep == -1:
            return f"[MCP] Malformed tool name: '{namespaced_name}'"

        server_name = remainder[:sep]
        tool_name   = remainder[sep + 2:]

        proc = self._servers.get(server_name)
        if proc is None:
            return f"[MCP] Server '{server_name}' is not running."

        return await proc.call_tool(tool_name, args)
