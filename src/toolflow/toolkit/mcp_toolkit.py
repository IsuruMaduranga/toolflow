from typing import List, Dict, Any
from ..core.protocols import BaseAsyncToolKit
from fastmcp import Client as FastMCPClient

class MCPToolkit(BaseAsyncToolKit):
    """Async MCP toolkit implementation that wraps a fastmcp.Client."""

    def __init__(self, client: FastMCPClient):
        self._client = client

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self._client.is_connected():
            await self._client._connect()
            return await self._client.list_tools()

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if not self._client.is_connected():
            await self._client._connect()
            return await self._client.call_tool(name, args)

    async def ping(self) -> bool:
        if not self._client.is_connected():
            await self._client._connect()
        return await self._client.ping()

    async def close(self) -> None:
        if self._client.is_connected(): 
            await self._client.close()

    async def __aenter__(self) -> "MCPToolkit":
        if not self._client.is_connected():
            await self._client._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
