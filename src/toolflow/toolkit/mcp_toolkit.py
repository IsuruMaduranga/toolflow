from typing import List, Dict, Any
from ..core.protocols import BaseAsyncToolKit
from fastmcp import Client as FastMCPClient

class MCPToolkit(BaseAsyncToolKit):
    """Async MCP toolkit implementation that wraps a fastmcp.Client."""
    
    __requires_async__ = True

    def __init__(self, client: FastMCPClient):
        self._client = client

    async def list_tools(self) -> List[Dict[str, Any]]:
        return await self._client.list_tools()

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        return await self._client.call_tool(name, args)

    async def ping(self) -> bool:
        try:
            await self.list_tools()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
