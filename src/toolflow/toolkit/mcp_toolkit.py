from typing import List, Dict, Any
import asyncio
from ..core.protocols import BaseAsyncToolKit
from ..core.logging_utils import get_toolflow_logger, log_tool_execution_error
from fastmcp import Client as FastMCPClient

logger = get_toolflow_logger("mcp_toolkit")

class MCPToolkit(BaseAsyncToolKit):
    """Async MCP toolkit implementation that wraps a fastmcp.Client."""

    def __init__(self, client: FastMCPClient):
        self._client = client

    async def list_tools(self) -> List[Dict[str, Any]]:
        async with self._client:
            tools = [tool.model_dump() for tool in await self._client.list_tools()]
            return self._convert_to_internal_format(tools)
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        try:
            async with self._client:
                result = await self._client.call_tool(name, arguments)
                if hasattr(result, "structured_content"):
                    return result.structured_content
                else:
                    raise ValueError("MCP server returned an unexpected result", result)
        except Exception as e:
            log_tool_execution_error(logger, name, e)
            raise


    async def ping(self) -> bool:
        try:
            if not self._client.is_connected():
                await self._client._connect()
            return await self._client.ping()
        except Exception:
            self._suppress_background_exceptions()
            return False

    async def close(self) -> None:
        try:
            if self._client.is_connected(): 
                await self._client.close()
        except Exception:
            pass  # Ignore cleanup errors
        finally:
            self._suppress_background_exceptions()

    async def __aenter__(self) -> "MCPToolkit":
        try:
            if not self._client.is_connected():
                await self._client._connect()
        except Exception:
            self._suppress_background_exceptions()
            # Still return self, but connection failed
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            await self.close()
        except Exception:
            pass  # Ignore cleanup errors during context exit
    
    def _convert_to_internal_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                    "strict": False,
                }
            }
            for tool in tools
        ]
    
    def _suppress_background_exceptions(self):
        """Suppress asyncio background task exceptions to reduce noise in logs."""
        try:
            # Get all pending tasks and cancel them quietly
            pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
            for task in pending_tasks:
                if hasattr(task, '_coro') and 'fastmcp' in str(task._coro):
                    task.cancel()
                    try:
                        # Wait briefly for task to finish
                        asyncio.create_task(asyncio.wait_for(task, timeout=0.1))
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass  # Suppress these exceptions
        except Exception:
            pass  # Ignore any errors in cleanup
