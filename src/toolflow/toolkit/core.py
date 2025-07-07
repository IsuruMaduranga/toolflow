from fastmcp import Client as FastMCPClient
from .mcp_toolkit import MCPToolkit


class ToolKit:
    """Factory class for creating different types of toolkits."""
    
    @staticmethod
    def from_mcp(client: FastMCPClient) -> MCPToolkit:
        """
        Create an MCP toolkit from a configuration, client, or URL.
        
        Args:
            client: fastmcp.Client instance
                
        Returns:
            SupportsToolKit: An MCP toolkit implementation
            
        Raises:
            ImportError: If fastmcp is not installed
            TypeError: If config_or_client is not a supported type
        """
        from .mcp_toolkit import MCPToolkit
        
        # Normalize config to fastmcp.Client
        if not isinstance(client, FastMCPClient):
            raise TypeError(
                f"Invalid MCP config/client type: {type(client)}. "
                "Expected fastmcp.Client instance."
            )

        return MCPToolkit(client)
