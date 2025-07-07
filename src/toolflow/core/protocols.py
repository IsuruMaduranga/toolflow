from typing import Protocol, runtime_checkable, List, Dict, Any
from typing import AsyncContextManager

@runtime_checkable
class BaseToolKit(Protocol):
    """Protocol for objects that provide tool functionality through an interface."""

    def list_tools(self) -> List[Dict[str, Any]]: 
        """Return a list of tool schemas that this toolkit provides."""
        ...
    
    def call_tool(self, name: str, args: Dict[str, Any]) -> Any: 
        """Call a specific tool by name with the given arguments."""
        ...
    
    def ping(self) -> bool: 
        """Check if the toolkit is available/responsive."""
        return True
    
    def close(self) -> None: 
        """Clean up any resources used by the toolkit."""
        pass

@runtime_checkable
class BaseAsyncToolKit(Protocol, AsyncContextManager):
    """Protocol for objects that provide tool functionality through an interface."""

    async def list_tools(self) -> List[Dict[str, Any]]: 
        """Return a list of tool schemas that this toolkit provides."""
        ...
    
    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any: 
        """Call a specific tool by name with the given arguments."""
        ...
    
    async def ping(self) -> bool: 
        """Check if the toolkit is available/responsive."""
        return True
    
    async def close(self) -> None: 
        """Clean up any resources used by the toolkit."""
        ...

    async def __aenter__(self) -> Any:
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...
