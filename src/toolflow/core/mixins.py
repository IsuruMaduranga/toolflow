from typing import Any
from .execution_loops import (
    sync_execution_loop, sync_streaming_execution_loop,
    async_execution_loop, async_streaming_execution_loop
)


class CreateMixin:
    """Mixin that provides common create() method logic for both sync and async wrappers."""
    
    def _prepare_kwargs(self, **kwargs: Any) -> dict:
        """Prepare kwargs by merging full_response and validating streaming + response_format."""
        # merge full_response with kwargs, but allow method-level override
        if "full_response" not in kwargs:
            kwargs["full_response"] = self.full_response
        
        # Validate streaming + response_format combination
        if kwargs.get("stream", False) and kwargs.get("response_format", None):
            raise ValueError("response_format is not supported for streaming")
        
        return kwargs
    
    def _create_sync(self, **kwargs: Any) -> Any:
        """Execute sync create with appropriate execution loop."""
        kwargs = self._prepare_kwargs(**kwargs)
        
        if kwargs.get("stream", False):
            return sync_streaming_execution_loop(handler=self.handler, **kwargs)
        else:
            return sync_execution_loop(handler=self.handler, **kwargs)
    
    async def _create_async(self, **kwargs: Any) -> Any:
        """Execute async create with appropriate execution loop.""" 
        kwargs = self._prepare_kwargs(**kwargs)
        
        if kwargs.get("stream", False):
            return async_streaming_execution_loop(handler=self.handler, **kwargs)
        else:
            return await async_execution_loop(handler=self.handler, **kwargs) 