import logging
import sys
from typing import Optional, Dict, Any

class ToolflowFormatter(logging.Formatter):
    """Custom formatter for Toolflow logs with colored output and better formatting."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color if terminal supports it
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Custom format for toolflow messages
        if record.name.startswith('toolflow'):
            return f"[Toolflow {record.levelname}] {record.getMessage()}"
        
        return super().format(record)


def get_toolflow_logger(name: str) -> logging.Logger:
    """Get a configured toolflow logger."""
    logger = logging.getLogger(f"toolflow.{name}")
    
    # Only add handler if it doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ToolflowFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # Default to WARNING and above
    
    return logger


def log_toolkit_connection_error(logger: logging.Logger, toolkit_name: str, error: Exception):
    """Log a meaningful toolkit connection error with context and suggestions."""
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Map common errors to user-friendly explanations
    error_explanations = {
        'ConnectError': 'Network connection failed',
        'ConnectionError': 'Unable to establish connection', 
        'OSError': 'System-level connection error',
        'TimeoutError': 'Connection timed out',
        'RuntimeError': 'Runtime connection issue',
        'HttpxConnectError': 'HTTP connection failed',
        'FastMCPError': 'MCP protocol error'
    }
    
    explanation = error_explanations.get(error_type, f'Connection error ({error_type})')
    
    # Build the main error message
    main_msg = f"Cannot connect to {toolkit_name}: {explanation}"
    
    # Add error details if helpful
    if error_msg and error_msg != error_type:
        # Truncate very long error messages
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."
        main_msg += f" - {error_msg}"
    
    logger.warning(main_msg)
    logger.warning(f"Continuing without {toolkit_name} tools - other tools remain available") 

def log_tool_execution_error(logger: logging.Logger, tool_name: str, error: Exception):
    """Log a meaningful tool execution error."""
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    main_msg = f"Tool '{tool_name}' execution failed: {error_type}"
    if error_msg and error_msg != error_type:
        if len(error_msg) > 150:
            error_msg = error_msg[:147] + "..."
        main_msg += f" - {error_msg}"
    
    logger.error(main_msg)
