# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Security Considerations

### Local Function Execution

**Important**: Toolflow executes all tool functions **locally** on your machine. This means:

- **No sandboxing**: Functions run with the same permissions as your Python process
- **Tool author responsibility**: Tool authors are responsible for implementing appropriate sandboxing and security measures
- **Code execution**: Tools can access your file system, network, and other system resources
- **Dependencies**: Tools can import and use any installed Python packages

### Security Best Practices

1. **Review tool code**: Always review the source code of tools before using them
2. **Use trusted tools**: Only use tools from trusted sources
3. **Implement sandboxing**: Consider using containers, virtual environments, or other isolation techniques
4. **Limit permissions**: Run your application with minimal necessary permissions
5. **Monitor execution**: Log and monitor tool execution for suspicious activity

### Reporting Security Issues

If you discover a security vulnerability, please:

1. **Do not** open a public GitHub issue
2. Email security details to: [imwijesiri@gmail.com](mailto:imwijesiri@gmail.com)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Security Updates

- Security patches will be released as patch versions (e.g., 0.3.1, 0.3.2)
- Critical security issues will be addressed within 48 hours
- Non-critical issues will be addressed in the next regular release

### Tool Development Guidelines

When developing tools for Toolflow:

1. **Validate inputs**: Always validate and sanitize tool inputs
2. **Use safe defaults**: Provide safe default values for all parameters
3. **Handle errors gracefully**: Don't expose sensitive information in error messages
4. **Document security implications**: Clearly document any security considerations
5. **Test thoroughly**: Test tools with various input types and edge cases

### Example Secure Tool

```python
import os
from pathlib import Path

def safe_file_reader(file_path: str, max_size: int = 1024 * 1024) -> str:
    """
    Safely read a file with size limits and path validation.
    
    Args:
        file_path: Path to the file to read
        max_size: Maximum file size in bytes (default: 1MB)
    
    Returns:
        File contents as string
    """
    # Validate and sanitize file path
    path = Path(file_path).resolve()
    
    # Security checks
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.is_dir():
        raise ValueError(f"Path is a directory: {file_path}")
    
    # Check file size
    if path.stat().st_size > max_size:
        raise ValueError(f"File too large: {path.stat().st_size} bytes (max: {max_size})")
    
    # Read file safely
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
```

## Contact

For security-related questions or concerns:
- Email: [imwijesiri@gmail.com](mailto:imwijesiri@gmail.com)
- GitHub: [Create a private security advisory](https://github.com/isurumaduranga/toolflow/security/advisories)
