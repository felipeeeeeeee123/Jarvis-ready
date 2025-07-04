"""Security utilities for input sanitization and validation."""

import re
import logging
from typing import Optional, Set
import sympy as sp

logger = logging.getLogger(__name__)

# Whitelist of safe sympy functions and constants
SAFE_SYMPY_FUNCTIONS = {
    # Basic math functions
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
    'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'exp', 'log', 'ln', 'sqrt', 'cbrt', 'abs', 'sign',
    'floor', 'ceiling', 'round',
    # Constants
    'pi', 'e', 'I', 'oo', 'zoo', 'nan',
    # Variables (letters)
    'x', 'y', 'z', 't', 'a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w',
    # Numbers and operators are handled separately
}

# Dangerous patterns that should never be allowed
DANGEROUS_PATTERNS = [
    r'__',  # Double underscore (dunder methods)
    r'import\s+',  # Import statements
    r'exec\s*\(',  # exec() calls
    r'eval\s*\(',  # eval() calls
    r'open\s*\(',  # File operations
    r'os\.',  # OS module access
    r'sys\.',  # System module access
    r'subprocess',  # Subprocess calls
    r'globals\s*\(',  # Global scope access
    r'locals\s*\(',  # Local scope access
    r'dir\s*\(',  # Directory listing
    r'getattr\s*\(',  # Attribute access
    r'setattr\s*\(',  # Attribute setting
    r'hasattr\s*\(',  # Attribute checking
    r'delattr\s*\(',  # Attribute deletion
    r'__builtins__',  # Built-in functions
    r'__import__',  # Dynamic imports
]


def sanitize_mathematical_expression(expr_str: str) -> Optional[str]:
    """
    Sanitize a mathematical expression for safe use with sympy.
    
    Args:
        expr_str: The expression string to sanitize
        
    Returns:
        Sanitized expression string or None if unsafe
    """
    if not expr_str or not isinstance(expr_str, str):
        return None
    
    # Remove leading/trailing whitespace
    expr_str = expr_str.strip()
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, expr_str, re.IGNORECASE):
            logger.warning(f"Dangerous pattern detected in expression: {pattern}")
            return None
    
    # Allow only safe characters: letters, numbers, operators, parentheses, spaces
    safe_chars = re.compile(r'^[a-zA-Z0-9+\-*/^().,\s=<>!]+$')
    if not safe_chars.match(expr_str):
        logger.warning(f"Unsafe characters detected in expression: {expr_str}")
        return None
    
    # Limit expression length to prevent DoS
    if len(expr_str) > 1000:
        logger.warning(f"Expression too long: {len(expr_str)} characters")
        return None
    
    # Check for balanced parentheses
    if expr_str.count('(') != expr_str.count(')'):
        logger.warning("Unbalanced parentheses in expression")
        return None
    
    return expr_str


def safe_sympify(expr_str: str) -> Optional[sp.Basic]:
    """
    Safely convert a string to a sympy expression.
    
    Args:
        expr_str: The expression string to convert
        
    Returns:
        Sympy expression or None if unsafe/invalid
    """
    sanitized = sanitize_mathematical_expression(expr_str)
    if not sanitized:
        return None
    
    try:
        # Use sympy's parse_expr with strict evaluation
        expr = sp.parse_expr(sanitized, evaluate=False)
        
        # Additional safety check: ensure only safe functions are used
        if expr.atoms(sp.Function):
            func_names = {f.func.__name__ for f in expr.atoms(sp.Function)}
            unsafe_funcs = func_names - SAFE_SYMPY_FUNCTIONS
            if unsafe_funcs:
                logger.warning(f"Unsafe functions detected: {unsafe_funcs}")
                return None
        
        return expr
    except (sp.SympifyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse expression '{sanitized}': {e}")
        return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove any path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\.\.', '', filename)  # Remove path traversal
    filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    # Ensure it's not empty after sanitization
    if not filename:
        return "untitled"
    
    return filename


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key format.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if valid format, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format validation - adjust based on your API key requirements
    # This example assumes a 32-character hex string
    if len(api_key) < 16 or len(api_key) > 128:
        return False
    
    # Check for reasonable character set
    if not re.match(r'^[a-zA-Z0-9_.-]+$', api_key):
        return False
    
    return True