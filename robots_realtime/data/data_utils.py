from typing import Any, Dict


def recusive_flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix (used for recursion)
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary with keys like "parent.child.grandchild"
    
    Example:
        >>> d = {"left": {"pos": [1, 2, 3]}, "right": {"pos": [4, 5, 6]}}
        >>> recusive_flatten(d)
        {"left.pos": [1, 2, 3], "right.pos": [4, 5, 6]}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(recusive_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reverse_flatten(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Reverse the flattening operation, converting a flat dictionary back to nested structure.
    
    Args:
        d: Flattened dictionary with keys like "parent.child.grandchild"
        sep: Separator used in flattened keys
        
    Returns:
        Nested dictionary
    
    Example:
        >>> d = {"left.pos": [1, 2, 3], "right.pos": [4, 5, 6]}
        >>> reverse_flatten(d)
        {"left": {"pos": [1, 2, 3]}, "right": {"pos": [4, 5, 6]}}
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result

