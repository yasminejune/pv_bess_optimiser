"""Simple utility for consistent terminal output formatting in optimization runs."""


def print_info(message: str, verbose: bool = True) -> None:
    """Print informational message if verbose mode is enabled."""
    if verbose:
        print(f"Info: {message}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"Success: {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"Warning: {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"Error: {message}")
