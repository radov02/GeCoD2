"""Creation timestamp for the .txt artifacts."""

import time


def now_str():
    """Local date and time as 'YYYY-MM-DD HH:MM:SS'."""
    return time.strftime('%Y-%m-%d %H:%M:%S')
