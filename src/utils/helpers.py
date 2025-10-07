import os
import time
import math
import datetime as dt
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

def retry_with_backoff(fn: Callable, retries: int = 3, base: float = 0.5, factor: float = 2.0, exceptions: Tuple = (Exception,), logger=None):
    def wrapper(*args, **kwargs):
        delay = base
        for i in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except exceptions as e:
                if i == retries:
                    if logger: logger.exception(f"Retry exhausted: {e}")
                    raise
                if logger: logger.warning(f"Retry {i+1}/{retries} after {delay:.2f}s due to: {e}")
                time.sleep(delay)
                delay *= factor
    return wrapper

def format_currency(value: Optional[float], currency: str = "USD", decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{currency} {value:,.{decimals}f}"