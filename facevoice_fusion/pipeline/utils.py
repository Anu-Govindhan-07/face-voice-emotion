from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from rich.console import Console

console = Console()


def now_ts() -> float:
    return time.time()


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())
