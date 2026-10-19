"""Local diagnostic files are recorded without requiring a manual export."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


def save_diagnostic(kind, report):
    root = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "raw-rig/diagnostics"
    path = root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{kind}_{uuid.uuid4().hex[:8]}.json"
    value = {**report, "diagnostic_path": str(path)}
    try:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return str(path)
    except OSError:
        return None  # Diagnostic export must never replace the actual error.
