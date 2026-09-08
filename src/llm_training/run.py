from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RunIdentity:
    """Stable identity shared by local outputs and optional durable artifacts."""

    run_id: str
    output_dir: str
    artifact_uri: str | None = None


def _default_run_id(output_dir: Path) -> str:
    """Derive the same run id on every rank without cross-process coordination."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", output_dir.name).strip("-._") or "run"
    digest = hashlib.sha256(str(output_dir).encode()).hexdigest()[:8]
    return f"{name[:118]}-{digest}"


def resolve_run_identity(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    artifact_root_uri: str | None = None,
) -> RunIdentity:
    root = Path(output_dir).expanduser().resolve()
    resolved_id = run_id or _default_run_id(root)
    if not _RUN_ID.fullmatch(resolved_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-' (max 128 characters)"
        )
    artifact_uri = None
    if artifact_root_uri is not None:
        base = artifact_root_uri.strip().rstrip("/")
        if not base:
            raise ValueError("artifact_root_uri must be non-empty when set")
        artifact_uri = f"{base}/{resolved_id}/"
    return RunIdentity(resolved_id, str(root), artifact_uri)


def write_run_identity(identity: RunIdentity) -> Path:
    """Persist run identity and reject accidental reuse by another run."""
    root = Path(identity.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run.json"
    payload = asdict(identity)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(
                f"Output directory already belongs to a different run: {path}"
            )
        return path
    if any(root.iterdir()):
        raise ValueError(
            "Refusing to attach a new run identity to a non-empty output directory "
            f"without run.json: {root}"
        )
    temporary = root / "run.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path
