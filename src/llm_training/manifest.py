"""Portable artifact identity and integrity checks."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer):
    value = dict(
        vocab=tokenizer.get_vocab(),
        special_tokens=tokenizer.special_tokens_map,
        chat_template=getattr(tokenizer, "chat_template", None),
    )
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def model_fingerprint(path):
    root = Path(path)
    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and (
            p.name == "config.json"
            or p.suffix in {".safetensors", ".bin"}
            or p.name.endswith(".index.json")
        )
    )
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise FileNotFoundError(f"No model weights found in {root}")
    return hashlib.sha256(
        json.dumps({p.name: file_sha256(p) for p in files}, sort_keys=True).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ArtifactManifest:
    kind: str
    base_path: str
    base_fingerprint: str
    tokenizer_fingerprint: str
    vocab_size: int
    new_tokens: dict[str, int]
    training_config: dict
    generation_config: dict
    files: dict[str, str]
    format_version: int = 1

    @classmethod
    def load(cls, artifact_path):
        return cls(**json.loads((Path(artifact_path) / "artifact.json").read_text()))

    def validate(self, artifact_path, *, base_path=None):
        root = Path(artifact_path)
        if self.format_version != 1 or self.kind not in {
            "base",
            "checkpoint",
            "adapter",
            "model",
        }:
            raise ValueError("Unsupported artifact manifest")
        for name, digest in self.files.items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Artifact file escapes its directory")
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"Artifact file missing or changed: {name}")
        if (
            base_path is not None
            and model_fingerprint(base_path) != self.base_fingerprint
        ):
            raise ValueError("Base model fingerprint mismatch")
        tokenizer_root = root / "tokenizer" if self.kind == "checkpoint" else root
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        if (
            len(tokenizer) != self.vocab_size
            or tokenizer_fingerprint(tokenizer) != self.tokenizer_fingerprint
        ):
            raise ValueError("Tokenizer/vocabulary fingerprint mismatch")
        if any(
            tokenizer.get_vocab().get(token) != index
            for token, index in self.new_tokens.items()
        ):
            raise ValueError("Added token IDs changed")
        return self


def write_artifact_manifest(
    artifact_path,
    *,
    kind,
    base_path,
    tokenizer,
    new_tokens=None,
    training_config=None,
    generation_config=None,
):
    root = Path(artifact_path)
    if kind not in {"base", "checkpoint", "adapter", "model"}:
        raise ValueError(f"Unknown artifact kind: {kind}")
    files = {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in {"artifact.json", "artifact.tmp"}
    }
    value = ArtifactManifest(
        kind=kind,
        base_path=str(Path(base_path).resolve()),
        base_fingerprint=model_fingerprint(base_path),
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        vocab_size=len(tokenizer),
        new_tokens=dict(new_tokens or {}),
        training_config=dict(training_config or {}),
        generation_config=dict(generation_config or {}),
        files=files,
    )
    temporary = root / "artifact.tmp"
    temporary.write_text(json.dumps(asdict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "artifact.json")
    return value
