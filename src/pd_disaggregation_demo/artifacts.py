"""Checksummed, bounded-memory native llama slot artifacts."""

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .models import ArtifactManifest, PrefillArtifact, RuntimeFingerprint

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(directory: Path) -> PrefillArtifact:
    manifest = ArtifactManifest.model_validate_json((directory / "manifest.json").read_text())
    artifact = PrefillArtifact(manifest=manifest, state_path=directory / "state.bin")
    validate_artifact(artifact)
    return artifact


def validate_artifact(
    artifact: PrefillArtifact, fingerprint: RuntimeFingerprint | None = None
) -> None:
    manifest = artifact.manifest
    if manifest.expires_at and manifest.expires_at <= datetime.now(UTC):
        raise ValueError("Prefill artifact has expired")
    if not artifact.state_path.is_file():
        raise ValueError("Artifact state file is missing")
    if artifact.state_path.stat().st_size != manifest.state_bytes:
        raise ValueError("Artifact state size mismatch")
    if sha256_file(artifact.state_path) != manifest.state_sha256:
        raise ValueError("Artifact state SHA256 mismatch")
    if fingerprint is not None and manifest.fingerprint != fingerprint:
        raise ValueError("Artifact fingerprint does not match decode runtime")


def write_manifest(artifact: PrefillArtifact) -> None:
    directory = artifact.state_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(artifact.manifest.model_dump_json(indent=2))
    temporary.replace(directory / "manifest.json")


def delete_artifact(artifact: PrefillArtifact) -> None:
    # Restrict deletion to an artifact directory, never an arbitrary state parent.
    directory = artifact.state_path.parent
    if directory.name == artifact.manifest.artifact_id:
        shutil.rmtree(directory, ignore_errors=True)
