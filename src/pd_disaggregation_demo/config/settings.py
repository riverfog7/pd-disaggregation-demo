from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENGINE_REVISION = "23b0202a189c44a54625aadcb37a946dd1d6278d"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDD_", env_file=".env", extra="ignore")

    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache/pd-disaggregation-demo")
    work_dir: Path = Field(default_factory=lambda: Path.cwd() / ".pd-demo")
    model_path: Path | None = None
    engine_binary: Path | None = None
    engine_revision: str = ENGINE_REVISION
    engine_port: int = Field(default=8081, ge=1, le=65535)
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    worker_port: int = Field(default=8090, ge=1, le=65535)
    remote_url: str = "http://127.0.0.1:8090"
    remote_ssh_host: str = "gpu_home"
    lan_ssh_host: str | None = None
    remote_repo: str = "/home/riverfog7/Workspace/pd-disaggregation-demo"
    git_revision: str = "unknown"
    request_timeout: float = 1200.0
    startup_timeout: float = 600.0
    artifact_ttl_seconds: int = 3600
    gpu_reserve_bytes: int = 1024**3
    remote_ram_reserve_bytes: int = 2 * 1024**3
    local_ram_reserve_bytes: int = 6 * 1024**3
    disk_reserve_bytes: int = 20 * 1024**3
    build_jobs: int = 2
    skip_admission: bool = False
    allow_metal_memory_overcommit: bool = False

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def artifacts_dir(self) -> Path:
        return self.work_dir / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.work_dir / "reports"
