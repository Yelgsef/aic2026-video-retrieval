from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    archive: Path
    workspace: Path
    manifests: Path
    reports: Path
    features: Path
    samples: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "PipelinePaths":
        root_path = Path(root).expanduser().resolve()
        workspace = root_path / "workspace"
        return cls(
            root=root_path,
            archive=root_path / "archive",
            workspace=workspace,
            manifests=workspace / "manifests",
            reports=workspace / "reports",
            features=workspace / "features",
            samples=workspace / "samples",
        )

    def ensure_output_dirs(self) -> None:
        self.manifests.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.features.mkdir(parents=True, exist_ok=True)
        self.samples.mkdir(parents=True, exist_ok=True)
