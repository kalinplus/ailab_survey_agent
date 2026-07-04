"""Lightweight path guard for A-owned file operations."""

from __future__ import annotations

from pathlib import Path


class SandboxViolation(ValueError):
    pass


class SandboxGuard:
    def __init__(self, root_dir: Path, allowed_dirs: list[str] | None = None) -> None:
        self.root_dir = root_dir.resolve()
        self.allowed_roots = [
            (self.root_dir / name).resolve()
            for name in (allowed_dirs or ["cache", "requests", "output", "logs", "templates"])
        ]

    def resolve(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            candidate = path.resolve()
        else:
            candidate = (self.root_dir / path).resolve()

        if not any(candidate == root or root in candidate.parents for root in self.allowed_roots):
            raise SandboxViolation(f"Path is outside allowed project artifact directories: {candidate}")
        return candidate

    def ensure_existing_file(self, relative_path: str | Path) -> Path:
        path = self.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact does not exist: {path}")
        return path
