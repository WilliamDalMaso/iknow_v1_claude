"""Reproducibility manifest for Phase 1 run artifacts.

Phase 1 runs are written under ``data/runs/`` which is git-ignored, so a fresh
clone cannot verify the documented canonical state. This tool captures a
committed manifest of ``filename -> {sha256, size}`` for a run directory, and
verifies a later run against it. It does not read, change, or promote any
extraction output; it only hashes bytes.

CLI:
    python3 src/baseline_manifest.py compute <run_dir> <manifest_path>
    python3 src/baseline_manifest.py verify  <run_dir> <manifest_path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_manifest(run_dir: Path) -> dict[str, dict[str, object]]:
    """Hash every file under ``run_dir`` recursively, keyed by POSIX relpath."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        entries[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return entries


def write_manifest(run_dir: Path, manifest_path: Path, *, label: str) -> dict[str, object]:
    entries = compute_manifest(run_dir)
    payload = {
        "label": label,
        "run_dir": run_dir.expanduser().resolve().name,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_manifest(run_dir: Path, manifest_path: Path) -> dict[str, list[str]]:
    """Compare a run directory against a stored manifest.

    Returns a dict with ``missing``, ``unexpected``, and ``changed`` lists. An
    empty value for all three means the run reproduces the manifest exactly.
    """
    stored = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    expected: dict[str, dict[str, object]] = stored["files"]
    actual = compute_manifest(run_dir)
    expected_keys = set(expected)
    actual_keys = set(actual)
    changed = sorted(
        key
        for key in expected_keys & actual_keys
        if actual[key]["sha256"] != expected[key]["sha256"]
    )
    return {
        "missing": sorted(expected_keys - actual_keys),
        "unexpected": sorted(actual_keys - expected_keys),
        "changed": changed,
    }


def manifest_matches(result: dict[str, list[str]]) -> bool:
    return not (result["missing"] or result["unexpected"] or result["changed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute or verify a Phase 1 artifact manifest.")
    sub = parser.add_subparsers(dest="command", required=True)
    compute = sub.add_parser("compute", help="Write a manifest for a run directory.")
    compute.add_argument("run_dir", type=Path)
    compute.add_argument("manifest_path", type=Path)
    compute.add_argument("--label", default="", help="Human label recorded in the manifest.")
    verify = sub.add_parser("verify", help="Verify a run directory against a manifest.")
    verify.add_argument("run_dir", type=Path)
    verify.add_argument("manifest_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compute":
        payload = write_manifest(args.run_dir, args.manifest_path, label=args.label)
        print(f"wrote {payload['file_count']} entries to {args.manifest_path}")
        return
    result = verify_manifest(args.run_dir, args.manifest_path)
    if manifest_matches(result):
        print("MATCH: run reproduces the manifest exactly")
        return
    for category in ("missing", "unexpected", "changed"):
        for name in result[category]:
            print(f"{category.upper()}: {name}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
