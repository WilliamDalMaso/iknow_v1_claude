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
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Keys whose values are run-time volatile (a re-run changes them) and so are
# excluded from the *normalized* content fingerprint used for regression.
VOLATILE_KEYS = {"created_at", "generated_at", "generated", "run_id", "source_pdf"}
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_TEXT_SUFFIXES = {".html", ".txt", ".csv", ".md"}


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


# --- Normalized content fingerprint (for golden-output regression) -----------
# The Phase 1 pipeline is deterministic given the same PDF, except for embedded
# timestamps, the run_id, and absolute paths. These helpers strip that volatility
# so a re-run can be compared for semantic stability rather than byte identity.

def _strip_volatile(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: ("<volatile>" if k in VOLATILE_KEYS else _strip_volatile(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def _neutralize_text(text: str, run_dir: Path) -> str:
    text = text.replace(str(run_dir.resolve()), "<run_dir>")
    text = text.replace(str(ROOT), "<root>")
    text = text.replace(run_dir.name, "<run_id>")
    return _TIMESTAMP.sub("<ts>", text)


def canonical_content(path: Path, run_dir: Path) -> bytes:
    """Return run-invariant bytes for a single artifact, for hashing.

    JSON/JSONL: volatile keys stripped, keys sorted. Text: timestamps, run_id,
    and absolute paths neutralized. Anything else (e.g. page images) is returned
    as-is because it is already byte-reproducible.
    """
    if path.suffix == ".json":
        try:
            payload = json.dumps(_strip_volatile(json.loads(path.read_text(encoding="utf-8"))), sort_keys=True)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return path.read_bytes()
        return _neutralize_text(payload, run_dir).encode("utf-8")
    if path.suffix == ".jsonl":
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.dumps(_strip_volatile(json.loads(line)), sort_keys=True))
            except json.JSONDecodeError:
                lines.append(line)
        return _neutralize_text("\n".join(lines), run_dir).encode("utf-8")
    if path.suffix in _TEXT_SUFFIXES:
        return _neutralize_text(path.read_text(encoding="utf-8", errors="replace"), run_dir).encode("utf-8")
    return path.read_bytes()


def compute_content_fingerprint(run_dir: Path) -> dict[str, str]:
    """Map each file under ``run_dir`` to the sha256 of its run-invariant content."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    out: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        out[rel] = hashlib.sha256(canonical_content(path, run_dir)).hexdigest()
    return out


def write_content_fingerprint(run_dir: Path, manifest_path: Path, *, label: str) -> dict[str, object]:
    entries = compute_content_fingerprint(run_dir)
    payload = {"kind": "normalized-content-fingerprint", "label": label, "file_count": len(entries), "files": entries}
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_content_fingerprint(run_dir: Path, manifest_path: Path) -> dict[str, list[str]]:
    stored = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))["files"]
    actual = compute_content_fingerprint(run_dir)
    stored_keys, actual_keys = set(stored), set(actual)
    changed = sorted(k for k in stored_keys & actual_keys if stored[k] != actual[k])
    return {
        "missing": sorted(stored_keys - actual_keys),
        "unexpected": sorted(actual_keys - stored_keys),
        "changed": changed,
    }


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
    fp = sub.add_parser("fingerprint", help="Write a normalized content fingerprint for a run.")
    fp.add_argument("run_dir", type=Path)
    fp.add_argument("manifest_path", type=Path)
    fp.add_argument("--label", default="")
    fpv = sub.add_parser("verify-fingerprint", help="Verify a run against a normalized fingerprint.")
    fpv.add_argument("run_dir", type=Path)
    fpv.add_argument("manifest_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compute":
        payload = write_manifest(args.run_dir, args.manifest_path, label=args.label)
        print(f"wrote {payload['file_count']} entries to {args.manifest_path}")
        return
    if args.command == "fingerprint":
        payload = write_content_fingerprint(args.run_dir, args.manifest_path, label=args.label)
        print(f"wrote {payload['file_count']} normalized fingerprints to {args.manifest_path}")
        return
    if args.command == "verify-fingerprint":
        result = verify_content_fingerprint(args.run_dir, args.manifest_path)
        if manifest_matches(result):
            print("MATCH: run reproduces the normalized fingerprint")
            return
        for category in ("missing", "unexpected", "changed"):
            for name in result[category]:
                print(f"{category.upper()}: {name}")
        raise SystemExit(1)
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
