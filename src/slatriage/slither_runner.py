"""Run Slither and normalize JSON alerts (P1)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_INSTALLED_SOLC: set[str] = set()
_FAILED_SOLC: set[str] = set()


def list_installed_solc() -> set[str]:
    try:
        proc = subprocess.run(["solc-select", "versions"], capture_output=True, timeout=30)
        text = _decode(proc.stdout) + "\n" + _decode(proc.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    found = set(re.findall(r"\b(\d+\.\d+\.\d+)\b", text))
    _INSTALLED_SOLC.update(found)
    return found


def ensure_solc(version: str) -> bool:
    """Install a solc via solc-select. Return True if the binary is usable."""
    if version in _INSTALLED_SOLC:
        return True
    if version in _FAILED_SOLC:
        return False
    if not _INSTALLED_SOLC:
        list_installed_solc()
    if version in _INSTALLED_SOLC:
        return True
    try:
        proc = subprocess.run(
            ["solc-select", "install", version],
            capture_output=True,
            timeout=180,
        )
        ok = proc.returncode == 0
        if ok:
            _INSTALLED_SOLC.add(version)
        else:
            _FAILED_SOLC.add(version)
        return ok
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def select_solc(version: str) -> bool:
    """Install and switch the global solc-select version (needed on Windows)."""
    if not ensure_solc(version):
        return False
    try:
        proc = subprocess.run(
            ["solc-select", "use", version, "--always-install"],
            capture_output=True,
            timeout=60,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="replace")


def _ascii_name(sol_path: Path) -> str:
    raw = f"{sol_path.parent.name}_{sol_path.name}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size < 2:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_text(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _stage_sol(sol_path: Path, tmp_root: Path) -> Path:
    """Copy .sol to an ASCII-only temp path so solc/Slither work on Windows."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    dest = tmp_root / _ascii_name(sol_path)
    shutil.copy2(sol_path, dest)
    return dest


def _run_once(
    staged_sol: Path,
    tmp_json: Path,
    timeout: int,
    solc_version: str | None,
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any] | None, str, str]:
    if tmp_json.exists():
        tmp_json.unlink()
    cmd = ["slither", str(staged_sol), "--json", str(tmp_json), "--disable-color"]
    if solc_version:
        cmd.extend(["--solc-solcs-select", solc_version])
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    stdout = _decode(proc.stdout)
    stderr = _decode(proc.stderr)
    payload = _load_json_file(tmp_json) or _load_json_text(stdout)
    return proc, payload, stdout, stderr


def run_slither_json(
    sol_path: Path,
    out_json: Path,
    timeout: int = 180,
    solc_version: str | None = None,
    solc_fallbacks: list[str] | None = None,
) -> dict[str, Any]:
    """
    Invoke Slither on a copy of the file under %TEMP% (ASCII path).
    Non-zero exit is OK when JSON with results.detectors is present.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    versions: list[str | None] = []
    for v in [solc_version, *(solc_fallbacks or [])]:
        if v and v not in versions:
            versions.append(v)
    if not versions:
        versions = [None]

    tmp_root = Path(tempfile.gettempdir()) / "slatriage_slither"
    staged = _stage_sol(Path(sol_path), tmp_root)
    tmp_json = tmp_root / f"{staged.name}.slither.json"

    last: dict[str, Any] = {
        "sol_path": str(sol_path),
        "returncode": -1,
        "stdout_tail": "",
        "stderr_tail": "",
        "ok": False,
        "detectors": [],
        "solc": solc_version,
    }

    for ver in versions:
        if ver and not select_solc(ver):
            continue
        try:
            proc, payload, stdout, stderr = _run_once(staged, tmp_json, timeout, ver)
        except FileNotFoundError:
            raise
        except subprocess.TimeoutExpired:
            last.update(
                {
                    "solc": ver,
                    "ok": False,
                    "error": "timeout",
                    "returncode": -1,
                }
            )
            continue

        detectors = normalize_detectors(payload) if payload else []
        analyzed = bool(payload) and (
            isinstance(payload.get("results"), dict) or payload.get("success") is True
        )
        last = {
            "sol_path": str(sol_path),
            "returncode": proc.returncode,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "ok": analyzed or bool(detectors),
            "detectors": detectors,
            "solc": ver,
            "success": None if payload is None else payload.get("success"),
        }
        if last["ok"]:
            wrapper = {
                "success": payload.get("success") if payload else False,
                "error": payload.get("error") if payload else None,
                "results": payload.get("results") if payload else {},
                "detectors": detectors,
                "sol_path": str(sol_path),
                "solc": ver,
            }
            out_json.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
            return last

    return last


def normalize_detectors(slither_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Slither JSON into a list of alert dicts."""
    alerts: list[dict[str, Any]] = []
    results = slither_payload.get("results", {})
    detectors = results.get("detectors", []) if isinstance(results, dict) else []
    for i, d in enumerate(detectors):
        elements = d.get("elements") or []
        lines: list[int] = []
        source = ""
        for el in elements:
            src = el.get("source_mapping") or {}
            lines.extend(src.get("lines") or [])
            source = src.get("filename_relative") or src.get("filename_absolute") or source
        alerts.append(
            {
                "alert_id": f"{d.get('check', 'unknown')}_{i}",
                "detector": d.get("check"),
                "impact": d.get("impact"),
                "confidence": d.get("confidence"),
                "description": d.get("description"),
                "lines": sorted(set(int(x) for x in lines)),
                "source": source,
            }
        )
    return alerts


def make_alert_id(source: str, detector: str, index: int) -> str:
    """Globally unique id: parent_stem + detector + per-file index."""
    p = Path(str(source))
    stem = f"{p.parent.name}_{p.stem}".replace(" ", "_") or "unknown"
    det = detector or "unknown"
    return f"{stem}__{det}_{index}"


def slither_stem(path: Path) -> str:
    name = path.name
    suffix = ".slither.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def snippet_around(sol_path: Path, lines: list[int], ctx: int = 3) -> str:
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if not lines or not text:
        return ""
    lo = max(1, min(lines) - ctx)
    hi = min(len(text), max(lines) + ctx)
    return "\n".join(f"{i}:{text[i - 1]}" for i in range(lo, hi + 1))
