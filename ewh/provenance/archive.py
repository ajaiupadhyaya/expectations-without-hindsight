"""Polite downloads into an append-only raw archive with hash sidecars (CLAUDE.md rules 2, 8).

Adapted from gpu-econ-life `src/gel/collect/http.py` (write_raw, Throttle, get_with_backoff),
with sha256 and byte counts added to the sidecar.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw"
SIDECAR_SUFFIX = ".sidecar.json"
RETRY_STATUSES = {429} | set(range(500, 600))
MIN_INTERVAL_SECONDS = 1.0
TIMEOUT = httpx.Timeout(connect=30, read=120, write=30, pool=30)
USER_AGENT = "expectations-without-hindsight/0.1 (academic research; replication archive)"


class FetchError(RuntimeError):
    """A download finished with a non-200 status after retries."""


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class Throttle:
    """Keep at least `min_interval` seconds between requests to the same host."""

    def __init__(
        self,
        min_interval: float = MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self.last: dict[str, float] = {}

    def wait(self, host: str) -> None:
        if host in self.last:
            remaining = self.last[host] + self.min_interval - self.clock()
            if remaining > 0:
                self.sleep(remaining)
        self.last[host] = self.clock()


_THROTTLE = Throttle()


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)


def get_with_backoff(
    http: httpx.Client,
    url: str,
    *,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET with exponential backoff on 429, 5xx and network errors."""
    for attempt in range(max_attempts):
        last = attempt == max_attempts - 1
        try:
            response = http.get(url, headers={"User-Agent": USER_AGENT})
        except httpx.TransportError:
            if last:
                raise
        else:
            if response.status_code not in RETRY_STATUSES or last:
                return response
        sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")


def write_raw(path: Path, content: bytes, *, url: str, status: int) -> dict:
    """Write a raw file and its sidecar. Refuses to overwrite either (append-only)."""
    side_path = sidecar_path(path)
    if side_path.exists():
        raise FileExistsError(side_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)
    sidecar = {
        "source_url": url,
        "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "http_status": status,
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }
    side_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def archive_manual(src: Path, dest: Path, *, source_url: str, note: str) -> dict:
    """Copy a file someone downloaded by hand into the raw archive, with a sidecar that says
    so. `retrieved_at` is the source file's modification time (UTC); the copy refuses to
    overwrite like every raw write."""
    content = src.read_bytes()
    side_path = sidecar_path(dest)
    if side_path.exists():
        raise FileExistsError(side_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "xb") as f:
        f.write(content)
    mtime = datetime.fromtimestamp(src.stat().st_mtime, UTC)
    sidecar = {
        "source_url": source_url,
        "retrieved_at": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "http_status": None,
        "retrieval": "manual",
        "note": note,
        "archived_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }
    side_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def read_sidecar(path: Path) -> dict:
    return json.loads(sidecar_path(path).read_text())


def fetch(
    url: str,
    dest: Path,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Download `url` into `dest` once. If `dest` is already archived, return its sidecar."""
    if dest.exists() and sidecar_path(dest).exists():
        return read_sidecar(dest)
    own = client is None
    http = client or globals()["client"]()
    try:
        _THROTTLE.wait(httpx.URL(url).host)
        response = get_with_backoff(http, url, sleep=sleep)
    finally:
        if own:
            http.close()
    if response.status_code != 200:
        raise FetchError(f"{url} -> HTTP {response.status_code}")
    return write_raw(dest, response.content, url=str(response.url), status=response.status_code)
