import hashlib
import json

import httpx
import pytest

from ewh.provenance import archive, manifest


def test_archive_writes_sidecar_and_refuses_overwrite(tmp_path):
    target = tmp_path / "spf" / "file.xlsx"
    side = archive.write_raw(target, b"hello", url="https://example.org/f", status=200)

    assert target.read_bytes() == b"hello"
    sidecar = json.loads(archive.sidecar_path(target).read_text())
    assert sidecar == side
    assert sidecar["source_url"] == "https://example.org/f"
    assert sidecar["http_status"] == 200
    assert sidecar["bytes"] == 5
    assert sidecar["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert sidecar["retrieved_at"].endswith("Z")

    with pytest.raises(FileExistsError):
        archive.write_raw(target, b"other", url="https://example.org/f", status=200)
    assert target.read_bytes() == b"hello"


def test_fetch_retries_then_archives(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=b"data")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "x.txt"
    side = archive.fetch("https://example.org/x.txt", dest, client=client, sleep=lambda s: None)
    assert len(calls) == 3
    assert dest.read_bytes() == b"data"
    assert side["sha256"] == hashlib.sha256(b"data").hexdigest()
    assert calls[-1].headers["user-agent"].startswith("expectations-without-hindsight/")


def test_fetch_is_idempotent_and_refuses_errors(tmp_path):
    hits = []

    def ok(request):
        hits.append(1)
        return httpx.Response(200, content=b"a")

    client = httpx.Client(transport=httpx.MockTransport(ok))
    dest = tmp_path / "a.txt"
    archive.fetch("https://example.org/a", dest, client=client)
    archive.fetch("https://example.org/a", dest, client=client)
    assert len(hits) == 1

    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(archive.FetchError):
        archive.fetch("https://example.org/b", tmp_path / "b.txt", client=bad)
    assert not (tmp_path / "b.txt").exists()


def test_manifest_verify_detects_changed_byte(tmp_path):
    archive.write_raw(tmp_path / "a" / "one.bin", b"abc", url="https://e.org/1", status=200)
    archive.write_raw(tmp_path / "two.bin", b"xyz", url="https://e.org/2", status=200)
    m = manifest.build(tmp_path)

    assert set(m["files"]) == {"a/one.bin", "two.bin"}
    assert (tmp_path / "MANIFEST.json").exists()
    assert manifest.verify(tmp_path) == []

    (tmp_path / "two.bin").write_bytes(b"xyZ")
    problems = manifest.verify(tmp_path)
    assert len(problems) == 1 and "two.bin" in problems[0]


def test_manifest_flags_missing_and_unsidecarred(tmp_path):
    archive.write_raw(tmp_path / "one.bin", b"abc", url="https://e.org/1", status=200)
    manifest.build(tmp_path)
    (tmp_path / "stray.bin").write_bytes(b"no sidecar")
    (tmp_path / "one.bin").unlink()
    problems = manifest.verify(tmp_path)
    assert any("missing" in p and "one.bin" in p for p in problems)
    assert any("no sidecar" in p and "stray.bin" in p for p in problems)


def test_manifest_is_canonical(tmp_path):
    archive.write_raw(tmp_path / "b.bin", b"2", url="https://e.org/b", status=200)
    archive.write_raw(tmp_path / "a.bin", b"1", url="https://e.org/a", status=200)
    manifest.build(tmp_path)
    first = (tmp_path / "MANIFEST.json").read_bytes()
    manifest.build(tmp_path)
    assert (tmp_path / "MANIFEST.json").read_bytes() == first
    assert list(json.loads(first)["files"]) == ["a.bin", "b.bin"]


def test_archive_manual_copies_with_manual_sidecar(tmp_path):
    src = tmp_path / "dl" / "pkg.zip"
    src.parent.mkdir()
    src.write_bytes(b"zipdata")
    dest = tmp_path / "raw" / "pkg" / "pkg.zip"
    side = archive.archive_manual(src, dest, source_url="https://doi.org/x", note="by hand")
    assert dest.read_bytes() == b"zipdata" and src.exists()
    assert side["retrieval"] == "manual" and side["http_status"] is None
    assert side["sha256"] == hashlib.sha256(b"zipdata").hexdigest()
    with pytest.raises(FileExistsError):
        archive.archive_manual(src, dest, source_url="https://doi.org/x", note="by hand")
    m = manifest.build(tmp_path / "raw")
    assert m["files"]["pkg/pkg.zip"]["source_url"] == "https://doi.org/x"
