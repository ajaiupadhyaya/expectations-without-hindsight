import json
import subprocess

import pytest

from ewh.confirm import binding as bd
from ewh.models.ledger import Ledger


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "ewh").mkdir(parents=True)
    (r / "ewh" / "a.py").write_text("x = 1\n")
    (r / "uv.lock").write_text("lock\n")
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.org")
    git(r, "config", "user.name", "t")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    return r


BOUND = ("ewh", "uv.lock")


def test_annotated_tag_passes(repo):
    git(repo, "tag", "-a", "pap-v1", "-m", "freeze")
    t0 = bd.check_binding(repo, "pap-v1", BOUND)
    assert t0.tzinfo is not None


def test_lightweight_tag_refused(repo):
    git(repo, "tag", "pap-v1")
    with pytest.raises(bd.Refused, match="lightweight"):
        bd.check_binding(repo, "pap-v1", BOUND)


def test_missing_tag_refused(repo):
    with pytest.raises(bd.Refused, match="no tag"):
        bd.check_binding(repo, "pap-v1", BOUND)


def test_changed_bound_file_refused(repo):
    git(repo, "tag", "-a", "pap-v1", "-m", "freeze")
    (repo / "ewh" / "a.py").write_text("x = 2\n")
    with pytest.raises(bd.Refused, match="differs"):
        bd.check_binding(repo, "pap-v1", BOUND)


def test_untracked_bound_file_refused(repo):
    git(repo, "tag", "-a", "pap-v1", "-m", "freeze")
    (repo / "ewh" / "new.py").write_text("y = 1\n")
    with pytest.raises(bd.Refused, match="untracked"):
        bd.check_binding(repo, "pap-v1", BOUND)


def test_committed_change_after_tag_refused(repo):
    git(repo, "tag", "-a", "pap-v1", "-m", "freeze")
    (repo / "ewh" / "a.py").write_text("x = 3\n")
    git(repo, "commit", "-qam", "edit")
    with pytest.raises(bd.Refused):
        bd.check_binding(repo, "pap-v1", BOUND)


def test_docs_may_change_after_tag(repo):
    git(repo, "tag", "-a", "pap-v1", "-m", "freeze")
    (repo / "docs").mkdir()
    (repo / "docs" / "deviations.md").write_text("none\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "docs")
    bd.check_binding(repo, "pap-v1", BOUND)


def test_marker_is_single_use(tmp_path):
    bd.create_marker("arm_a", tmp_path)
    with pytest.raises(bd.Refused, match="already been attempted"):
        bd.create_marker("arm_a", tmp_path)
    bd.create_marker("arm_b", tmp_path)  # families are independent


def test_write_once(tmp_path):
    bd.write_once(tmp_path / "r.md", "a")
    with pytest.raises(FileExistsError):
        bd.write_once(tmp_path / "r.md", "b")


def _session(root, shards=2, model="m"):
    (root / model).mkdir(parents=True)
    led = Ledger(root / "ledger.jsonl")
    for s in range(shards):
        f = root / model / f"shard_{s:05d}.parquet"
        f.write_bytes(f"data{s}".encode())
        led.append(event="start", model=model, shard=s)
        led.append(event="done", model=model, shard=s, sha256=bd.sha256_file(f))
    bd.build_generation_manifest(root)
    return led


def test_check_inputs_accepts_complete_session(tmp_path):
    _session(tmp_path)
    bd.check_inputs({"m": 2}, tmp_path)


def test_check_inputs_refuses_missing_extra_or_changed_shards(tmp_path):
    _session(tmp_path)
    with pytest.raises(bd.Refused, match="missing"):
        bd.check_inputs({"m": 3}, tmp_path)
    with pytest.raises(bd.Refused, match="extra"):
        bd.check_inputs({"m": 1}, tmp_path)
    (tmp_path / "m" / "shard_00001.parquet").write_bytes(b"tampered")
    with pytest.raises(bd.Refused, match="does not match"):
        bd.check_inputs({"m": 2}, tmp_path)


def test_check_inputs_refuses_replaced_shard(tmp_path):
    led = _session(tmp_path)
    led.append(event="redo", model="m", shard=0)
    led.append(event="done", model="m", shard=0, sha256="0" * 64)
    with pytest.raises(bd.Refused, match="completed twice"):
        bd.check_inputs({"m": 2}, tmp_path)


def test_check_inputs_refuses_broken_ledger(tmp_path):
    _session(tmp_path)
    p = tmp_path / "ledger.jsonl"
    lines = p.read_text().splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "shard": 9})
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(bd.Refused, match="chain"):
        bd.check_inputs({"m": 2}, tmp_path)
