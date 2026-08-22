from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.cli import main


def _key(graph_hash: str = "rmsnorm_v1") -> CacheKey:
    return CacheKey(
        graph_hash=graph_hash,
        shapes=((2048, 4096),),
        dtypes=("float16",),
        constants_hash="deadbeef",
        compute_capability="8.9",
        torch_version="2.3.0",
        triton_version="3.0.0",
        cuda_version="12.1",
        library_version="0.1.0",
        template_hash="feedfacecafe",
    )


def _seed(db: Path, n: int = 1) -> None:
    repo = KernelRepository(db)
    for i in range(n):
        key = _key(f"rmsnorm_v{i}")
        repo.put(
            key,
            CachedKernel(
                cache_key=key,
                params={"block_size": 1024},
                kernel_code="def k(): pass",
                benchmark_json={"median_us": 42.0 + i},
                created_at=datetime.now(UTC),
            ),
        )
    repo.close()


def test_cache_list_empty(tmp_path, capsys) -> None:
    db = tmp_path / "cache.db"
    rc = main(["cache", "list", "--db", str(db)])
    assert rc == 0
    assert "空" in capsys.readouterr().out


def test_cache_list_json(tmp_path, capsys) -> None:
    db = tmp_path / "cache.db"
    _seed(db, n=2)
    rc = main(["cache", "list", "--json", "--db", str(db)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert payload[0]["graph_hash"].startswith("rmsnorm_v")
    assert payload[0]["median_us"] in (42.0, 43.0)


def test_cache_list_table(tmp_path, capsys) -> None:
    db = tmp_path / "cache.db"
    _seed(db, n=1)
    rc = main(["cache", "list", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "graph_hash" in out
    assert "2048x4096" in out
    assert "1 件" in out


def test_cache_list_shows_speedup(tmp_path, capsys) -> None:
    db = tmp_path / "cache.db"
    repo = KernelRepository(db)
    key = _key("rmsnorm_speed")
    repo.put(
        key,
        CachedKernel(
            cache_key=key,
            params={"block_size": 1024},
            kernel_code="def k(): pass",
            benchmark_json={"median_us": 50.0},
            baseline_us=100.0,  # -> 2.00x
            created_at=datetime.now(UTC),
        ),
    )
    repo.close()

    rc = main(["cache", "list", "--json", "--db", str(db)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["speedup"] == 2.0

    rc = main(["cache", "list", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "speedup" in out
    assert "2.00x" in out


def test_cache_clear_force(tmp_path, capsys) -> None:
    db = tmp_path / "cache.db"
    _seed(db, n=3)
    rc = main(["cache", "clear", "--force", "--db", str(db)])
    assert rc == 0
    assert "3 件削除" in capsys.readouterr().out
    assert KernelRepository(db).count() == 0


def test_cache_clear_prompt_no(tmp_path, capsys, monkeypatch) -> None:
    db = tmp_path / "cache.db"
    _seed(db, n=2)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    rc = main(["cache", "clear", "--db", str(db)])
    assert rc == 0
    assert "中止" in capsys.readouterr().out
    assert KernelRepository(db).count() == 2


def test_cache_clear_prompt_yes(tmp_path, capsys, monkeypatch) -> None:
    db = tmp_path / "cache.db"
    _seed(db, n=2)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    rc = main(["cache", "clear", "--db", str(db)])
    assert rc == 0
    assert "2 件削除" in capsys.readouterr().out
    assert KernelRepository(db).count() == 0


def test_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cache_list_does_not_create_db(tmp_path, capsys) -> None:
    # read-only の list が mkdir + 空 DB を作らないこと（typo パス対策）。
    db = tmp_path / "no" / "such" / "cache.db"
    rc = main(["cache", "list", "--db", str(db)])
    assert rc == 0
    assert "空" in capsys.readouterr().out
    assert not db.exists()
    assert not db.parent.exists()


def test_cache_clear_does_not_create_db(tmp_path, capsys) -> None:
    db = tmp_path / "missing.db"
    rc = main(["cache", "clear", "--db", str(db)])
    assert rc == 0
    assert not db.exists()


def test_cache_clear_eof_aborts(tmp_path, capsys, monkeypatch) -> None:
    # 非対話 (stdin closed) では EOFError を握って安全側 = 中止。
    db = tmp_path / "cache.db"
    _seed(db, n=2)

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    rc = main(["cache", "clear", "--db", str(db)])
    assert rc == 0
    assert "中止" in capsys.readouterr().out
    assert KernelRepository(db).count() == 2


class TestSQLiteErrorHandling:
    """SQLite 破損・操作失敗時に rc=1 とユーザーフレンドリーなメッセージを返す。"""

    def test_list_operational_error_returns_1(self, tmp_path, capsys, monkeypatch) -> None:
        import sqlite3

        db = tmp_path / "cache.db"
        _seed(db, n=1)

        from forge.cache import repository as repo_mod

        original_cls = repo_mod.KernelRepository

        class _BrokenRepo(original_cls):
            def list_summaries(self):
                raise sqlite3.OperationalError("database disk image is malformed")

        monkeypatch.setattr(repo_mod, "KernelRepository", _BrokenRepo)
        rc = main(["cache", "list", "--db", str(db)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "malformed" in err
        assert "rm" in err

    def test_list_database_error_returns_1(self, tmp_path, capsys, monkeypatch) -> None:
        import sqlite3

        db = tmp_path / "cache.db"
        _seed(db, n=1)

        from forge.cache import repository as repo_mod

        original_cls = repo_mod.KernelRepository

        class _BrokenRepo(original_cls):
            def list_summaries(self):
                raise sqlite3.DatabaseError("file is not a database")

        monkeypatch.setattr(repo_mod, "KernelRepository", _BrokenRepo)
        rc = main(["cache", "list", "--db", str(db)])
        assert rc == 1
        assert "not a database" in capsys.readouterr().err

    def test_clear_operational_error_returns_1(self, tmp_path, capsys, monkeypatch) -> None:
        import sqlite3

        db = tmp_path / "cache.db"
        _seed(db, n=1)

        from forge.cache import repository as repo_mod

        original_cls = repo_mod.KernelRepository

        class _BrokenRepo(original_cls):
            def count(self):
                raise sqlite3.OperationalError("no such table: kernels")

        monkeypatch.setattr(repo_mod, "KernelRepository", _BrokenRepo)
        rc = main(["cache", "clear", "--force", "--db", str(db)])
        assert rc == 1
        assert "no such table" in capsys.readouterr().err
