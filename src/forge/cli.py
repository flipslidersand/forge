from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from forge import __version__
from forge.cache.repository import DEFAULT_DB_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge — automatic GPU kernel optimizer",
    )
    parser.add_argument("--version", action="version", version=f"forge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache", help="キャッシュ済みカーネルの管理")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)

    list_p = cache_sub.add_parser("list", help="キャッシュ済みカーネル一覧")
    list_p.add_argument("--json", action="store_true", help="JSON で出力")
    list_p.add_argument("--db", default=DEFAULT_DB_PATH, help="キャッシュ DB のパス")
    list_p.set_defaults(func=_cmd_list)

    clear_p = cache_sub.add_parser("clear", help="全キャッシュ削除")
    clear_p.add_argument("--force", action="store_true", help="確認プロンプトなしで削除")
    clear_p.add_argument("--db", default=DEFAULT_DB_PATH, help="キャッシュ DB のパス")
    clear_p.set_defaults(func=_cmd_clear)

    return parser


def _db_missing(db: str) -> bool:
    """DB ファイルが存在しない場合 True。

    KernelRepository.__init__ は mkdir + 空 DB 作成の副作用を持つため、
    read 系コマンドは開く前に存在確認する（typo パスへの空 DB 散乱防止）。
    """
    return not Path(db).expanduser().exists()


def _cmd_list(args: argparse.Namespace) -> int:
    import sqlite3

    from forge.cache.repository import KernelRepository

    if _db_missing(args.db):
        print(json.dumps([]) if args.json else "キャッシュは空です。(DB 未作成)")
        return 0

    repo = KernelRepository(args.db)
    try:
        summaries = repo.list_summaries()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        print(
            f"エラー: キャッシュ DB の読み取りに失敗しました: {e}\n"
            f"対処法: rm {args.db}  (次回実行時にキャッシュが再構築されます)",
            file=sys.stderr,
        )
        return 1
    finally:
        repo.close()

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "cache_key_hash": s.cache_key_hash,
                        "graph_hash": s.cache_key.graph_hash,
                        "shapes": [list(sh) for sh in s.cache_key.shapes],
                        "dtypes": list(s.cache_key.dtypes),
                        "median_us": s.median_us,
                        "speedup": s.speedup,
                        "created_at": s.created_at,
                    }
                    for s in summaries
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not summaries:
        print("キャッシュは空です。")
        return 0

    rows = [
        (
            s.cache_key_hash[:12],
            s.cache_key.graph_hash,
            ", ".join("x".join(str(d) for d in shape) for shape in s.cache_key.shapes) or "-",
            ",".join(s.cache_key.dtypes) or "-",
            f"{s.median_us:.1f}" if s.median_us is not None else "-",
            f"{s.speedup:.2f}x" if s.speedup is not None else "-",
            s.created_at[:19],
        )
        for s in summaries
    ]
    headers = ("hash", "graph_hash", "shapes", "dtypes", "median_us", "speedup", "created_at")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for r in rows:
        print(fmt.format(*r))
    print(f"\n{len(summaries)} 件")
    return 0


def _confirm(prompt: str) -> bool:
    """y/N 確認。非対話 (EOF) と Ctrl-C は安全側 = No に倒す。"""
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _cmd_clear(args: argparse.Namespace) -> int:
    import sqlite3

    from forge.cache.repository import KernelRepository

    if _db_missing(args.db):
        print("キャッシュは空です。(DB 未作成)")
        return 0

    repo = KernelRepository(args.db)
    try:
        n = repo.count()
        if n == 0:
            print("キャッシュは空です。")
            return 0
        if not args.force and not _confirm(
            f"{n} 件のキャッシュを削除します。よろしいですか? [y/N] "
        ):
            print("中止しました。")
            return 0
        deleted = repo.clear()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        print(
            f"エラー: キャッシュ DB の操作に失敗しました: {e}\n"
            f"対処法: rm {args.db}  (次回実行時にキャッシュが再構築されます)",
            file=sys.stderr,
        )
        return 1
    finally:
        repo.close()
    print(f"{deleted} 件削除しました。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
