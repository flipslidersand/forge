"""DiscordNotifier のユニットテスト（CPU / ネットワーク不要）。

webhook 未設定時はエラーにならず False を返すこと、設定時は _send_webhook を
経由して HTTP 204 で True を返すこと、通信失敗を握って False を返すことを検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from forge.notifiers.discord import DiscordNotifier


def _configured() -> DiscordNotifier:
    with patch.dict(
        "os.environ",
        {
            "DISCORD_WEBHOOK_COMPLETION": "https://discord.test/completion",
            "DISCORD_WEBHOOK_ERRORS": "https://discord.test/errors",
        },
    ):
        return DiscordNotifier()


class TestUnconfigured:
    """webhook 未設定なら全メソッドが False（例外を出さない）。"""

    def test_no_env_reads_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.completion_webhook is None
        assert n.error_webhook is None

    def test_send_cache_hit_returns_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.send_cache_hit("rmsnorm") is False

    def test_send_optimization_complete_returns_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.send_optimization_complete("rmsnorm", 1.0, 10, 2.0) is False

    def test_send_optimization_error_returns_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.send_optimization_error("rmsnorm", "boom", "ERR") is False


class TestConfiguredSuccess:
    """webhook 設定 + HTTP 204 → True。urlopen をモックしてネットワークを遮断。"""

    def _mock_204(self):
        resp = MagicMock()
        resp.status = 204
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        return ctx

    def test_cache_hit_success(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", return_value=self._mock_204()):
            assert n.send_cache_hit("rmsnorm") is True

    def test_optimization_complete_success(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", return_value=self._mock_204()):
            assert n.send_optimization_complete("softmax", 0.5, 20, 3.0) is True

    def test_optimization_error_success(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", return_value=self._mock_204()):
            assert n.send_optimization_error("gelu", "boom", "OPT_FAIL") is True


class TestConfiguredFailure:
    """通信失敗・非 204 応答でも例外を伝播させず False を返す。"""

    def test_non_204_returns_false(self) -> None:
        n = _configured()
        resp = MagicMock()
        resp.status = 500
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        with patch("forge.notifiers.discord.urlopen", return_value=ctx):
            assert n.send_cache_hit("rmsnorm") is False

    def test_urlopen_raises_returns_false(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", side_effect=OSError("network down")):
            assert n.send_optimization_complete("rmsnorm", 1.0, 5, 1.0) is False


class TestUnexpectedExceptions:
    """URLError 以外の例外でも伝播させず False を返すことを検証。

    discord.py の2層構造を確認する:
    - 外層 except Exception (:70/:112/:137): embed 構築中の予期しない例外
    - 内層 except Exception (:170):         urlopen が URLError 以外で失敗した場合
    """

    def test_send_webhook_runtime_error_returns_false(self) -> None:
        """_send_webhook 内で URLError 以外の例外 → False（内層 except Exception）。"""
        n = _configured()
        with patch(
            "forge.notifiers.discord.urlopen", side_effect=RuntimeError("ssl context broken")
        ):
            assert n.send_optimization_complete("rmsnorm", 1.0, 5, 2.0) is False

    def test_send_webhook_attribute_error_returns_false(self) -> None:
        """urlopen が AttributeError を投げても False を返す。"""
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", side_effect=AttributeError("mock attr")):
            assert n.send_cache_hit("rmsnorm") is False

    def test_send_webhook_value_error_returns_false(self) -> None:
        """urlopen が ValueError を投げても False を返す。"""
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", side_effect=ValueError("unexpected value")):
            assert n.send_optimization_error("rmsnorm", "msg", "ERR") is False

    def test_embed_construction_error_returns_false(self) -> None:
        """embed 構築中（datetime.now）の例外を外層 except Exception が握る。"""
        n = _configured()
        with patch("forge.notifiers.discord.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("clock unavailable")
            assert n.send_optimization_complete("rmsnorm", 1.0, 5, 2.0) is False

    def test_send_optimization_error_embed_construction_error_returns_false(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("clock unavailable")
            assert n.send_optimization_error("rmsnorm", "fail", "TYPE") is False

    def test_send_cache_hit_embed_construction_error_returns_false(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("clock unavailable")
            assert n.send_cache_hit("rmsnorm") is False
