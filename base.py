"""
MAGI-agents base.py — OpenRouter API 呼び出し共通基盤

設計方針:
  - AGENTS_CANONICAL.md §5 に準拠（暴走防止・idempotent・timer）
  - MAGI-agents/AGENTS.md の Hard Rules を遵守
  - OpenRouter APIスキーマは tank/docs-ref/Coding_ref/OpenRouter_Docs-API_Reference/ を正とする

実装仕様:
  - models: List[str] でフォールバックリストを管理（YAML側で並び順を制御）
  - 1件 → payload["model"]、複数 → payload["models"] を自動切替
  - ask(..., json_mode=True) で response_format: {type: json_object} を渡せる
  - idempotentマーカー: {agent_name}_{key}.done（エージェント間のキー衝突防止）
  - 最新3件保持のマーカークリーンアップ
  - フォールバック追跡ログ: 実際に使われたモデルを warn で記録
  - 残高確認: GET /api/v1/auth/key → data.limit_remaining
  - 推奨ヘッダー: HTTP-Referer / X-Title（無料モデルのブロック回避）
  - タイムアウト: Chat=180秒・残高確認=10秒

変更履歴:
  20260220 v1: スケルトン v5 をベースに本番実装
"""

from __future__ import annotations

import os
import json
import hashlib
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
import requests
from dotenv import load_dotenv

# ログ設定
logger = logging.getLogger("magi-agents")


# -----------------------------
# Exceptions
# -----------------------------
class MAGIError(Exception):
    """MAGI-agents 共通例外基底クラス"""
    pass


class ConfigError(MAGIError):
    """設定エラー"""
    pass


class GuardrailError(MAGIError):
    """ガードレール違反（暴走防止・idempotent等）"""
    pass


class OpenRouterError(MAGIError):
    """OpenRouter API通信エラー"""
    pass


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class Guardrails:
    max_items_per_run: int
    max_tokens_per_item: int
    credit_threshold_usd: Optional[float] = None  # 有料エージェントのみ


@dataclass(frozen=True)
class AgentConfig:
    name: str
    models: List[str]       # フォールバックリスト（並び順=優先順位）
    mode: str               # "manual" | "timer"
    is_paid: bool
    guardrails: Guardrails
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 2048


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    tank_root: Path
    tank_logs_dir: Path    # tank/MAGI-agents/logs  ← ログ正本
    local_cache_dir: Path  # repo_root/.cache       ← 軽微なキャッシュのみ
    state_dir: Path        # tank/MAGI-agents/state ← idempotentマーカー


# -----------------------------
# Utilities
# -----------------------------
def now_ts() -> str:
    """JSTタイムスタンプ"""
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d-%H%M%S")


def sha256_text(s: str) -> str:
    """文字列のSHA256ハッシュ"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    """ディレクトリを再帰的に作成（存在すればスキップ）"""
    p.mkdir(parents=True, exist_ok=True)


def safe_write_text(path: Path, text: str) -> None:
    """atomic-ish: tempファイルに書いてからrename"""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_text_truncated(path: Path, max_chars: int) -> str:
    """トークン上限の簡易近似（文字数カット）"""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]\n"


# -----------------------------
# Config loader
# -----------------------------
class ConfigLoader:
    """config.yaml から AgentConfig を読み込む"""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def load_agent(self, agent_name: str) -> AgentConfig:
        if not self.config_path.exists():
            raise ConfigError(f"config.yaml not found: {self.config_path}")

        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        agents = data.get("agents", {})
        if agent_name not in agents:
            raise ConfigError(f"agent '{agent_name}' not found in config.yaml")

        a = agents[agent_name]
        guard = a.get("guardrails", {})
        g = Guardrails(
            max_items_per_run=int(guard.get("max_items_per_run", 20)),
            max_tokens_per_item=int(guard.get("max_tokens_per_item", 8000)),
            credit_threshold_usd=(
                float(guard["credit_threshold_usd"])
                if "credit_threshold_usd" in guard else None
            ),
        )

        # models: List[str] で管理（YAML側で並び順=優先順位を制御）
        # model（単一）も互換性のためサポート → リスト化
        raw_models = a.get("models")
        if raw_models is None:
            single = a.get("model")
            if single is None:
                raise ConfigError(f"agent '{agent_name}': 'models' or 'model' is required")
            raw_models = [str(single)]
        elif isinstance(raw_models, str):
            raw_models = [raw_models]
        else:
            raw_models = [str(m) for m in raw_models]

        if not raw_models:
            raise ConfigError(f"agent '{agent_name}': models list is empty")

        return AgentConfig(
            name=agent_name,
            models=raw_models,
            mode=str(a.get("mode", "manual")),
            is_paid=bool(a.get("is_paid", False)),
            guardrails=g,
            temperature=float(a.get("temperature", 0.2)),
            top_p=float(a.get("top_p", 0.9)),
            max_output_tokens=int(a.get("max_output_tokens", 2048)),
        )


# -----------------------------
# OpenRouter client
# -----------------------------
class OpenRouterClient:
    """
    Minimal OpenRouter Chat Completions client.
    APIキーはenv: OPENROUTER_API_KEY から読み込む。
    シークレットはログに出力しない。

    APIスキーマ参照:
      tank/docs-ref/Coding_ref/OpenRouter_Docs-API_Reference/
    """
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AETS-ORLAPI-GIKEN/ROCm-MCP",
            "X-Title": "MAGI-agents",
        }

    def chat(
        self,
        models: List[str],
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        OpenRouter Chat Completions API を呼び出す。

        models が1件の場合: payload["model"] にセット
        models が複数の場合: payload["models"] にセット（OpenRouterフォールバック機能）

        Args:
            models: フォールバックリスト（並び順=優先順位）
            messages: チャットメッセージ
            temperature: 温度パラメータ
            top_p: top_pパラメータ
            max_tokens: 最大出力トークン
            response_format: {"type": "json_object"} でJSON mode

        Returns:
            OpenRouterのレスポンスJSON

        Raises:
            OpenRouterError: APIエラー
        """
        url = f"{self.BASE_URL}/chat/completions"

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # 1件→model、複数→models（OpenRouterフォールバック機能）
        if len(models) == 1:
            payload["model"] = models[0]
        else:
            payload["models"] = models

        if response_format is not None:
            payload["response_format"] = response_format

        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=180,  # LLM応答待ちを考慮
            )
        except requests.RequestException as e:
            raise OpenRouterError(f"request failed: {e}") from e

        if resp.status_code >= 400:
            raise OpenRouterError(
                f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise OpenRouterError(f"invalid json response: {e}") from e

    def credits(self) -> Optional[float]:
        """
        残高確認。取得失敗時はNoneを返す（呼び出し元が保守的に判断する）

        エンドポイント: GET /api/v1/auth/key
        レスポンス: {"data": {"limit_remaining": 15.503, ...}}

        docs-ref参照: tank/docs-ref/Coding_ref/OpenRouter_Docs-API_Reference/
                     Get current API key _ OpenRouter _ Documentation.md
        """
        url = f"{self.BASE_URL}/auth/key"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=10,  # 残高確認は短めに
            )
            if resp.status_code >= 400:
                return None
            data = resp.json()
        except Exception:
            return None

        # レスポンス: data.limit_remaining
        try:
            return float(data["data"]["limit_remaining"])
        except (KeyError, TypeError, ValueError):
            return None


# -----------------------------
# Base agent
# -----------------------------
class MAGIAgentBase:
    """
    全エージェント共通基盤。
    各エージェントは _run_impl() だけ実装する。

    機能:
    - ガードレール（上限制御・idempotent・残高チェック）
    - ログ出力（tankへの正本ログ）
    - OpenRouter APIヘルパー（ask()）
    - マーカークリーンアップ（最新3件保持）
    """

    MARKER_KEEP_COUNT = 3  # 最新3件のマーカーを保持

    def __init__(
        self,
        agent_name: str,
        *,
        repo_root: Path,
        tank_root: Path,
        config_path: Path,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("OPENROUTER_API_KEY is missing in environment (.env)")

        self.cfg = ConfigLoader(config_path).load_agent(agent_name)
        self.paths = self._init_paths(repo_root, tank_root)
        self.client = OpenRouterClient(api_key=api_key)
        self._log_file: Optional[Path] = None

        ensure_dir(self.paths.tank_logs_dir)
        ensure_dir(self.paths.local_cache_dir)
        ensure_dir(self.paths.state_dir)

    def _init_paths(self, repo_root: Path, tank_root: Path) -> RuntimePaths:
        return RuntimePaths(
            repo_root=repo_root,
            tank_root=tank_root,
            tank_logs_dir=tank_root / "MAGI-agents" / "logs",
            local_cache_dir=repo_root / "MAGI-agents" / ".cache",
            state_dir=tank_root / "MAGI-agents" / "state",
        )

    # ---- Guardrails ----

    def enforce_limits(self, items: List[Any]) -> List[Any]:
        """処理対象数の上限制御。超過時はwarnログ+上限分のみ返す。"""
        lim = self.cfg.guardrails.max_items_per_run
        if len(items) > lim:
            self.log_event("warn", f"Too many items ({len(items)}), truncating to {lim}")
            return items[:lim]
        return items

    def ensure_not_processed(self, key: str) -> None:
        """
        idempotentマーカー。
        マーカー名: {agent_name}_{key}.done（エージェント間のキー衝突防止）
        処理済みならGuardrailErrorを送出。
        """
        marker_name = f"{self.cfg.name}_{key}.done"
        marker = self.paths.state_dir / marker_name
        if marker.exists():
            raise GuardrailError(f"idempotent: already processed: {marker_name}")
        safe_write_text(marker, f"{now_ts()} processed\n")
        self._cleanup_markers()

    def _cleanup_markers(self) -> None:
        """最新N件保持のマーカークリーンアップ"""
        prefix = f"{self.cfg.name}_"
        markers = sorted(
            [
                p for p in self.paths.state_dir.iterdir()
                if p.name.startswith(prefix) and p.name.endswith(".done")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in markers[self.MARKER_KEEP_COUNT:]:
            try:
                old.unlink()
                logger.debug("cleaned up old marker: %s", old.name)
            except OSError:
                pass

    def check_paid_credit(self) -> None:
        """
        有料エージェントの残高チェック。
        取得できない場合・スキーマが一致しない場合は保守的に停止。
        """
        if not self.cfg.is_paid:
            return
        thr = self.cfg.guardrails.credit_threshold_usd
        if thr is None:
            raise GuardrailError("paid agent requires credit_threshold_usd in config")
        bal = self.client.credits()
        if bal is None:
            self.log_event("error", "cannot verify OpenRouter credits; refusing to run paid agent")
            raise GuardrailError(
                "cannot verify OpenRouter credits (API returned None); "
                "refusing to run paid agent (conservative stop)"
            )
        if bal < thr:
            self.log_event("error", f"credits too low: {bal:.2f} < threshold {thr:.2f}")
            raise GuardrailError(f"credits too low: {bal:.2f} < {thr:.2f}")
        self.log_event("info", f"credit check passed: {bal:.2f} >= {thr:.2f}")

    # ---- Logging ----

    def _get_log_path(self) -> Path:
        """1実行1ログファイル"""
        if self._log_file is None:
            self._log_file = self.paths.tank_logs_dir / f"{now_ts()}_{self.cfg.name}.md"
        return self._log_file

    def log_event(self, level: str, msg: str) -> None:
        """tankへの追記ログ（正本）"""
        ts = now_ts()
        line = f"- [{level.upper()}] {ts} {msg}\n"
        p = self._get_log_path()
        if p.exists():
            with p.open("a", encoding="utf-8") as f:
                f.write(line)
        else:
            header = f"# MAGI-agents log: {self.cfg.name}\n\n"
            safe_write_text(p, header + line)

        # stdoutにも出力（journalctl等で確認可能）
        log_func = getattr(logger, level.lower(), logger.info)
        log_func("[%s] %s", self.cfg.name, msg)

    # ---- OpenRouter helper ----

    def ask(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """
        OpenRouter にチャットリクエストを送信。

        Args:
            system: システムプロンプト
            user: ユーザープロンプト
            json_mode: True の場合 response_format: {type: json_object} をセット

        Returns:
            LLMの返答テキスト

        Raises:
            OpenRouterError: レスポンスが不正な場合
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if json_mode else None

        resp = self.client.chat(
            models=self.cfg.models,
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_output_tokens,
            response_format=response_format,
        )

        # フォールバック追跡: 実際に使われたモデルを記録
        actual_model = resp.get("model", "unknown")
        requested_first = self.cfg.models[0] if self.cfg.models else "unknown"
        if actual_model != requested_first:
            self.log_event(
                "warn",
                f"fallback occurred: requested={requested_first} → actual={actual_model}",
            )
        else:
            self.log_event("info", f"model used: {actual_model}")

        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise OpenRouterError(f"unexpected response schema: {resp}") from e

    # ---- Lifecycle ----

    def run(self) -> None:
        """
        テンプレートメソッド:
        1. 有料エージェントは残高チェック
        2. _run_impl() を実行
        3. 結果をログに記録
        """
        self.log_event(
            "info",
            f"start: models={self.cfg.models} paid={self.cfg.is_paid} mode={self.cfg.mode}",
        )
        try:
            self.check_paid_credit()
            self._run_impl()
            self.log_event("info", "done")
        except GuardrailError as e:
            self.log_event("error", f"guardrail: {e}")
            raise
        except Exception as e:
            self.log_event("error", f"exception: {type(e).__name__}: {e}")
            raise

    def _run_impl(self) -> None:
        """各エージェントが実装する。"""
        raise NotImplementedError("implement in child agent")
