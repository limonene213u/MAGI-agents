# skeleton: base.py
# /home/limonene/ROCm-project/MAGI-agents/base.py
#
# 注意: これはスケルトンコードです。
# Codexはこのファイルを参考に実装を行い、
# 実際のコードを /home/limonene/ROCm-project/MAGI-agents/base.py に配置してください。
# 設計方針・仕様は MAGI-agents_AGENTS_20260220.md を参照。
#
# 【Codexへの参照指示】
# 実装前に以下のドキュメントを必ず参照すること:
#   - OpenRouter APIスキーマ:  tank/docs-ref/Coding_ref/OpenRouter_Docs-API_Reference/
#   - HTTP通信ライブラリ:      tank/docs-ref/Coding_ref/HTTPX-A_next-generation_HTTP_client_for_Python/
# 特にOpenRouterのエンドポイント・ヘッダー・レスポンス形式はtank内のドキュメントを正とし、
# 学習データより優先すること（APIは頻繁に変更されるため）。
#
# 変更履歴:
#   20260220 v1: 初版
#   20260220 v2: Gemini査読対応
#     - 残高確認エンドポイントを /api/v1/credits → /api/v1/auth/key に修正
#     - レスポンスキーを data.limit_remaining に修正
#     - 推奨ヘッダー（HTTP-Referer / X-Title）を追加
#     - タイムアウトを明示的に設定（残高確認:10秒 / Chat:180秒）
#     - 429リトライロジックはTODO（MVP後に追加）
#   20260220 v3: Structured Outputs対応
#     - ask() に response_format 引数を追加（JSON modeサポート）

from __future__ import annotations

import os
import json
import hashlib
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
import requests
from dotenv import load_dotenv


# -----------------------------
# Exceptions
# -----------------------------
class MAGIError(Exception):
    pass

class ConfigError(MAGIError):
    pass

class GuardrailError(MAGIError):
    pass

class OpenRouterError(MAGIError):
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
    model: str
    mode: str  # "manual" | "timer"
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
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d-%H%M%S")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_text(path: Path, text: str) -> None:
    """atomic-ish: tempファイルに書いてからrename"""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_text_truncated(path: Path, max_chars: int) -> str:
    """トークン上限の簡易近似（文字数カット）。tiktoken等は後で追加。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]\n"


# -----------------------------
# Config loader
# -----------------------------
class ConfigLoader:
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
        return AgentConfig(
            name=agent_name,
            model=str(a["model"]),
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
    """
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, timeout_sec: int = 60) -> None:
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None,  # {"type": "json_object"} でJSON mode
    ) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aets-rimoko/MAGI-agents",  # スパム判定回避（特に無料モデル）
            "X-Title": "MAGI-agents",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format  # JSON mode: {"type": "json_object"}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)  # LLM応答待ちを考慮
            # TODO: 429 Too Many Requests の指数バックオフリトライ（MVP後に追加）
        except requests.RequestException as e:
            raise OpenRouterError(f"request failed: {e}") from e

        if resp.status_code >= 400:
            raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise OpenRouterError(f"invalid json response: {e}") from e

    def credits(self) -> Optional[float]:
        """
        残高確認。取得失敗時はNoneを返す（呼び出し元が保守的に判断する）
        エンドポイント: GET /api/v1/auth/key
        レスポンス: {"data": {"limit_remaining": 15.503, "is_free_tier": false}}
        """
        url = f"{self.BASE_URL}/auth/key"  # /credits は存在しない（Gemini査読で修正）
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)  # 残高確認は短めに
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
    """

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

        ensure_dir(self.paths.tank_logs_dir)
        ensure_dir(self.paths.local_cache_dir)
        ensure_dir(self.paths.state_dir)

    def _init_paths(self, repo_root: Path, tank_root: Path) -> RuntimePaths:
        return RuntimePaths(
            repo_root=repo_root,
            tank_root=tank_root,
            tank_logs_dir=tank_root / "MAGI-agents" / "logs",
            local_cache_dir=repo_root / ".cache",
            state_dir=tank_root / "MAGI-agents" / "state",
        )

    # ---- Guardrails ----

    def enforce_limits(self, items: List[Any]) -> List[Any]:
        lim = self.cfg.guardrails.max_items_per_run
        if len(items) > lim:
            self.log_event("warn", f"Too many items ({len(items)}), truncating to {lim}")
            return items[:lim]
        return items

    def ensure_not_processed(self, key: str) -> None:
        """idempotentマーカー。処理済みならGuardrailErrorを送出。"""
        marker = self.paths.state_dir / f"{key}.done"
        if marker.exists():
            raise GuardrailError(f"idempotent: already processed: {key}")
        safe_write_text(marker, f"{now_ts()} processed\n")

    def check_paid_credit(self) -> None:
        if not self.cfg.is_paid:
            return
        thr = self.cfg.guardrails.credit_threshold_usd
        if thr is None:
            raise GuardrailError("paid agent requires credit_threshold_usd in config")
        bal = self.client.credits()
        if bal is None:
            raise GuardrailError("cannot verify OpenRouter credits; refusing to run paid agent")
        if bal < thr:
            raise GuardrailError(f"credits too low: {bal:.2f} < {thr:.2f}")

    # ---- Logging ----

    def log_path(self) -> Path:
        return self.paths.tank_logs_dir / f"{now_ts()}_{self.cfg.name}.md"

    def log_event(self, level: str, msg: str) -> None:
        """tankへの追記ログ（正本）"""
        line = f"- [{level.upper()}] {now_ts()} {msg}\n"
        p = self.log_path()
        if p.exists():
            p.write_text(p.read_text(encoding="utf-8") + line, encoding="utf-8")
        else:
            header = f"# MAGI-agents log: {self.cfg.name}\n\n"
            safe_write_text(p, header + line)

    # ---- OpenRouter helper ----

    def ask(self, system: str, user: str, *, json_mode: bool = False) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if json_mode else None
        resp = self.client.chat(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_output_tokens,
            response_format=response_format,
        )
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            raise OpenRouterError(f"unexpected response schema: {resp}") from e

    # ---- Lifecycle ----

    def run(self) -> None:
        """
        テンプレートメソッド:
        1. 有料エージェントは残高チェック
        2. _run_impl() を実行
        3. 結果をログに記録
        """
        self.log_event("info", f"start: model={self.cfg.model} paid={self.cfg.is_paid}")
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
        raise NotImplementedError("implement in child agent")
