# skeleton: result_analyst.py
# /home/limonene/ROCm-project/MAGI-agents/result_analyst.py
#
# 注意: これはスケルトンコードです。MVP（最小実装）。
# Codexはこのファイルを参考に実装を行い、
# 実際のコードを /home/limonene/ROCm-project/MAGI-agents/result_analyst.py に配置してください。
# 設計方針・仕様は以下を参照:
#   - MAGI-agents_AGENTS_20260220.md（Hard Rules・I/O契約）
#   - MAGI-agents_idempotent_design_20260220.md（idempotentキー設計）
#
# 【Codexへの参照指示】
# 実装前に以下のドキュメントを必ず参照すること:
#   - Pydantic V2:             tank/docs-ref/Coding_ref/Pydantic_V2/
#   - OpenRouter Structured Outputs: tank/docs-ref/Coding_ref/OpenRouter_Docs-API_Reference/
#   - 研究論文（評価基準の確認・著者: 伊藤あきら / Akira Ito）:
#       tank/docs-ref/ 直下の PDF（DeepSeek R1 Japanese Language Adaptation...）
#       tank/docs-ref/ 直下の PDF（日本語LLM評価のための再現可能な検証...）
#   ※ tank/docs-ref/AI_Papers/ は第三者の参照論文なので混同しないこと
# failure_phase の分類基準や key_findings の粒度は、上記りもこ論文の評価フレームワークに合わせること。
#
# 変更履歴:
#   20260220 v1: 初版（にゃんにゃん原案 + 軽微修正）
#   20260220 v2: Pydantic V2 + Structured Outputs対応
#     - ReportJSON モデルを追加（型安全なバリデーション）
#     - _extract_report_json を廃止 → ask(json_mode=True) + Pydantic に置換
#     - report.md は ReportJSON から自動生成（LLM呼び出し1回で完結）
#     - SYSTEM_PROMPT をJSON専用に変更

# /home/limonene/ROCm-project/MAGI-agents/result_analyst.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ValidationError

from base import (
    MAGIAgentBase,
    sha256_text,
    safe_write_text,
    GuardrailError,
    now_ts,
)


# -----------------------------
# Pydantic V2 スキーマ
# -----------------------------
class ReportJSON(BaseModel):
    run_id: str
    status: Literal["success", "failure"]
    failure_phase: Optional[Literal[
        "build", "import", "runtime", "oom", "perf", "numerical", "unknown"
    ]] = None
    key_findings: List[str]
    next_actions: List[str]
    evidence: List[str]

    def to_markdown(self) -> str:
        """ReportJSONからreport.mdを生成（LLM追加呼び出し不要）"""
        lines = [
            f"# 実験結果レポート: {self.run_id}",
            f"\n**ステータス**: {self.status}",
        ]
        if self.failure_phase:
            lines.append(f"**失敗フェーズ**: {self.failure_phase}")

        lines.append("\n## 主要な発見")
        for f in self.key_findings:
            lines.append(f"- {f}")

        lines.append("\n## 次のアクション")
        for a in self.next_actions:
            lines.append(f"- {a}")

        lines.append("\n## 根拠・証跡")
        for e in self.evidence:
            lines.append(f"- {e}")

        return "\n".join(lines)


# -----------------------------
# プロンプト（JSON専用）
# -----------------------------
SYSTEM_PROMPT = """
You are a research results analyst for reproducible LLM evaluation.

Respond ONLY with a valid JSON object. Do not include any explanation or markdown.

Your JSON must have exactly these fields:
- run_id: string
- status: "success" or "failure"
- failure_phase: one of "build"|"import"|"runtime"|"oom"|"perf"|"numerical"|"unknown", or null if success
- key_findings: array of strings (Japanese, concise)
- next_actions: array of strings (Japanese, commands if possible)
- evidence: array of strings (cite log filenames or error messages from input)
"""


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


class ResultAnalystAgent(MAGIAgentBase):

    def __init__(self, repo_root: Path, tank_root: Path, config_path: Path) -> None:
        super().__init__(
            agent_name="result_analyst",
            repo_root=repo_root,
            tank_root=tank_root,
            config_path=config_path,
        )

    def _runs_root(self) -> Path:
        return self.paths.tank_root / "artifacts" / "runs"

    def _find_candidates(self) -> List[Path]:
        runs_root = self._runs_root()
        if not runs_root.exists():
            self.log_event("warn", f"runs root not found: {runs_root}")
            return []

        candidates: List[Path] = []
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "result.json").exists():
                candidates.append(run_dir)
        return candidates

    def _already_reported(self, run_dir: Path) -> bool:
        """生成物が揃っていたらスキップ（後方互換）"""
        return (run_dir / "report.md").exists() and (run_dir / "report.json").exists()

    def _make_idempotent_key(self, run_dir: Path) -> str:
        """idempotentキー = run_dir名 + result.json内容のhash（run_id方式）"""
        payload = (run_dir / "result.json").read_text(encoding="utf-8", errors="replace")
        return sha256_text(str(run_dir) + "\n" + payload)[:16]

    def _build_user_prompt(self, run_dir: Path) -> str:
        """LLMへの入力を構築（小さく・必要十分に）"""
        result = load_json(run_dir / "result.json")

        extra: Dict[str, Any] = {}
        for name in ("env.json", "spec.json", "manifest.yaml"):
            p = run_dir / name
            if not p.exists():
                continue
            if p.suffix == ".yaml":
                extra[name] = p.read_text(encoding="utf-8", errors="replace")[:2000]
            else:
                try:
                    extra[name] = load_json(p)
                except Exception:
                    extra[name] = p.read_text(encoding="utf-8", errors="replace")[:2000]

        run_id = result.get("run_id") or run_dir.name
        prompt = {
            "run_dir": str(run_dir),
            "run_id": run_id,
            "result_json": result,
            "extra": extra,
        }
        return (
            "Analyze the following run artifacts and return JSON only.\n\n"
            "INPUT(JSON):\n"
            + json.dumps(prompt, ensure_ascii=False, indent=2)
        )

    def _run_impl(self) -> None:
        candidates = [d for d in self._find_candidates() if not self._already_reported(d)]
        if not candidates:
            self.log_event("info", "no new runs to analyze")
            return

        candidates = self.enforce_limits(candidates)

        for run_dir in candidates:
            key = self._make_idempotent_key(run_dir)
            try:
                self.ensure_not_processed(key)
            except GuardrailError as e:
                self.log_event("warn", f"skip (idempotent): {run_dir.name} {e}")
                continue

            user_prompt = self._build_user_prompt(run_dir)

            # JSON mode で呼び出し → LLMが型保証済みJSONを返す
            raw = self.ask(SYSTEM_PROMPT, user_prompt, json_mode=True)

            # Pydantic V2 でバリデーション
            try:
                report = ReportJSON.model_validate_json(raw)
            except ValidationError as e:
                self.log_event("warn", f"schema validation failed: {run_dir.name}: {e}")
                # フォールバック: rawをそのままreport.md に保存して続行
                safe_write_text(run_dir / "report.md", f"# Parse Error\n\n```\n{raw}\n```")
                continue

            # report.json と report.md を書き出し（atomic）
            safe_write_text(
                run_dir / "report.json",
                report.model_dump_json(indent=2),
            )
            safe_write_text(run_dir / "report.md", report.to_markdown())

            self.log_event("info", f"wrote report: {run_dir.name}")


if __name__ == "__main__":
    magi_root = Path(__file__).resolve().parent
    repo_root = magi_root.parent   # /home/limonene/ROCm-project
    tank_root = repo_root / "tank"
    config_path = magi_root / "config.yaml"

    agent = ResultAnalystAgent(repo_root=repo_root, tank_root=tank_root, config_path=config_path)
    agent.run()
