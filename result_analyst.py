"""
MAGI-agents result_analyst.py — 実験結果整理エージェント

設計方針:
  - AGENTS.md の result_analyst I/O契約に準拠
  - 入力: tank/artifacts/runs/*/result.json
  - 出力: tank/artifacts/runs/*/report.json（Pydantic V2でバリデーション）
           tank/artifacts/runs/*/report.md（report.jsonから自動生成）
  - idempotent: run_id方式（sha256_text(run_dir + result.json内容)[:16]）
  - ask(json_mode=True) でJSON mode指定 → ReportJSON.model_validate_json() で検証
  - バリデーション失敗時: parse errorをreport.mdに保存して続行（全体を止めない）
  - 起動: 手動CLI → timerへ移行

failure_phase分類:
  AGENTS_CANONICAL.md §4.3 Failure Taxonomy に準拠
  りもこ論文の評価フレームワークに合わせた分類基準:
  - build: ビルドエラー
  - import: インポートエラー
  - runtime: 実行時エラー
  - oom: メモリ不足
  - perf: 性能劣化
  - numerical: 数値的不整合
  - unknown: 分類不能

変更履歴:
  20260220 v1: スケルトン v2 をベースに本番実装
"""

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
    """
    実験結果レポートのスキーマ。
    Pydantic V2 で型安全にバリデーション。

    failure_phase: AGENTS_CANONICAL.md §4.3 Failure Taxonomy 準拠
    key_findings / next_actions: りもこ論文の評価フレームワークに合わせた粒度
    evidence: ログファイル名・エラーメッセージ等の根拠
    """
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

        lines.append("")  # 末尾改行
        return "\n".join(lines)


# -----------------------------
# プロンプト（JSON専用）
# -----------------------------
SYSTEM_PROMPT = """\
You are a research results analyst for reproducible LLM evaluation.

Respond ONLY with a valid JSON object. Do not include any explanation or markdown.

Your JSON must have exactly these fields:
- run_id: string (use the run_id from the input)
- status: "success" or "failure"
- failure_phase: one of "build"|"import"|"runtime"|"oom"|"perf"|"numerical"|"unknown", or null if success
- key_findings: array of strings (Japanese, concise, aligned with reproducibility evaluation criteria)
- next_actions: array of strings (Japanese, actionable commands/steps if possible)
- evidence: array of strings (cite log filenames, error messages, or metric values from the input)

Failure phase classification criteria:
- build: Build errors (compilation, dependency resolution)
- import: Import errors (missing modules, version conflicts)
- runtime: Runtime errors (exceptions during execution)
- oom: Out of memory (GPU/CPU memory exhaustion)
- perf: Performance degradation (unexpectedly slow, below baseline)
- numerical: Numerical inconsistency (NaN, precision loss, score anomalies)
- unknown: Cannot classify into above categories
"""


def load_json(p: Path) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
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
        """result.jsonが存在するrunディレクトリを列挙"""
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
        """生成物が揃っていたらスキップ（後方互換チェック）"""
        return (run_dir / "report.md").exists() and (run_dir / "report.json").exists()

    def _make_idempotent_key(self, run_dir: Path) -> str:
        """
        idempotentキー = run_dir名 + result.json内容のhash（run_id方式）
        マーカー名: {agent_name}_{key}.done
        """
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

        # lab_notebook の既存実験ログを参照コンテキストとして追加
        lab_context = ""
        lab_log = self.paths.tank_root / "lab_notebook" / "experiment_log.md"
        if lab_log.exists():
            try:
                lab_text = lab_log.read_text(encoding="utf-8", errors="replace")
                # トークン節約: 最大3000文字に制限
                lab_context = lab_text[:3000]
            except Exception:
                pass

        prompt = {
            "run_dir": str(run_dir),
            "run_id": run_id,
            "result_json": result,
            "extra": extra,
        }
        if lab_context:
            prompt["lab_notebook_context"] = lab_context

        return (
            "Analyze the following run artifacts and return JSON only.\n"
            "If lab_notebook_context is provided, use it to compare with previous runs.\n\n"
            "INPUT(JSON):\n"
            + json.dumps(prompt, ensure_ascii=False, indent=2)
        )

    def _run_impl(self) -> None:
        candidates = [d for d in self._find_candidates() if not self._already_reported(d)]
        if not candidates:
            self.log_event("info", "no new runs to analyze")
            return

        candidates = self.enforce_limits(candidates)

        processed = 0
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
                self.log_event(
                    "warn",
                    f"schema validation failed: {run_dir.name}: {e}",
                )
                # フォールバック: rawをそのままreport.mdに保存して続行（全体を止めない）
                safe_write_text(
                    run_dir / "report.md",
                    f"# Parse Error — {run_dir.name}\n\n"
                    f"Pydantic validation failed. Raw LLM output:\n\n```\n{raw}\n```\n",
                )
                continue

            # report.json と report.md を書き出し（atomic）
            safe_write_text(
                run_dir / "report.json",
                report.model_dump_json(indent=2),
            )
            safe_write_text(run_dir / "report.md", report.to_markdown())

            self.log_event("info", f"wrote report: {run_dir.name}")
            processed += 1

        self.log_event("info", f"analyzed {processed}/{len(candidates)} runs")


if __name__ == "__main__":
    magi_root = Path(__file__).resolve().parent
    repo_root = magi_root.parent   # /home/limonene/ROCm-project
    tank_root = repo_root / "tank"
    config_path = magi_root / "config.yaml"

    agent = ResultAnalystAgent(
        repo_root=repo_root,
        tank_root=tank_root,
        config_path=config_path,
    )
    agent.run()
