"""
MAGI-agents result_analyst.py — 実験結果整理エージェント

設計方針:
  - AGENTS.md の result_analyst I/O契約に準拠
  - 入力:  tank/artifacts/runs/*/result.json
           tank/artifacts/*/result.json（直下も対象、20990101-* 除外）
  - 出力:  tank/lab_notebook/reports/{run_id}/report.json（Pydantic V2でバリデーション）
           tank/lab_notebook/reports/{run_id}/report.md（report.jsonから自動生成）
           tank/lab_notebook/notes/NotebookLM/{run_id}/notebooklm_bundle-{run_id}.md
           tank/lab_notebook/notes/NotebookLM/runs_master_table.md（全run一覧テーブル）
           tank/lab_notebook/notes/NotebookLM/reports_master_table.md（全レポート一覧テーブル）
           tank/lab_notebook/notes/NotebookLM/raw_data_master_table.md（全run×タスク生データテーブル）
  - artifacts/ には一切書き込まない（runner の領域、不変）
  - 後方互換: artifacts/ 内の旧 report.json/md は読み取りのみ許可（再分析スキップ）
  - idempotent: run_id方式（sha256_text(run_dir + result.json内容)[:16]）
  - ask(json_mode=True) でJSON mode指定 → ReportJSON.model_validate_json() で検証
  - バリデーション失敗時: parse errorをreport.mdに保存して続行（全体を止めない）
  - bundle生成: LLM呼び出し不要（result.json + report.json の純粋な整形）
  - master table: 全 run 処理後に常に全件再生成（idempotency marker 不要）
  - 起動: 手動CLI → timerへ移行
  - --dry-run: ファイル書き出しなしで処理対象を表示
  - raw_data_master_table.md: 全run×全タスクの生データ（details）を展開した行構造テーブル

スキャン優先順位:
  1. artifacts/runs/ を先にスキャン（既存 idempotent キーを保持）
  2. artifacts/ 直下を追加スキャン（runs/ に無いもののみ追加）
  20990101-* はテストダミー、常にスキップ

failure_phase分類:
  Research_Governance.md §4.3 Failure Taxonomy に準拠
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
  20260301 v2: artifacts/ 直下もスキャン対象に追加、NotebookLM bundle 生成を追加
  20260301 v3: runs_master_table / reports_master_table 自動生成を追加
  20260301 v4: raw_data_master_table 追加（details を run×task 行に展開、LLM不要）
"""

from __future__ import annotations

import json
import re
import sys
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
# 定数
# -----------------------------
# 有効な run ディレクトリ名: YYYYMMDD-HHMMSS-8hexchars
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")

# テストダミー run のプレフィックス（常にスキップ）
_TEST_PREFIX = "20990101-"


# -----------------------------
# Pydantic V2 スキーマ
# -----------------------------
class ReportJSON(BaseModel):
    """
    実験結果レポートのスキーマ。
    Pydantic V2 で型安全にバリデーション。

    failure_phase: Research_Governance.md §4.3 Failure Taxonomy 準拠
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
# Bundle 生成（LLM不要・純整形）
# -----------------------------
def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def generate_bundle_md(
    result: Dict[str, Any],
    report: Optional[ReportJSON],
    run_id: str,
) -> str:
    """
    result.json + report.json から人間可読な NotebookLM bundle Markdown を生成。
    LLM呼び出し不要（純粋な整形処理）。
    report が None の場合は LLM 分析セクションを省略する。
    """
    lines: List[str] = [f"# 実験 Run Bundle: {run_id}", ""]

    # --- 実験設定 ---
    spec = result.get("spec", {})
    lines += [
        "## 実験設定",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        f"| モデル | {spec.get('model_id', '—')} |",
        f"| 量子化 | {spec.get('quantization', '—')} |",
        f"| データセット | {spec.get('dataset', '—')} |",
        "",
    ]

    # --- 実行環境（新旧フォーマット両対応）---
    env = result.get("env", {})
    os_info = env.get("os", {})
    gpu_info = env.get("gpu", {})
    driver = env.get("driver", {})
    cp = result.get("control_plane", {})

    # os: 新フォーマット=dict, 旧フォーマット=str
    if isinstance(os_info, dict):
        os_str = (os_info.get("name", "") + " " + os_info.get("version", "")).strip() or "—"
    else:
        os_str = str(os_info) if os_info else "—"

    # backend: 新=env.backend, 旧=env.inference_backend
    backend_str = env.get("backend") or env.get("inference_backend") or "—"

    # gpu: 新フォーマット=dict, 旧フォーマット=str
    if isinstance(gpu_info, dict):
        gpu_str = gpu_info.get("name", "—")
        if gpu_info.get("vram"):
            gpu_str += f" ({gpu_info['vram']})"
    else:
        gpu_str = str(gpu_info) if gpu_info else None

    lines += [
        "## 実行環境",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        f"| ノード | {env.get('hostname', '—')} |",
        f"| OS | {os_str} |",
        f"| バックエンド | {backend_str} |",
    ]
    if gpu_str:
        lines.append(f"| GPU | {gpu_str} |")
    # ROCm バージョン: 新=driver.rocm, 旧=なし
    if isinstance(driver, dict) and driver.get("rocm"):
        lines.append(f"| ROCm | {driver['rocm']} |")
    if cp:
        cp_str = cp.get("hostname", "—")
        if cp.get("runner_version"):
            cp_str += f" (runner v{cp['runner_version']})"
        lines.append(f"| コントロールプレーン | {cp_str} |")
    lines.append("")

    # --- 結果サマリ ---
    metrics = result.get("metrics", {})
    total = metrics.get("total", 0)
    passed = metrics.get("passed", 0)
    failed = metrics.get("failed", 0)
    status_val = result.get("status", "")
    status_icon = "✅" if status_val == "success" or failed == 0 else "❌"

    lines += [
        "## 結果サマリ",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| ステータス | {status_icon} {status_val or '—'} |",
        f"| 合計タスク | {total} |",
        f"| passed | {passed} / {total} ({_pct(passed / total) if total else '—'}) |",
    ]
    if "json_parse_rate" in metrics:
        lines.append(f"| JSON解析成功率 | {_pct(metrics['json_parse_rate'])} |")
    if "constraint_pass_rate" in metrics:
        lines.append(f"| 制約充足率 | {_pct(metrics['constraint_pass_rate'])} |")
    lines.append("")

    intent = result.get("deterministic_intent", False)
    effective = result.get("deterministic_effective", False)
    if intent and not effective:
        lines += [
            "> [!WARNING]",
            "> **再現性の懸念あり**",
            "> `deterministic_intent=true` でしたが、実行時の判定 `deterministic_effective` は `false` でした。設定や環境のseed対応を確認してください。",
            ""
        ]

    contract_summary = metrics.get("output_contract_summary", {})
    if contract_summary:
        lines += [
            "### Output Contract サマリ",
            "",
            "| contract | 件数 |",
            "|---|---|",
        ]
        for k, v in contract_summary.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")


    # --- タスク別詳細 ---
    details = metrics.get("details", [])
    if details:
        lines += [
            "## タスク別詳細",
            "",
            "| task_id | 制約通過 | parse_error |",
            "|---|---|---|",
        ]
        for d in details:
            icon = "✅" if d.get("constraint_pass") else "❌"
            parse_err = d.get("parse_error") or "—"
            lines.append(f"| {d.get('item_id', '—')} | {icon} | {parse_err} |")
        lines.append("")

    # --- 制約一覧 ---
    tasks = spec.get("tasks", [])
    if tasks:
        lines += [
            "## 制約一覧",
            "",
            "| task_id | 制約 | 設定値 |",
            "|---|---|---|",
        ]
        for t in tasks:
            for k, v in t.get("constraints", {}).items():
                lines.append(f"| {t.get('task_id', '—')} | {k} | {v} |")
        lines.append("")

    # --- LLM分析結果（report.jsonがあれば） ---
    if report is not None:
        lines += ["## 主要な発見（result_analyst より）", ""]
        for f in report.key_findings:
            lines.append(f"- {f}")
        lines += ["", "## 次のアクション", ""]
        for a in report.next_actions:
            lines.append(f"- {a}")
        lines += ["", "## 根拠・証跡", ""]
        for e in report.evidence:
            lines.append(f"- {e}")
        lines.append("")

    # --- NotebookLM アップロード状態 ---
    lines += [
        "## NotebookLM アップロード状態",
        "",
        "- [ ] 未アップロード",
        "",
        f"<!-- generated: {now_ts()} -->",
        "",
    ]

    return "\n".join(lines)


# -----------------------------
# Master Table 生成（LLM不要・純整形）
# -----------------------------

def generate_runs_master_table_md(all_dirs: List[Path]) -> str:
    """
    全 run の一覧テーブル Markdown を生成。
    result.json から環境・スコア情報を抽出する（LLM不要）。
    常に全件再生成する（idempotency marker 不要）。
    """
    lines: List[str] = [
        "# Runs Master Table",
        "",
        f"全 {len(all_dirs)} run の一覧。result_analyst が自動生成（`{now_ts()}`）。",
        "",
        "| 日付 | run_id | ノード | モデル | データセット | 量子化 | 合計 | Passed | JSON解析率 | 制約充足率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for run_dir in sorted(all_dirs, key=lambda d: d.name):
        run_id = run_dir.name
        date_str = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"

        try:
            result = load_json(run_dir / "result.json")
        except Exception:
            lines.append(f"| {date_str} | {run_id} | — | — | — | — | — | — | — | — |")
            continue

        spec = result.get("spec", {})
        env = result.get("env", {})

        # ノード: spec.compute_node があればそちら優先、なければ env.hostname
        node = spec.get("compute_node") or env.get("hostname", "—")

        model = spec.get("model_id", "—")
        dataset = spec.get("dataset", "—")
        quant = spec.get("quantization", "—")

        metrics = result.get("metrics", {})
        total = metrics.get("total", "—")
        passed = metrics.get("passed", "—")
        passed_str = f"{passed}/{total}" if isinstance(passed, int) and isinstance(total, int) else "—"
        json_rate = _pct(metrics["json_parse_rate"]) if "json_parse_rate" in metrics else "—"
        cp_rate = _pct(metrics["constraint_pass_rate"]) if "constraint_pass_rate" in metrics else "—"

        lines.append(
            f"| {date_str} | {run_id} | {node} | {model} | {dataset} | {quant}"
            f" | {total} | {passed_str} | {json_rate} | {cp_rate} |"
        )

    lines += [
        "",
        f"<!-- generated: {now_ts()} runs: {len(all_dirs)} -->",
        "",
    ]
    return "\n".join(lines)


def generate_reports_master_table_md(
    all_dirs: List[Path],
    load_report_fn: Any,
) -> str:
    """
    全 run の LLM 分析結果一覧テーブル Markdown を生成。
    report.json が存在しない run は「未分析」と表示する（LLM不要）。
    常に全件再生成する（idempotency marker 不要）。
    """
    lines: List[str] = [
        "# Reports Master Table",
        "",
        f"全 {len(all_dirs)} run の分析結果一覧。result_analyst が自動生成（`{now_ts()}`）。",
        "",
        "| 日付 | run_id | ステータス | 失敗フェーズ | 主要な発見（1件目） | 次のアクション（1件目） |",
        "|---|---|---|---|---|---|",
    ]

    for run_dir in sorted(all_dirs, key=lambda d: d.name):
        run_id = run_dir.name
        date_str = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"

        report: Optional[ReportJSON] = load_report_fn(run_dir)
        if report is None:
            lines.append(f"| {date_str} | {run_id} | 未分析 | — | — | — |")
            continue

        status_icon = "✅" if report.status == "success" else "❌"
        status_str = f"{status_icon} {report.status}"
        phase = report.failure_phase or "—"
        finding = report.key_findings[0][:60] + "…" if report.key_findings else "—"
        action = report.next_actions[0][:60] + "…" if report.next_actions else "—"

        # | パイプ文字はテーブル崩れ防止のため置換
        finding = finding.replace("|", "｜")
        action = action.replace("|", "｜")

        lines.append(f"| {date_str} | {run_id} | {status_str} | {phase} | {finding} | {action} |")

    lines += [
        "",
        f"<!-- generated: {now_ts()} runs: {len(all_dirs)} -->",
        "",
    ]
    return "\n".join(lines)


def generate_raw_data_master_table_md(all_dirs: List[Path]) -> str:
    """
    全 run × 全タスクの生データを展開した行構造テーブルを生成（LLM不要）。
    result.json の metrics.details を1行1タスクで展開する。
    details が空の run は run 単位で1行だけ出力する（task_id = —）。
    常に全件再生成する（idempotency marker 不要）。
    """
    lines: List[str] = [
        "# Raw Data Master Table",
        "",
        f"全 {len(all_dirs)} run のタスク別生データ。result_analyst が自動生成（`{now_ts()}`）。",
        "",
        "| 日付 | run_id | task_id | constraint_pass | parse_error | json_parse_rate | constraint_pass_rate |",
        "|---|---|---|---|---|---|---|",
    ]

    for run_dir in sorted(all_dirs, key=lambda d: d.name):
        run_id = run_dir.name
        date_str = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"

        try:
            result = load_json(run_dir / "result.json")
        except Exception:
            lines.append(f"| {date_str} | {run_id} | — | — | — | — | — |")
            continue

        metrics = result.get("metrics", {})
        json_rate = _pct(metrics["json_parse_rate"]) if "json_parse_rate" in metrics else "—"
        cp_rate = _pct(metrics["constraint_pass_rate"]) if "constraint_pass_rate" in metrics else "—"
        details = metrics.get("details", [])

        if not details:
            lines.append(f"| {date_str} | {run_id} | — | — | — | {json_rate} | {cp_rate} |")
            continue

        for d in details:
            task_id = d.get("item_id", "—")
            cp_icon = "✅" if d.get("constraint_pass") else "❌"
            parse_err = str(d.get("parse_error") or "—").replace("|", "｜")
            lines.append(
                f"| {date_str} | {run_id} | {task_id} | {cp_icon} | {parse_err} | {json_rate} | {cp_rate} |"
            )

    lines += [
        "",
        f"<!-- generated: {now_ts()} runs: {len(all_dirs)} -->",
        "",
    ]
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
        self.dry_run: bool = False

    # ---- パス ----

    def _artifacts_root(self) -> Path:
        return self.paths.tank_root / "artifacts"

    def _runs_root(self) -> Path:
        return self._artifacts_root() / "runs"

    def _report_dir(self, run_id: str) -> Path:
        return self.paths.tank_root / "lab_notebook" / "reports" / run_id

    def _notebooklm_dir(self) -> Path:
        return self.paths.tank_root / "lab_notebook" / "notes" / "NotebookLM"

    def _bundle_path(self, run_dir: Path) -> Path:
        run_id = run_dir.name
        return self._notebooklm_dir() / run_id / f"notebooklm_bundle-{run_id}.md"

    def _master_runs_table_path(self) -> Path:
        return self._notebooklm_dir() / "runs_master_table.md"

    def _master_reports_table_path(self) -> Path:
        return self._notebooklm_dir() / "reports_master_table.md"

    def _master_raw_data_table_path(self) -> Path:
        return self._notebooklm_dir() / "raw_data_master_table.md"

    # ---- スキャン ----

    def _find_run_dirs(self) -> List[Path]:
        """
        artifacts/runs/ と artifacts/ 直下を両方スキャン。
        - artifacts/runs/ を優先（既存 idempotent キーを保持）
        - 20990101-* はテストダミー、スキップ
        - result.json が存在するディレクトリのみ対象
        """
        seen: set = set()
        candidates: List[Path] = []

        for base in [self._runs_root(), self._artifacts_root()]:
            if not base.exists():
                continue
            for d in sorted(base.iterdir()):
                if not d.is_dir():
                    continue
                if not _RUN_ID_RE.match(d.name):
                    continue
                if d.name.startswith(_TEST_PREFIX):
                    continue
                if d.name in seen:
                    continue
                if not (d / "result.json").exists():
                    continue
                seen.add(d.name)
                candidates.append(d)

        return candidates

    # ---- 状態チェック ----

    def _report_exists(self, run_dir: Path) -> bool:
        """
        新しい正規の場所（lab_notebook/reports/）を優先チェック。
        後方互換として旧 artifacts/ 内も確認（再分析を防ぐため）。
        """
        run_id = run_dir.name
        new_dir = self._report_dir(run_id)
        if (new_dir / "report.md").exists() and (new_dir / "report.json").exists():
            return True
        # 後方互換: 旧 artifacts/ 内の report（読み取り専用扱い）
        return (run_dir / "report.md").exists() and (run_dir / "report.json").exists()

    def _load_report(self, run_dir: Path) -> Optional[ReportJSON]:
        """
        report.json を新旧どちらの場所からでも読み込む。
        新しい場所（lab_notebook/reports/）を優先する。
        """
        run_id = run_dir.name
        candidates = [
            self._report_dir(run_id) / "report.json",
            run_dir / "report.json",  # 旧 artifacts/ 内（後方互換）
        ]
        for rp in candidates:
            if rp.exists():
                try:
                    return ReportJSON.model_validate_json(rp.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return None

    def _bundle_exists(self, run_dir: Path) -> bool:
        return self._bundle_path(run_dir).exists()

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

    # ---- メイン処理 ----

    def _run_impl(self) -> None:
        all_dirs = self._find_run_dirs()

        needs_report = [d for d in all_dirs if not self._report_exists(d)]
        needs_bundle = [d for d in all_dirs if not self._bundle_exists(d)]

        master_tables_exist = (
            self._master_runs_table_path().exists()
            and self._master_reports_table_path().exists()
            and self._master_raw_data_table_path().exists()
        )

        if self.dry_run:
            self.log_event("info", f"[DRY RUN] total run dirs found: {len(all_dirs)}")
            self.log_event("info", f"[DRY RUN] needs report (LLM): {len(needs_report)}")
            self.log_event("info", f"[DRY RUN] needs bundle (no LLM): {len(needs_bundle)}")
            self.log_event("info", f"[DRY RUN] master tables exist: {master_tables_exist}")
            for d in needs_report:
                self.log_event("info", f"[DRY RUN]   report → {d}")
            for d in needs_bundle:
                self.log_event("info", f"[DRY RUN]   bundle → {self._bundle_path(d)}")
            if not master_tables_exist:
                self.log_event("info", f"[DRY RUN]   runs_master_table → {self._master_runs_table_path()}")
                self.log_event("info", f"[DRY RUN]   reports_master_table → {self._master_reports_table_path()}")
                self.log_event("info", f"[DRY RUN]   raw_data_master_table → {self._master_raw_data_table_path()}")
            return

        if not needs_report and not needs_bundle and master_tables_exist:
            self.log_event("info", "no new runs to analyze")
            return

        # --- LLM フェーズ: report 生成（上限あり）---
        to_analyze = self.enforce_limits(needs_report)
        processed = 0

        for run_dir in to_analyze:
            key = self._make_idempotent_key(run_dir)
            try:
                self.ensure_not_processed(key)
            except GuardrailError as e:
                self.log_event("warn", f"skip (idempotent): {run_dir.name} {e}")
                continue

            user_prompt = self._build_user_prompt(run_dir)
            raw = self.ask(SYSTEM_PROMPT, user_prompt, json_mode=True)

            try:
                report = ReportJSON.model_validate_json(raw)
            except ValidationError as e:
                self.log_event("warn", f"schema validation failed: {run_dir.name}: {e}")
                out_dir = self._report_dir(run_dir.name)
                safe_write_text(
                    out_dir / "report.md",
                    f"# Parse Error — {run_dir.name}\n\n"
                    f"Pydantic validation failed. Raw LLM output:\n\n```\n{raw}\n```\n",
                )
                continue

            out_dir = self._report_dir(run_dir.name)
            safe_write_text(out_dir / "report.json", report.model_dump_json(indent=2))
            safe_write_text(out_dir / "report.md", report.to_markdown())
            self.log_event("info", f"wrote report: {run_dir.name} → {out_dir}")
            processed += 1

        self.log_event("info", f"analyzed {processed}/{len(to_analyze)} runs")

        # --- Bundle フェーズ: 全件対象（LLM不要、上限なし）---
        # needs_bundle を再計算（LLM フェーズで新たに report が生成された分を含む）
        needs_bundle = [d for d in all_dirs if not self._bundle_exists(d)]
        bundled = 0

        for run_dir in needs_bundle:
            result = load_json(run_dir / "result.json")
            run_id = run_dir.name
            report_obj = self._load_report(run_dir)

            bundle_md = generate_bundle_md(result, report_obj, run_id)
            safe_write_text(self._bundle_path(run_dir), bundle_md)
            self.log_event("info", f"wrote bundle: {run_id}")
            bundled += 1

        self.log_event("info", f"generated {bundled}/{len(needs_bundle)} bundles")

        # --- Master Table フェーズ: 常に全件再生成 ---
        runs_md = generate_runs_master_table_md(all_dirs)
        safe_write_text(self._master_runs_table_path(), runs_md)
        self.log_event("info", f"wrote runs_master_table: {len(all_dirs)} runs")

        reports_md = generate_reports_master_table_md(all_dirs, self._load_report)
        safe_write_text(self._master_reports_table_path(), reports_md)
        self.log_event("info", f"wrote reports_master_table: {len(all_dirs)} runs")

        raw_data_md = generate_raw_data_master_table_md(all_dirs)
        safe_write_text(self._master_raw_data_table_path(), raw_data_md)
        self.log_event("info", f"wrote raw_data_master_table: {len(all_dirs)} runs")


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

    if "--dry-run" in sys.argv:
        agent.dry_run = True

    agent.run()
