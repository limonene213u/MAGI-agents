"""
MAGI-agents reviewer.py — コーディング査読エージェント

設計方針:
  - AGENTS.md の reviewer I/O契約に準拠
  - 入力: ROCm-MCP/ ・ Peer_review_logs/ の .md ファイル（最大10ファイル）
  - 出力: tank/MAGI-agents/logs/YYYYMMDD-HHMMSS_reviewer.md
  - idempotent: コンテンツhash方式（sha256_text(text)[:16]）
  - 起動: 手動CLI（python reviewer.py）

査読観点:
  - contract consistency（契約の一貫性）
  - architectural clarity（アーキテクチャの明確性）
  - missing documentation（ドキュメント不足）
  - reproducibility risks（再現性リスク）
  ※ りもこ論文の評価フレームワークに合わせた観点

変更履歴:
  20260220 v1: スケルトン v2 をベースに本番実装
"""

from pathlib import Path

from base import (
    MAGIAgentBase,
    read_text_truncated,
    sha256_text,
    now_ts,
    safe_write_text,
)


# りもこ論文の評価フレームワークに合わせた査読プロンプト
# 参照: tank/docs-ref/日本語LLM評価のための再現可能な検証基盤設計.pdf
# 参照: tank/docs-ref/DeepSeek R1 Japanese Language Adaptation.pdf
SYSTEM_PROMPT = """\
You are a strict research code reviewer for a reproducible LLM evaluation study.

Your role:
- Identify architectural inconsistencies in the research infrastructure.
- Detect violations of documented contracts (spec vs implementation).
- Point out unclear specifications that would hinder reproducibility.
- Check for missing documentation required by the research governance.
- Assess whether failure taxonomy (build/import/runtime/oom/perf/numerical/unknown) is properly handled.
- Verify that artifact-first principles are maintained (all outputs must be traceable).

Review criteria aligned with the evaluation framework:
1. **Contract Consistency**: Do implementations match their spec/contract documents?
2. **Reproducibility**: Can experiments be re-run with identical results given the same spec?
3. **Failure Handling**: Are failures properly classified and logged as assets?
4. **Governance Compliance**: Are canonical sources, edit permissions, and audit trails respected?
5. **Safety Guardrails**: Are idempotency, rate limits, and credit checks properly implemented?

Do NOT rewrite everything.
Be concise, structured, and actionable.
Respond in Japanese when the input contains Japanese, otherwise in English.
"""


class ReviewerAgent(MAGIAgentBase):

    def __init__(self, repo_root: Path, tank_root: Path, config_path: Path):
        super().__init__(
            agent_name="reviewer",
            repo_root=repo_root,
            tank_root=tank_root,
            config_path=config_path,
        )

    def _collect_texts(self) -> str:
        """
        対象ディレクトリから最大10ファイルずつテキストを収集する。
        トークン爆発を防ぐため read_text_truncated でカット。
        """
        targets = [
            self.paths.repo_root / "ROCm-MCP",
            self.paths.repo_root / "ROCm-MCP" / "Peer_review_logs",
        ]

        combined = []
        for t in targets:
            if not t.exists():
                self.log_event("warn", f"target not found, skipping: {t}")
                continue

            files = sorted(t.rglob("*.md"))[:10]  # 最大10ファイル
            for p in files:
                text = read_text_truncated(p, self.cfg.guardrails.max_tokens_per_item)
                combined.append(f"\n# FILE: {p.relative_to(self.paths.repo_root)}\n{text}")

        return "\n".join(combined)

    def _run_impl(self) -> None:
        text = self._collect_texts()

        if not text.strip():
            self.log_event("warn", "No reviewable files found.")
            return

        # idempotentチェック（コンテンツhash方式）
        # マーカー名は {agent_name}_{key}.done 形式
        key = sha256_text(text)[:16]
        self.ensure_not_processed(key)

        user_prompt = f"""\
Review the following research repository snapshot.

Focus on:
1. Contract consistency (spec vs implementation)
2. Architectural clarity (separation of concerns)
3. Missing documentation (required by governance)
4. Potential reproducibility risks (non-determinism, missing artifacts)
5. Failure taxonomy coverage (build/import/runtime/oom/perf/numerical/unknown)

--- BEGIN SNAPSHOT ---
{text}
--- END SNAPSHOT ---
"""

        response = self.ask(SYSTEM_PROMPT, user_prompt)

        # tankへ書き出し（タイムスタンプ付き・MAGI規約準拠）
        out_path = (
            self.paths.tank_logs_dir
            / f"{now_ts()}_{self.cfg.name}.md"
        )
        safe_write_text(out_path, response)

        self.log_event("info", f"review written: {out_path}")


if __name__ == "__main__":
    repo_root = Path("/home/limonene/ROCm-project")
    tank_root = repo_root / "tank"
    config_path = repo_root / "MAGI-agents" / "config.yaml"

    agent = ReviewerAgent(repo_root, tank_root, config_path)
    agent.run()
