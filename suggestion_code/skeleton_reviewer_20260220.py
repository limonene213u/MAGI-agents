# skeleton: reviewer.py
# /home/limonene/ROCm-project/MAGI-agents/reviewer.py
#
# 注意: これはスケルトンコードです。MVP（最小実装）。
# Codexはこのファイルを参考に実装を行い、
# 実際のコードを /home/limonene/ROCm-project/MAGI-agents/reviewer.py に配置してください。
# 設計方針・仕様は以下を参照:
#   - MAGI-agents_AGENTS_20260220.md（Hard Rules・I/O契約）
#   - MAGI-agents_idempotent_design_20260220.md（idempotentキー設計）
#
# 【Codexへの参照指示】
# 実装前に以下のドキュメントを必ず参照すること:
#   - OpenRouter APIスキーマ:  tank/docs-ref/Coding_ref/OpenRouter_Docs-API_Reference/
#   - 研究論文（査読観点の確認・著者: 伊藤あきら / Akira Ito）:
#       tank/docs-ref/ 直下の PDF（DeepSeek R1 Japanese Language Adaptation...）
#       tank/docs-ref/ 直下の PDF（日本語LLM評価のための再現可能な検証...）
#   ※ tank/docs-ref/AI_Papers/ は第三者の参照論文なので混同しないこと
# SYSTEM_PROMPTの査読観点（contract consistency / reproducibility risksなど）は、
# 上記りもこ論文の評価フレームワーク・再現性基準と整合させること。
#
# 変更点（にゃんにゃん原案からの修正）:
#   - 出力ファイル名にタイムスタンプを付与（MAGI規約準拠）
#   - idempotentマーカー名を {agent_name}_{key}.done 形式に統一
#   - make_file_key() をbase.pyと同様のhash方式に統一

# /home/limonene/ROCm-project/MAGI-agents/reviewer.py

from pathlib import Path

from base import (
    MAGIAgentBase,
    read_text_truncated,
    sha256_text,
    now_ts,
    safe_write_text,
)


SYSTEM_PROMPT = """
You are a strict research code reviewer.

Your role:
- Identify architectural inconsistencies.
- Detect violations of documented contracts.
- Point out unclear specifications.
- Suggest minimal, precise improvements.

Do NOT rewrite everything.
Be concise, structured, and actionable.
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
            self.paths.repo_root / "Peer_review_logs",
        ]

        combined = []
        for t in targets:
            if not t.exists():
                self.log_event("warn", f"target not found, skipping: {t}")
                continue

            files = sorted(t.rglob("*.md"))[:10]  # 最大10ファイル
            for p in files:
                text = read_text_truncated(p, self.cfg.guardrails.max_tokens_per_item)
                combined.append(f"\n# FILE: {p}\n{text}")

        return "\n".join(combined)

    def _run_impl(self) -> None:
        text = self._collect_texts()

        if not text.strip():
            self.log_event("warn", "No reviewable files found.")
            return

        # idempotentチェック（コンテンツhash方式）
        # マーカー名は {agent_name}_{key}.done 形式（設計: idempotent_design参照）
        key = sha256_text(text)[:16]
        self.ensure_not_processed(key)  # base.py側でマーカー管理・最新3件保持

        user_prompt = f"""
Review the following research repository snapshot.

Focus on:
1. Contract consistency
2. Architectural clarity
3. Missing documentation
4. Potential reproducibility risks

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
