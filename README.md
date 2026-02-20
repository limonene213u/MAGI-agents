# MAGI agents

![my image](./img/logo.png)

OpenRouter APIを利用したLLMエージェント群。  
EPYCサーバー（DL325 Gen10）上でsystemdにより管理・運用される。

### 構成概念
この研究およびシステムは、以下の`MELCHIOR`（理論）、`BALTHASAR`（実装）、`CASPER`（検証）という**三つの要素**を持つ。

1. MELCHIOR（理性）：中立判断・再現性設計・MoE/MLA評価軸
2. BALTHASAR（科学）：Runner / Unified JSON / Control-Compute分離
3. CASPER（人間性）：実験ログ / 実行環境差 / 現実の揺らぎ

この３要素により、研究を進める。

### DL325 Gen10 - Control plane powerd by EPYC
そのうち、本システムの中核となるControl Plane環境がHPE DL325 Gen10(EPYC Server)である。

## 設計方針

- 推論はOpenRouter経由のクラウドモデルで行う（**EPYCでの推論禁止方針に違反しない**）
- 目的別に複数エージェントを構成し、役割ごとに最適なモデルを使い分ける
- 実装言語はPython（既存runnerと統一）
- ログ・成果物は `tank/MAGI-agents/logs/` に書き出す
- エージェント間連携は現時点では疎結合・ファイルベース

## ディレクトリ構成

```
/home/limonene/ROCm-project/MAGI-agents/
├── base.py              # 共通基盤（OpenRouter API呼び出し）
├── config.yaml          # エージェント設定（モデル・役割ごと）
├── reviewer.py          # コーディング査読エージェント
├── doc_writer.py        # ドキュメント整理エージェント
├── result_analyst.py    # 実験結果整理エージェント
├── experimenter.py      # 実験継続実行エージェント
├── deploy/              # systemdユニットファイル
│   ├── magi-reviewer.service
│   ├── magi-doc-writer.service
│   ├── magi-result-analyst.service
│   └── magi-experimenter.service
├── AGENT_LOGS/          # 作業ログ（タイムスタンプ付きMarkdown）
├── CLI_Err_Logs/        # CLIエラーログ
├── AGENTS.md            # エージェント設計指針
├── AGENT_TASK.md        # 現在のタスク定義
└── ISSUES.md            # 既知の問題・課題

# ログ・成果物の出力先（tank）
/home/limonene/ROCm-project/tank/
└── MAGI-agents/
    └── logs/            # エージェント実行ログ
```

## エージェント一覧

| エージェント | ファイル | モデル（OpenRouter） | 有料/無料 | 役割 |
|------------|---------|-------------------|---------|------|
| コーディング査読 | `reviewer.py` | `deepseek/deepseek-r1` | 無料 | コードレビュー・ISSUES記録 |
| ドキュメント整理 | `doc_writer.py` | `google/gemini-flash-1.5` | 無料/激安 | Markdown整備・suggestions生成 |
| 実験結果整理 | `result_analyst.py` | `google/gemini-flash-1.5` | 無料/激安 | result.json解析・サマリ生成 |
| 実験継続実行 | `experimenter.py` | `anthropic/claude-sonnet-4-5` | 有料 | 複雑なタスク判断・実験フロー制御 |

モデルは `config.yaml` で管理し、容易に変更できる。

## セットアップ

```bash
cd /home/limonene/ROCm-project/MAGI-agents
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .envにOpenRouter APIキーを記載
```

## 認証

`.env` でAPIキーを管理する。`.gitignore` に含まれているため、リポジトリには含まれない。

```
OPENROUTER_API_KEY=sk-or-xxxxxxxx
```

## ログ規約

既存MAGI規約に準拠：

- ファイル名: `YYYYMMDD-HHMMSS_[agent-name].md`
- 形式: タイムスタンプ付きMarkdown
- 原則: 1作業1ファイル、履歴不変（過去エントリ改変禁止・訂正は追記）

## systemd運用

各エージェントは個別のserviceファイルで管理。

```bash
# 例：査読エージェントの起動
sudo systemctl enable --now magi-reviewer.service
sudo systemctl status magi-reviewer.service
```

## 参照ドキュメント

- `AGENTS.md` - エージェント設計指針
- `AGENT_TASK.md` - 現在のタスク定義
- `ISSUES.md` - 既知の問題・課題
- `../ROCm-MCP/AGENTS.md` - MAGIシステム全体の設計指針
- `../ROCm-MCP/MARCH_ROADMAP.md` - 3月発表に向けたロードマップ
- 設計提案: `~/Projects/research/DL325_Gen10/MAGI-agents/suggestions/MAGI-agents_20260220.md`
