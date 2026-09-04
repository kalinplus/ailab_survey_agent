# T2 评测度量 gold-free 改造（含评测器引用归一化）

Status: dispatching (2026-09-04 晚)

## Motivation

S0 评测暴露两类评测侧问题：

1. **gold 指标失真**：reference recall 分母是单篇 gold 综述的 338 篇，
   语料 51 篇时上限只有 0.15，对广度型语料天然不公；单篇 gold 也不该当
   ground truth。
2. **格式耦合低估**：GLM 把引用放句号后（"…scale. [id]."），评测器句子
   切分后 citation-only fragment 被丢 → L1 cited=2、L0 正文节假性零引用。
   写作端已修（`e007ac7` 归位到句内），评测器侧也需对残留 fragment 归一。

## Boundary（触达文件）

- `tools/evaluate_survey.py`（gold 区块 + 引用归一化）
- `scripts/run_survey_eval.py`（新指标接线/CLI 参数）
- 对应既有测试（`tests/unit/test_evaluate_survey.py`，不建新文件）
- `cache/seed_survey_bibs.json`（新产物，由脚本生成而非手写）
- **不碰**：`tools/write_survey.py`、`tools/phases/`、`PROGRESS.md`、`specs/`。

## Slices

1. **引用归一化**：`_iter_sentence_claims` 前对 citation-only fragment
   （剥括号后为空的句）与前句合并再判；保证 `…text. [paper:x].` 形态的
   引用可绑定（单测覆盖该形态）。
2. **gold-free 主指标**（替换 reference recall 的主位）：
   a. `cache/seed_survey_bibs.json`：从 P2 已读的 7 篇种子综述提取参考文献
      并集（标题规范化去重）。生成脚本放 `scripts/`（一次性、可重跑，
      真实数据来自种子综述 artifact，不手写）。
   b. 报 **seed-bib recall**（语料 ∩ seed bib 并集 / 并集大小）与
      **in-seed-bib rate**（语料中被引比例）。
   c. **canonical 命中清单**：`cache/canonical_papers.json` 手工维护
      ~20 篇主题地标（主会话提供初始清单），报 hit@N。
   d. 语料多样性：每 aspect 的年份分布 / 来源（venue）数 / 引用数分位。
3. 原 gold in-gold 降级为次要诊断指标，报告里保留但标注 caveat。

## Acceptance Criteria（unit，无网络）

1. citation-only fragment 合并后，`…scale. [paper:x].` 形态产出 1 条
   claim 对（当前实现为 0）。
2. seed-bib recall / in-seed-bib / hit@N / 多样性指标在构造样本上数值可
   复算（手算断言）。
3. 不带 `--seed-bibs` / `--canonical` 参数时报告结构向后兼容（缺省区块
   显示 skipped 而非报错）。
4. `pytest tests/unit/` 全绿。

## Real-API Acceptance（S0 下轮真跑，主会话）

用尝试 5 的工件回放：L1 cited 句数从 2 恢复到 ≥15；新指标对基线 vs 尝试 5
给出可解释的方向（尝试 5 的 seed-bib 指标应显著高于 0）。
