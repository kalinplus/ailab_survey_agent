# P0 修复基线快照（2026-09-09）

诊断见 `docs/DeepSeek与MinerU输出质量问题及后续方案.md`。本目录冻结修复前的两轮真实运行工件，
用于修复后回归对照（§7 最终验收口径：同批论文冷/暖缓存复验 + 回归案例）。大文件被 .gitignore 排除，仅本清单入库。

## 文件与哈希

| 文件 | 字节 | sha256 |
|---|---:|---|
| deepseek_loose_run.log | 229850 | 7b96a37870d5f706… |
| deepseek_nothink_run.log | 474628 | 167c2395643f5c63… |
| evidence_store.json | 864984 | 0ae12a159d9d06d0… |
| paper_cards.json | 105080 | c46b59ffb8744ea9… |
| parsed_papers.json | 423187 | 1672654226bbff41… |

## 关键统计（修复前，用于验收对照）

- 当前轮（deepseek-loose）evidence：808 条 / 唯一 id 503 / 重号额外 305 条
- parsed_papers：2 篇；figure 记录 21（img_path 均未保存）；table 记录 0（恒空）
- v4 尝试 8 / 成功 2；agent 后备 6 次全失败（5×SSL + 1×read-file；详见日志）
- claim_map 33 条（6 supported / 9 weak / 18 unsupported）；grounding 0.318；stop=coverage_fail

## 回归案例（修复后必须逐条复核）

1. AvalonBench 名下出现 Resistance Avalanche 片段（跨论文冒名归属，phase5/repair 两处 backfill）
2. AlphaStar 名下挂一般 MARL 假设片段（同上）
3. MuZero「需要完美模拟器」被当成其自身局限（上一轮 nothink 稿 47 行）
4. evidence_id 冲突：GameNGen 149 段→44 组坐标；GTBench 262 段→63 组坐标

## .env 键清单（值不入库）

INTERN_API_BASE_URL, INTERN_API_KEY, MINERU_API_KEY, SCIVERSE_API_KEY, SCIVERSE_MIN_INTERVAL, HEAVY_LLM_BASE_URL, HEAVY_LLM_API_KEY, HEAVY_LLM_MODEL, HEAVY_LLM_THINKING_EFFORT
