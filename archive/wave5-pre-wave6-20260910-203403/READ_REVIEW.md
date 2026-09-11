# 旧波5最终稿通读记录

本记录是波6的固定回归输入，不是最终验收。已完整阅读本归档 output/survey.md，未做旧稿视觉验收。

- 最终进程退出：quality_failed；fixpoint；42 pairs：supported17 / weak23 / unsupported2。
- 正文9/21/31/35/37/49/53/63/67行：Rule-based taxonomy fallback 泄漏到引言/过渡/正文。
- 23/27行：AlphaStar与MuZero领域背景被放在Procedural Content Generation节；MuZero perfect-simulator背景仍被方法框架吸收。
- 对照cache/paper_cards.json：MuZero key_results=摘要首句Constructing agents...，method=第二句Tree-based planning...perfect simulator。均指向不存在语义意义的para_p0_0。是extractive fallback直接证据。
- cache/evidence_store.json：5853证据；supports_claims direct114 / indirect10923 / contradictory660。非空supports不等于支持。
- 55/57/59行：AlphaZero背景描述传统棋类引擎，被拿来和GameNGen比较；不能通过换句话说解决归属。
- 61/75行：内部证据覆盖状态作为论文事实引用，误把本系统“没有看到限制”当论文“没有限制”。
- 69/71行：Genie latent action描述近重复。
- 73/85行：evaluation_protocol_matrix重复嵌入。
- 87/89行：Future Directions重复标题。
- 91行：大量重复建议句，引用让新提出的实验建议看起来已经被原论文支持。
- 102行：Conclusion固定泛句，未总结文内具体发现。
- 摘要4行末句：多组数值的引用没有逐命题绑定。
- 旧render日志明确WeasyPrint native libraries导入失败，降级纯文本PDF；DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib已验证import成功，波6真实运行采用此环境。

上述不能随修复prompt自动认定解决；需在波6同run成品与原证据中重新核对。
