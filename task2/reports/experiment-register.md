# 任务二实验记录

方法的研究动机、对照设计和结果如何引出下一阶段，见[研究路线](../docs/research-evolution.md)；融合与复核阶段归纳见[阶段总结](stage-summary-v0.5.md)。

| 日期 | 实验编号 | 内容 | 状态 | 结果 |
| --- | --- | --- | --- | --- |
| 2026-10-16 | cclue-task2-v0.1-l256 / l400 | 串行、固定单窗与批量滑窗初版对照 | 已完成 | [初版汇总](initial-summary.md) |
| 2026-10-17 | cclue-context-budget-v0.2 | 训练句长指导上下文预算，33 条件、165 次正式测量 | 已完成并获用户验收 | [实验报告](cclue-context-budget-v0.2/experiment.md)、[结果分析](cclue-context-budget-v0.2/conclusions.md) |
| 2026-10-17 | cclue-batch-scaling-v0.3 | 开发集 batch 扩展与锁定参数后的测试集评价 | 已完成并获用户验收，180 次正式测量与独立核验通过 | [实验报告](cclue-batch-scaling-v0.3/experiment.md)、[结果分析](cclue-batch-scaling-v0.3/conclusions.md)、[参数选择](cclue-batch-scaling-v0.3/selection.json) |
| 2026-10-17 | cclue-window-fusion-v0.4 | 融合初始实现的开发集测量 | 11 条件、55 次正式测量保留为优化前对照 | [优化记录](cclue-window-fusion-v0.4/pilot-note.md) |
| 2026-10-17 | cclue-window-fusion-v0.4-optimized | 局部索引优化后的概率融合 | 开发与测试共 20 条件、100 次正式测量，独立核验通过 | [报告](cclue-window-fusion-v0.4-optimized/experiment.md)、[结果分析](cclue-window-fusion-v0.4-optimized/conclusions.md)、[参数选择](cclue-window-fusion-v0.4-optimized/selection.json) |
| 2026-10-17 | cclue-selective-review-v0.5 | 不确定性/分歧筛选的交界批量复核，与随机和全量复核对照 | 开发与测试共 19 条件、95 次正式测量，独立核验通过；未达到质量采用条件 | [报告](cclue-selective-review-v0.5/experiment.md)、[结果分析](cclue-selective-review-v0.5/conclusions.md)、[参数选择](cclue-selective-review-v0.5/selection.json) |

上下文预算实验的验收记录及文件校验值见 [context-v0.2-accepted.json](checkpoints/context-v0.2-accepted.json)。固定核心 192 时，H=14 相对 H=0 总体标点 micro-F1 提高 1.552 个百分点，交界区域提高 20.018 个百分点；H=8～66 总体差异较小，H=96 下降。该结果作为已完成阶段保留，后续实验使用新的输出目录。

Batch 扩展实验显示，当前四种窗口方案在 batch=16～32 附近达到较好的耗时水平，64 未继续提速。句长中位数方案 222/192/14、batch=32 的测试 micro-F1 为 68.509%，中位耗时 2.142 秒，相对 256-token 串行参考加速 10.88 倍，质量差值为 −0.124 个百分点，配对区间包含零。开发集 H=49 的分数优势未在测试集得到明确确认。34 项工程测试通过，图表提供 PNG、SVG、PDF 三种格式。

Batch 扩展实验的验收记录见 [batch-v0.3-accepted.json](checkpoints/batch-v0.3-accepted.json)。前三阶段归纳见[阶段总结](stage-summary.md)，[跨窗口概率融合计划](../docs/next-stage-plan.md)已完成实施。

融合实验中，original256 的线性中心加权使测试 micro-F1 提高 0.176 个百分点，区间为 [0.087, 0.243]，耗时增加约 5.1%，相对串行加速 9.10 倍；等权融合降低总体质量。初始实现的聚合开销经过局部索引优化后显著下降，两轮开发集概率及预测保持一致。46 项工程测试与概率缓存重放核验通过。

选择性复核实验锁定 disagreement-10，测试 micro-F1 为 68.650%，无复核基线为 68.652%，质量差值区间包含零；耗时增加 10.16%。相对同预算随机复核的增益区间也包含零。分歧评分集中选中了较难区域，但直接替换同时引入新错误，未确认净质量收益；继续保留上一阶段单遍线性融合。56 项工程测试和独立概率重放通过，新增六幅图均提供 PNG、SVG、PDF。前一融合阶段的保护记录见 [fusion-v0.4-completed.json](checkpoints/fusion-v0.4-completed.json)。
