# 任务二实验记录

| 日期 | 实验编号 | 内容 | 状态 | 结果 |
| --- | --- | --- | --- | --- |
| 2026-10-16 | cclue-task2-v0.1-l256 / l400 | 串行、固定单窗与批量滑窗初版对照 | 已完成 | [初版汇总](initial-summary.md) |
| 2026-10-17 | cclue-context-budget-v0.2 | 训练句长指导上下文预算，33 条件、165 次正式测量 | 已完成并获用户验收 | [实验报告](cclue-context-budget-v0.2/experiment.md)、[结果分析](cclue-context-budget-v0.2/conclusions.md) |
| 2026-10-17 | cclue-batch-scaling-v0.3 | 开发集 batch 扩展与锁定参数后的测试集评价 | 已完成并获用户验收，180 次正式测量与独立核验通过 | [实验报告](cclue-batch-scaling-v0.3/experiment.md)、[结果分析](cclue-batch-scaling-v0.3/conclusions.md)、[参数选择](cclue-batch-scaling-v0.3/selection.json) |

上下文预算实验的验收记录及文件校验值见 [context-v0.2-accepted.json](checkpoints/context-v0.2-accepted.json)。固定核心 192 时，H=14 相对 H=0 总体标点 micro-F1 提高 1.552 个百分点，交界区域提高 20.018 个百分点；H=8～66 总体差异较小，H=96 下降。该结果作为已完成阶段保留，后续实验使用新的输出目录。

Batch 扩展实验显示，当前四种窗口方案在 batch=16～32 附近达到较好的耗时水平，64 未继续提速。句长中位数方案 222/192/14、batch=32 的测试 micro-F1 为 68.509%，中位耗时 2.142 秒，相对 256-token 串行参考加速 10.88 倍，质量差值为 −0.124 个百分点，配对区间包含零。开发集 H=49 的分数优势未在测试集得到明确确认。34 项工程测试通过，图表提供 PNG、SVG、PDF 三种格式。

Batch 扩展实验的验收记录见 [batch-v0.3-accepted.json](checkpoints/batch-v0.3-accepted.json)。本阶段归纳见[阶段总结](stage-summary.md)，下一阶段[跨窗口概率融合](../docs/next-stage-plan.md)待实施。
