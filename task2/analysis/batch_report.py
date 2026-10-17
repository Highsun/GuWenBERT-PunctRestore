"""Render the completed v0.3 measurements; does not load the model or run inference."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guwenbert_task2.common import ROOT, TASK, digest, read_json, write_json

import numpy as np


def render(experiment):
    directory = TASK / "reports" / experiment
    if (directory / "report-bundle.json").exists():
        raise FileExistsError("Report bundle exists; preserve it and use a new experiment ID")
    dev = read_json(directory / "development.json")
    test = read_json(directory / "test.json")
    selection = read_json(directory / "selection.json")
    if digest(directory / "selection.json") != test["selection_sha256"]:
        raise ValueError("Test selection receipt differs")
    os.environ.setdefault("MPLCONFIGDIR", str(TASK / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
                         "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.dpi": 300})
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    windows = dev["config"]["windows"]
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    saved = []

    def save(fig, name):
        for extension in ("png", "svg", "pdf"):
            path = figures / f"{name}.{extension}"
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            saved.append(path)
        plt.close(fig)

    def records(window):
        return sorted([r for r in dev["conditions"] if r["condition"]["window_id"] == window["id"]],
                      key=lambda r: r["condition"]["batch_size"])

    def batch_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(dev["config"]["batch_sizes"], [str(b) for b in dev["config"]["batch_sizes"]])
        ax.set_xlabel("Batch size")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for i, w in enumerate(windows):
        rs = records(w)
        batches = [r["condition"]["batch_size"] for r in rs]
        options = dict(color=colors[i], marker=markers[i], markersize=5, label=w["label"])
        axes[0].errorbar(batches, [r["median_seconds"] for r in rs],
                         yerr=[r["stdev_seconds"] for r in rs], capsize=3, **options)
        axes[1].plot(batches, [np.median([s["gpu_forward_argmax_seconds"] for s in r["measurements"]])
                               for r in rs], **options)
    for ax in axes:
        batch_axis(ax)
    axes[0].set(ylabel="End-to-end latency (s)", title="(a) Median latency with timing SD")
    axes[1].set(ylabel="GPU forward + argmax time (s)", title="(b) Median CUDA-event time")
    axes[1].legend(fontsize=8)
    save(fig, "01_batch_latency")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout="constrained")
    for i, (w, ax) in enumerate(zip(windows, axes.flat)):
        rs = records(w)
        batches = [r["condition"]["batch_size"] for r in rs]
        base_gpu = np.median([s["gpu_forward_argmax_seconds"] for s in rs[0]["measurements"]])
        ax.plot(batches, [r["speedup_vs_batch1"] for r in rs], "o-", color=colors[i], label="End-to-end / fixed B1")
        ax.plot(batches, [base_gpu / np.median([s["gpu_forward_argmax_seconds"] for s in r["measurements"]])
                         for r in rs], "s--", color="#666666", label="GPU time / fixed B1")
        ax.plot(batches, [r["comparisons"]["adaptive"]["speedup"] for r in rs], "^:", color="#D55E00",
                label="End-to-end / sentence-driven")
        chosen = selection["selected"][w["id"]]["batch_size"]
        ax.axvline(chosen, color=colors[i], alpha=.4, linewidth=1)
        batch_axis(ax)
        ax.set(ylabel="Speedup (x)", title=w["label"])
        ax.legend(fontsize=8)
    save(fig, "02_batch_speedup")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for i, w in enumerate(windows):
        rs = records(w)
        batches = [r["condition"]["batch_size"] for r in rs]
        options = dict(color=colors[i], marker=markers[i], label=w["label"])
        axes[0].plot(batches, [r["peak_allocated_bytes"] / 2**30 for r in rs], **options)
        axes[1].plot(batches, [r["forward_calls"] for r in rs], **options)
    for ax in axes:
        batch_axis(ax)
    axes[0].set(ylabel="Peak allocated GPU memory (GiB)", title="(a) Allocated memory")
    axes[1].set(ylabel="Forward calls", yscale="log", title="(b) Forward calls for identical windows")
    axes[0].legend(fontsize=8)
    save(fig, "03_batch_resources")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for i, w in enumerate(windows):
        rs = records(w)
        batches = [r["condition"]["batch_size"] for r in rs]
        base = rs[0]["quality"]["overall"]["punctuation_micro"]["f1"]
        options = dict(color=colors[i], marker=markers[i], label=w["label"])
        axes[0].plot(batches, [100 * (r["quality"]["overall"]["punctuation_micro"]["f1"] - base) for r in rs], **options)
        axes[1].plot(batches, [100 * r["different_characters_vs_batch1"] / dev["characters"] for r in rs], **options)
    for ax in axes:
        batch_axis(ax)
    axes[0].axhline(0, color="#777777", linewidth=.8)
    axes[0].set(ylabel="Micro-F1 change vs fixed B1 (pp)", title="(a) Numerical quality drift")
    axes[1].set(ylabel="Changed character labels (%)", title="(b) Character-level prediction drift")
    axes[1].legend(fontsize=8)
    save(fig, "04_batch_quality_drift")

    lookup = {r["condition"]["id"]: r for r in test["conditions"]}
    names = ["serial256-b1", "serial400-b1"]
    for w in windows:
        for name in (f"{w['id']}-b1", selection["selected"][w["id"]]["condition_id"]):
            if name not in names:
                names.append(name)
    rs = [lookup[name] for name in names]
    y = np.arange(len(rs))
    score = np.array([100 * r["quality"]["overall"]["punctuation_micro"]["f1"] for r in rs])
    interval = np.array([np.asarray(r["micro_f1_bootstrap_95ci"]) * 100 for r in rs])
    # Percentile intervals need not contain the point estimate; draw endpoints directly.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True, layout="constrained")
    row_colors = ["#666666" if r["condition"]["strategy"] == "adaptive" else
                  colors[next(i for i, w in enumerate(windows) if w["id"] == r["condition"]["window_id"])] for r in rs]
    axes[0].hlines(y, interval[:, 0], interval[:, 1], color=row_colors, linewidth=2)
    axes[0].scatter(score, y, color=row_colors, s=35, zorder=3)
    for i, r in enumerate(rs):
        times = [s["seconds"] for s in r["measurements"]]
        axes[1].hlines(i, min(times), max(times), color=row_colors[i], alpha=.4, linewidth=2)
        lo, hi = np.quantile(times, [.25, .75])
        axes[1].hlines(i, lo, hi, color=row_colors[i], linewidth=5)
        axes[1].scatter(r["median_seconds"], i, color=row_colors[i], s=35, zorder=3)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Punctuation micro-F1 (%)", title="(a) Test quality and document-bootstrap 95% CI")
    axes[1].set(xlabel="End-to-end latency (s, log scale)", xscale="log",
                title="(b) Median, IQR and full timing range")
    save(fig, "05_test_quality_latency")

    lines = ["# 任务二：Batch 扩展与测试集验证实验", "", "实验日期：2026-10-17。实验编号：`cclue-batch-scaling-v0.3`。", "",
             "## 1. 实验目的与前期结论", "",
             "本实验研究 batch 扩展对长文本断句标点推理的效率、显存与数值一致性的影响，并验证开发集选定参数在测试集上的表现。句长预算实验已完成并验收：少量上下文能够显著改善窗口交界，继续扩大预算的收益有限。本轮沿用任务一冻结的通用中文 BERT，不重新训练模型，也不改写前期结果。", "",
             "## 2. 实验设置", "",
             f"开发集为 {dev['documents']} 个匿名重建文档组、{dev['characters']:,} 字符；测试集为 {test['documents']} 组、{test['characters']:,} 字符。", "",
             "| 窗口方案 | 逻辑输入预算 | 核心预算 | 左右各上下文 | 划分规则 |", "| --- | ---: | ---: | ---: | --- |"]
    for w in windows:
        lines.append(f"| {w['id']} | {w['max_length']} | {w['core_tokens']} | {(w['max_length'] - 2 - w['core_tokens']) // 2} | {'所有长度保持固定核心' if w['anchored'] else '初版固定窗口，短文可整窗'} |")
    lines += ["", "开发集每个窗口方案比较 batch=1、4、8、16、32、64，并加入 256/192 与 400/300 两种句号推进串行参考。串行方法每次预测后在推进预算内最后一个预测句号处结束，无句号时按预算前进。前面三种固定方案与串行 256 比较，teacher400 与串行 400 比较；每种固定方案另保留相同窗口的 batch=1，分别衡量窗口设计变化与批量推理的作用。", "",
              "设备为 NVIDIA GeForce RTX 5070 12 GB，使用 BF16、SDPA 和动态 padding。输入预算包含两个特殊 token；median222 的逻辑输入上限为 222，实际 tensor 补齐至 224。每个条件预热一次，按五轮随机顺序测量。端到端耗时包含编码、构窗、排序、padding、传输、前向、argmax、解码和文本恢复，排除数据/模型加载、文件校验、计分和结果写出；GPU 事件时间单独记录。", "",
              "每种方案按开发集选择 batch：五次预测一致，micro-F1 与该方案 batch=16 的差值绝对值不超过 0.02 个百分点；在合格条件中选择中位耗时距最快值不超过 3% 的最小 batch。这一规则用于约束数值漂移和控制资源成本，不代表统计等效性。选择及校验值在测试前保存为 selection.json，测试结果不参与参数选择。", "",
              "## 3. 开发集 Batch 扩展", "",
              "![Batch 与耗时](figures/01_batch_latency.png)", "", "图 1：开发集端到端中位耗时及计时标准差，以及 GPU 前向与 argmax 的中位事件时间。", "",
              "![Batch 与加速比](figures/02_batch_speedup.png)", "", "图 2：相同固定窗口 batch=1 和句号推进串行方法的加速对照。竖线标出每个窗口方案选定的 batch。", "",
              "| 窗口方案 | 选定 batch | 开发 micro-F1/% | 中位耗时/s | 原文字符/秒 | 相对相同窗口 batch=1 加速 | 峰值 allocated/GiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    dev_lookup = {r["condition"]["id"]: r for r in dev["conditions"]}
    for w in windows:
        r = dev_lookup[selection["selected"][w["id"]]["condition_id"]]
        lines.append(f"| {w['id']} | {r['condition']['batch_size']} | {r['quality']['overall']['punctuation_micro']['f1']*100:.3f} | {r['median_seconds']:.4f} | {r['characters_per_second']:.0f} | {r['speedup_vs_batch1']:.2f}× | {r['peak_allocated_bytes']/2**30:.3f} |")
    lines += ["", "![Batch 与资源](figures/03_batch_resources.png)", "", "图 3：峰值已分配显存和 GPU 前向次数。reserved 显存受此前分配器缓存影响，原始值保存在 JSON 中，图中以 allocated 峰值比较资源需求。", "",
              "![Batch 数值漂移](figures/04_batch_quality_drift.png)", "", "图 4：相对同一窗口 batch=1 的 micro-F1 差值和字符标签变化比例。窗口位置不随 batch 改变。", "",
              "## 4. 测试集质量与速度", "",
              "![测试集质量与耗时](figures/05_test_quality_latency.png)", "", "图 5：测试集标点质量及文档级 bootstrap 95% 区间，和端到端计时的中位数、四分位区间及完整范围。", "",
              "| 测试条件 | micro-F1/% | macro-F1/% | 断句 F1/% | 句末边界 F1/% | 中位耗时/s | 原文字符/秒 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rs:
        q = r["quality"]["overall"]
        lines.append(f"| {r['condition']['id']} | {q['punctuation_micro']['f1']*100:.3f} | {q['punctuation_macro_f1']*100:.3f} | {q['segmentation_any_punctuation']['f1']*100:.3f} | {q['sentence_boundary']['f1']*100:.3f} | {r['median_seconds']:.4f} | {r['characters_per_second']:.0f} |")
    lines += ["", "断句 F1 衡量是否在字符后出现任意一种标点；句末边界 F1 将 。？！ 合并为句末；标点 micro/macro-F1 保留七类标点类型区别，均不包含无标点类别 O。", "",
              "| 选定测试方案 | 相同窗口 batch=1 加速 | 相对串行加速 | 相对串行 micro-F1 差值/百分点 | 配对 95% 区间 | 与同窗 batch=1 不同标签数 |",
              "| --- | ---: | ---: | ---: | --- | ---: |"]
    for w in windows:
        r = lookup[selection["selected"][w["id"]]["condition_id"]]
        contrast = r["comparisons"]["adaptive"]
        ci = contrast["paired_95ci_pp"]
        lines.append(f"| {r['condition']['id']} | {r['speedup_vs_batch1']:.2f}× | {contrast['speedup']:.2f}× | {contrast['delta_micro_f1_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['different_characters_vs_batch1']} |")
    lines += ["", "Bootstrap 按匿名文档组进行 2,000 次配对重采样，区间反映文档抽样波动，未作多重比较校正。计时重复使用同一模型与输入，不构成独立模型重复实验。", "",
              "## 5. 不同长度文档的质量", "",
              "| 选定测试方案 | ≤512 字符 micro-F1/% | 513～4096 字符 micro-F1/% | >4096 字符 micro-F1/% |",
              "| --- | ---: | ---: | ---: |"]
    for w in windows:
        r = lookup[selection["selected"][w["id"]]["condition_id"]]
        values = [r["quality_by_length"][key]["quality"]["punctuation_micro"]["f1"] * 100 for key in ("le512", "513to4096", "gt4096")]
        lines.append(f"| {r['condition']['id']} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")
    lines += ["", "长度分组按重建文档正文字符数划分；每组使用该组所有字符的汇总计数计算 F1。文档数量和逐类指标保存在 JSON 中。", "",
              "## 6. 实验范围与复现", "",
              f"开发集完成 {len(dev['conditions'])} 个条件，测试集完成 {len(test['conditions'])} 个条件。显存不足记录数量分别为 {len(dev['failed_conditions'])} 和 {len(test['failed_conditions'])}。所有条件五次预测一致性：{'全部一致' if all(r['repeat_predictions_identical'] for r in dev['conditions']+test['conditions']) else '存在差异，见原始测量记录'}。任务一冻结文件和已验收的句长预算实验通过运行前后校验。", "",
              "计时受 GPU 负载、频率及系统调度影响，全部原始测量均保留；接近的耗时应结合波动范围解读。本实验覆盖至 batch=64，不将其视为显存容量上限。长文由匿名片段组重建，无法恢复完整原始排版。测试结果验证当前四种窗口方案，尚未评估两侧不对称上下文、跨窗口概率融合或交界复核。", "",
              "```bash", "cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2", "uv sync --locked",
              "uv run --locked --offline pytest -q", "uv run --locked --offline python scripts/batch_study.py development",
              "uv run --locked --offline python scripts/batch_study.py test", "uv run --locked --offline python analysis/batch_report.py", "```", "",
              "已存在的实验阶段和图表报告拒绝覆盖。重新测量时复制配置并修改 experiment，两个阶段使用同一配置。新的图表报告可用 --experiment 指定实验目录。", "",
              "开发与测试汇总分别为 [development.csv](development.csv) 和 [test.csv](test.csv)，完整指标及原始计时见 [development.json](development.json)、[test.json](test.json)。测试前的选择记录见 [selection.json](selection.json)。详细预测、窗口索引及运行计划位于 code/outputs/task2/cclue-batch-scaling-v0.3/，运行源码快照位于 code/outputs/task2/source-snapshots/cclue-batch-scaling-v0.3/。五张图均提供 300 dpi PNG、SVG 和 PDF。", ""]
    path = directory / "experiment.md"
    with path.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    write_json(directory / "report-bundle.json", {
        "analysis_script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(Path(__file__))},
        "input_files_sha256": {p.name: digest(p) for p in (directory / "development.json", directory / "test.json", directory / "selection.json")},
        "generated_files_sha256": {str(p.relative_to(directory)): digest(p) for p in [*saved, path]},
        "matplotlib": matplotlib.__version__, "numpy": np.__version__})
    print(json.dumps({"report": str(path.relative_to(ROOT)), "figure_files": len(saved)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="cclue-batch-scaling-v0.3")
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in (".", ".."):
        parser.error("Invalid experiment ID")
    render(args.experiment)
