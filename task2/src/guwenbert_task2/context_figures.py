from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .common import TASK, digest, write_json

FAMILY_LABELS = {"fixed-window-256": "Fixed window W=256", "fixed-window-400": "Fixed window W=400",
                 "fixed-core-192": "Fixed core C=192"}
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "s", "^"]


def families(result):
    names = list(dict.fromkeys(r["condition"]["family"] for r in result["conditions"]))
    return [(name, sorted([r for r in result["conditions"] if r["condition"]["family"] == name],
                          key=lambda r: r["condition"]["context_tokens"])) for name in names]


def family_label(name):
    return FAMILY_LABELS.get(name, name)


def make_figures(result, directory: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(TASK / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "grid.linewidth": .6,
                         "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.dpi": 300})
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, name):
        for extension in ("png", "svg", "pdf"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    stats = result["training_sentence_statistics"]
    x = np.array([int(k) for k in stats["histogram"]])
    counts = np.array(list(stats["histogram"].values()))
    cdf = np.cumsum(counts) / counts.sum()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    axes[0].bar(x, counts, width=.9, color=COLORS[0])
    axes[0].set(xlabel="Sentence length (content tokens)", ylabel="Number of sentences (log scale)",
                yscale="log", xlim=(0, 120), title="(a) Training sentence-length distribution")
    axes[1].step(x, cdf * 100, where="post", color=COLORS[0], linewidth=2)
    for key, color, yy in [("median", COLORS[2], 52), ("p90", "#CC79A7", 75), ("p95", COLORS[1], 85), ("p99", "#555555", 95)]:
        value = stats[key]
        axes[1].axvline(value, color=color, linestyle="--", linewidth=.9)
        axes[1].text(value + 1, yy, f"{key.upper()}={value:g}", fontsize=8, color=color)
    axes[1].set(xlabel="Sentence length (content tokens)", ylabel="Cumulative percentage (%)",
                xlim=(0, 120), ylim=(0, 102), title="(b) Empirical cumulative distribution")
    fig.suptitle(f"Cleaned training data: {stats['count']:,} bounded spans; mean={stats['mean']:.2f}, max={stats['maximum']}", fontsize=11)
    save(fig, "01_sentence_lengths")

    groups = families(result)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    for i, (name, records) in enumerate(groups):
        h = np.array([r["condition"]["context_tokens"] for r in records])
        y = np.array([r["quality"]["overall"]["punctuation_micro"]["f1"] * 100 for r in records])
        axes[0].plot(h, y, marker=MARKERS[i], markersize=5, color=COLORS[i], label=family_label(name))
        delta = np.array([r["micro_f1_delta_vs_no_context_pp"] for r in records])
        ci = np.array([r["paired_delta_bootstrap_95ci_pp"] for r in records])
        # A percentile interval need not contain the point estimate. Plot interval endpoints directly.
        shifted = h + (i - 1) * .65
        axes[1].vlines(shifted, ci[:, 0], ci[:, 1], color=COLORS[i], alpha=.75, linewidth=1)
        axes[1].plot(shifted, delta, marker=MARKERS[i], markersize=4, color=COLORS[i], linewidth=1.2)
    axes[0].set(title="(a) Overall punctuation quality", xlabel="Context per side H (tokens)", ylabel="Punctuation micro-F1 (%)")
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=.8)
    axes[1].set(title="(b) Gain over H=0 with paired 95% intervals", xlabel="Context per side H (tokens)", ylabel="Micro-F1 change (percentage points)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Development documents; batch={result['config']['batch_size']}; document-level paired bootstrap", fontsize=11)
    save(fig, "02_quality_vs_context")

    fig, axes = plt.subplots(1, len(groups), figsize=(12.6, 4.1), sharey=True, layout="constrained")
    for i, ((name, records), ax) in enumerate(zip(groups, axes)):
        h = np.array([r["condition"]["context_tokens"] for r in records])
        speed = np.array([r["characters_per_second"] / 1000 for r in records])
        y = np.array([r["quality"]["overall"]["punctuation_micro"]["f1"] * 100 for r in records])
        ax.plot(speed, y, color="#AAAAAA", linewidth=.8, zorder=1)
        dots = ax.scatter(speed, y, c=h, cmap="viridis", vmin=min(h), vmax=max(h), s=42, edgecolors="white", linewidth=.5, zorder=2)
        best = int(h[np.argmax(y)])
        for j, (xx, yy, context) in enumerate(zip(speed, y, h)):
            if context in {0, 14, 31, 49, 96, best}:
                ax.annotate(f"H={context}", (xx, yy), xytext=(4, 6 if j % 2 else -11), textcoords="offset points", fontsize=7)
        ax.set(title=family_label(name), xlabel="Throughput (thousand original chars/s)")
    axes[0].set_ylabel("Punctuation micro-F1 (%)")
    fig.colorbar(dots, ax=axes, shrink=.8, label="Context per side H (tokens)")
    save(fig, "03_quality_throughput")

    anchored = next(records for name, records in groups if name.startswith("fixed-core"))
    h = [r["condition"]["context_tokens"] for r in anchored]
    edge = np.array([r["quality"]["window_edges"]["punctuation_micro"]["f1"] * 100 for r in anchored])
    inside = np.array([r["quality"]["window_interiors"]["punctuation_micro"]["f1"] * 100 for r in anchored])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    axes[0].plot(h, edge, "o-", color=COLORS[1], label="Near fixed core cuts")
    axes[0].plot(h, inside, "s-", color=COLORS[0], label="Core interiors")
    axes[0].set(title="(a) Same evaluation positions for every H", xlabel="Context per side H (tokens)", ylabel="Punctuation micro-F1 (%)")
    axes[0].legend(fontsize=9)
    axes[1].plot(h, edge - edge[0], "o-", color=COLORS[1], label="Near fixed core cuts")
    axes[1].plot(h, inside - inside[0], "s-", color=COLORS[0], label="Core interiors")
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=.8)
    axes[1].set(title="(b) Improvement over H=0", xlabel="Context per side H (tokens)", ylabel="Micro-F1 change (percentage points)")
    fig.suptitle(f"Fixed core C={result['config']['anchor_core_tokens']}; cut neighborhood: +/-{result['config']['boundary_margin']} characters", fontsize=11)
    save(fig, "04_boundary_effect")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), layout="constrained")
    for i, (name, records) in enumerate(groups):
        h = [r["condition"]["context_tokens"] for r in records]
        options = dict(marker=MARKERS[i], markersize=4, color=COLORS[i], label=family_label(name))
        axes[0, 0].errorbar(h, [r["median_seconds"] for r in records], yerr=[r["stdev_seconds"] for r in records], capsize=2, **options)
        axes[0, 1].plot(h, [r["windows"] for r in records], **options)
        axes[1, 0].plot(h, [r["overlap_work_factor"] for r in records], **options)
        axes[1, 1].plot(h, [r["peak_allocated_bytes"] / 2**30 for r in records], **options)
    for ax, title, ylabel in [(axes[0, 0], "(a) End-to-end latency", "Median time (s), +/- timing SD"),
                             (axes[0, 1], "(b) Number of windows", "Windows"),
                             (axes[1, 0], "(c) Repeated content computation", "Processed / original content tokens"),
                             (axes[1, 1], "(d) GPU memory", "Peak allocated memory (GiB)")]:
        ax.set(title=title, xlabel="Context per side H (tokens)", ylabel=ylabel)
    axes[0, 1].legend(fontsize=8)
    save(fig, "05_resource_cost")
    write_json(directory / "manifest.json", {"formats": ["png (300 dpi)", "SVG", "PDF"],
                                            "files_sha256": {p.name: digest(p) for p in outputs}})
    return outputs


def write_report(result, directory):
    stats, dataset, cfg = result["training_sentence_statistics"], result["dataset"], result["config"]
    lines = ["# 任务二：句长统计指导上下文预算的对照实验", "",
             f"实验日期：{result['date']}。实验编号：`{cfg['experiment']}`。", "",
             "## 实验目的", "",
             "研究固定窗口推理中，左右上下文长度对古文标点恢复质量及推理成本的影响。模型采用任务一冻结的通用中文 BERT，保持权重、字符编码、解码规则、BF16 精度和 batch size 一致。", "",
             "## 数据与实验设计", "",
             f"句长统计来自清理后的训练集，以 。？！ 为句末，统计相邻句末间的正文 token 数。连续片段仅在同一匿名文档且原始记录编号连续时连接，各连续区间首个缺少左边界的跨度及末尾不完整跨度排除。共获得 {stats['count']:,} 个句长样本。", "",
             "| 统计量 | 正文 token 数 |", "| --- | ---: |",
             f"| 均值 | {stats['mean']:.2f} |", f"| 中位数 | {stats['median']:g} |",
             *[f"| {key.upper()} | {stats[key]} |" for key in ("p75", "p90", "p95", "p99")],
             f"| 最大值 | {stats['maximum']} |", "",
             "![训练集句长分布](figures/01_sentence_lengths.png)", "",
             "图 1：训练集句长直方图与累计分布。横轴展示至 120 token，完整尾部保存在统计文件中。句长用于提出候选预算，窗口位置不读取句末金标。", "",
             f"评价使用官方开发集重建的 {dataset['documents']} 个匿名文档组，共 {dataset['characters']:,} 字符；其中 {dataset['longer_than_512_characters']} 组超过 512 字符，最长 {dataset['maximum_characters']:,} 字符。连接处及来源片段区间保留。测试集不参与本轮参数选择。", "",
             "候选上下文 H 取训练均值四舍五入、中位数及 P75/P90/P95/P99，另外加入 H=0、8、原配置 31/49 和较大预算 96 作为对照。左右各提供 H token。", "",
             "| 实验组 | 固定量 | 随 H 改变的量 | 作用 |", "| --- | --- | --- | --- |",
             "| 固定总窗长 256 | 总输入预算 256 | 核心预算 C=254−2H，核心边界 | 观察任务一训练长度下的质量与吞吐取舍 |",
             "| 固定总窗长 400 | 总输入预算 400 | 核心预算 C=398−2H，核心边界 | 观察题目示例附近的工程取舍 |",
             f"| 固定核心 {cfg['anchor_core_tokens']} | 核心区间及所有输出交界 | 逻辑窗长 C+2H+2 | 保持相同预测归属位置，直接观察添加上下文的影响 |", "",
             "固定核心组始终按同一核心预算划分，包括可放入较大输入窗的短文；因此各 H 的核心位置严格一致。逻辑窗口含两个特殊 token，实际 batch tensor 向上补齐至 8 的倍数，且不超过模型上限。固定总窗长组沿用初版短文整窗处理规则。", "",
             f"硬件为 {result['environment']['gpu']}，batch={cfg['batch_size']}，精度为 {cfg['precision']}。每个条件预热 {cfg['warmup_runs']} 次，再进行 {cfg['repetitions']} 个随机顺序测量轮次；计时报告中位数和标准差。", "",
             "## 总体质量与效果量", "",
             "![上下文长度与质量](figures/02_quality_vs_context.png)", "",
             "图 2：左图为标点 micro-F1；右图为相对各组 H=0 的百分点差值及配对文档 bootstrap 95% 区间。区间基于开发文档组重采样，属于探索性比较，未进行多重比较校正。", "",
             "| 实验组 | H | 核心预算 | 逻辑窗长 | micro-F1/% | macro-F1/% | 相对 H=0 差值/百分点 | 差值 95% 区间 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for record in result["conditions"]:
        c, q = record["condition"], record["quality"]["overall"]
        low, high = record["paired_delta_bootstrap_95ci_pp"]
        lines.append(f"| {c['family']} | {c['context_tokens']} | {c['core_tokens']} | {c['max_length']} | {q['punctuation_micro']['f1']*100:.3f} | {q['punctuation_macro_f1']*100:.3f} | {record['micro_f1_delta_vs_no_context_pp']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
    lines += ["", "各组结果概括：", ""]
    for name, records in families(result):
        best = max(records, key=lambda r: r["quality"]["overall"]["punctuation_micro"]["f1"])
        values = [r["quality"]["overall"]["punctuation_micro"]["f1"] * 100 for r in records]
        lines.append(f"- `{name}`：最高 micro-F1 为 {max(values):.3f}%，对应 H={best['condition']['context_tokens']}；各候选间的分数范围为 {max(values)-min(values):.3f} 个百分点。该条件相对无上下文基线的变化为 {best['micro_f1_delta_vs_no_context_pp']:+.3f} 个百分点。")
    anchored = next(records for name, records in families(result) if name.startswith("fixed-core"))
    best_anchor = max(anchored, key=lambda r: r["quality"]["overall"]["punctuation_micro"]["f1"])
    base_anchor = next(r for r in anchored if r["condition"]["context_tokens"] == 0)
    edge_gain = 100 * (best_anchor["quality"]["window_edges"]["punctuation_micro"]["f1"] - base_anchor["quality"]["window_edges"]["punctuation_micro"]["f1"])
    inner_gain = 100 * (best_anchor["quality"]["window_interiors"]["punctuation_micro"]["f1"] - base_anchor["quality"]["window_interiors"]["punctuation_micro"]["f1"])
    lines += ["", "## 窗口交界与核心内部", "",
              "![固定核心区的边界效果](figures/04_boundary_effect.png)", "",
              f"图 3：固定核心组各条件使用相同输出交界，交界左右各 {cfg['boundary_margin']} 字符归入边界区域，其余为内部区域。固定核心最佳条件 H={best_anchor['condition']['context_tokens']} 相对 H=0，边界区域 micro-F1 变化为 {edge_gain:+.3f} 个百分点，内部区域变化为 {inner_gain:+.3f} 个百分点。", "",
              "固定总窗长组改变核心预算后，实际交界位置也会变化。结果文件另外保存以 C=192 参考核心为基础的统一位置集合指标，供相同位置上的比较。", "",
              "## 效率与资源成本", "",
              "![质量与吞吐的关系](figures/03_quality_throughput.png)", "",
              "图 4：不同上下文预算的质量与原文字符吞吐，颜色表示每侧上下文 H。", "",
              "![时间与资源成本](figures/05_resource_cost.png)", "",
              "图 5：中位耗时及计时标准差、窗口数量、重复正文计算系数和峰值已分配显存。固定核心组的窗口数量保持不变；固定总窗长组在上下文增大时核心变短。", "",
              "| 实验组 | H | 中位耗时/s | 标准差/s | 字符/秒 | 窗口数 | 重复计算系数 | 峰值 allocated/GiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in result["conditions"]:
        c = r["condition"]
        lines.append(f"| {c['family']} | {c['context_tokens']} | {r['median_seconds']:.4f} | {r['stdev_seconds']:.4f} | {r['characters_per_second']:.0f} | {r['windows']} | {r['overlap_work_factor']:.3f} | {r['peak_allocated_bytes']/2**30:.3f} |")
    lines += ["", "## 讨论与复现", "",
              "句长均值和分位数提供了有依据的候选范围，最优上下文还受模型、窗口长度和文本内容影响。固定核心组隔离了核心位置变化，固定总窗长组则体现实际部署中的质量与重复计算取舍。较长上下文可能涵盖更多语义，也会引入其他句子的内容并改变训练时的上下文分布。", "",
              "本轮使用匿名开发文档重建长文，片段连接后的上下文与原始排版存在差异。bootstrap 描述文档抽样波动，计时标准差描述硬件测量波动；五次重复推理使用相同模型和输入，不视为五次独立质量实验。", "",
              f"全部条件的重复预测一致性：{all(r['repeat_predictions_identical'] for r in result['conditions'])}。模型权重保持冻结，任务一文件通过运行前后校验。", "",
              "端到端计时覆盖已加载模型下的编码、构窗、排序、padding、传输、前向、解码和文本恢复；不包含数据/模型加载、冻结校验、计分和文件写出。GPU 前向时间、原始重复计时、逐类指标、统一位置指标及文档级 TP/预测数/金标数保存在 `results.json`；汇总表为 `conditions.csv`。", "",
              "```bash", "cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2",
              "uv sync --locked", "uv run --locked --offline python scripts/context_study.py run",
              "uv run --locked --offline python scripts/context_study.py plot", "```", "",
              "完整实验已存在时拒绝覆盖。新实验使用新的 experiment；plot 仅从保存结果重新生成图表。图片提供 300 dpi PNG、SVG 和 PDF 三种格式，英文坐标轴配合本文中文图注可直接用于实验报告。", ""]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.md").write_text("\n".join(lines), encoding="utf-8")
