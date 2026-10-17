"""Generate the completed fusion-study report and scientific figures from saved results."""
from pathlib import Path
import argparse
import os
import sys
import statistics

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guwenbert_task2.common import ROOT,TASK,digest,read_json,write_json
import numpy as np


def render(experiment):
    directory=TASK/"reports"/experiment
    if (directory/"report-bundle.json").exists():raise FileExistsError("Report bundle exists; preserve previous results")
    dev=read_json(directory/"development.json");test=read_json(directory/"test.json");selection=read_json(directory/"selection.json")
    assert test["selection_sha256"]==digest(directory/"selection.json")
    os.environ.setdefault("MPLCONFIGDIR",str(TASK/".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.2,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42,"savefig.dpi":300})
    colors=["#0072B2","#009E73","#D55E00","#CC79A7","#666666"]
    labels={"core":"Integer core","prob-core":"Probability core","uniform":"Uniform","weighted1":"Center linear","weighted2":"Center squared"}
    figdir=directory/"figures";figdir.mkdir(exist_ok=True);paths=[]
    windows=dev["config"]["windows"]
    optimization=None
    if "pilot_experiment" in dev["config"]:
        pilot=read_json(TASK/"reports"/dev["config"]["pilot_experiment"]/"development.json")
        before={r["condition"]["id"]:r for r in pilot["conditions"]}
        optimization=[]
        for r in dev["conditions"]:
            old=before[r["condition"]["id"]]
            assert old["prediction_sha256"]==r["prediction_sha256"]
            assert old["probability_sha256"]==r["probability_sha256"]
            optimization.append({"condition":r["condition"]["id"],"predictions_identical":True,"probabilities_identical":True,
                                 "before_seconds":old["median_seconds"],"after_seconds":r["median_seconds"],
                                 "before_aggregation_seconds":statistics.median(s.get("aggregation_seconds",0) for s in old["measurements"]),
                                 "after_aggregation_seconds":statistics.median(s.get("aggregation_seconds",0) for s in r["measurements"]),
                                 "before_slowdown_vs_core":old.get("comparisons",{}).get("core",{}).get("latency_ratio"),
                                 "after_slowdown_vs_core":r.get("comparisons",{}).get("core",{}).get("latency_ratio")})
        write_json(directory/"implementation-comparison.json",{"split":"development","pilot_experiment":dev["config"]["pilot_experiment"],"conditions":optimization})

    def save(fig,name):
        for extension in ("png","svg","pdf"):
            p=figdir/f"{name}.{extension}"
            if p.exists():raise FileExistsError(p)
            fig.savefig(p,bbox_inches="tight",facecolor="white");paths.append(p)
        plt.close(fig)

    def mode_name(r):
        c=r["condition"];return f"weighted{c['power']}" if c["mode"]=="weighted" else c["mode"]

    def family(result,w):return [r for r in result["conditions"] if r["condition"]["window_id"]==w["id"]]

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for ax,w in zip(axes,windows):
        rs=family(dev,w);x=np.arange(len(rs));y=[r["quality"]["overall"]["punctuation_micro"]["f1"]*100 for r in rs]
        ax.bar(x,y,color=colors[:len(rs)],width=.65)
        ax.set_xticks(x,[labels[mode_name(r)] for r in rs],rotation=25,ha="right")
        ax.set(ylabel="Punctuation micro-F1 (%)",title=w["label"],ylim=(min(y)-.15,max(y)+.15))
        for xx,yy in zip(x,y):ax.annotate(f"{yy:.3f}",(xx,yy),xytext=(0,5),textcoords="offset points",ha="center",fontsize=8)
    save(fig,"01_development_quality")

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for ax,w in zip(axes,windows):
        rs=family(test,w)
        for i,r in enumerate(rs):
            comp=r["comparisons"]["core"];lo,hi=comp["paired_95ci_pp"]
            ax.hlines(i,lo,hi,color=colors[i],linewidth=2)
            ax.scatter(comp["delta_micro_f1_pp"],i,color=colors[i],s=35,zorder=3)
        ax.axvline(0,color="#777777",linestyle="--",linewidth=1)
        ax.set_yticks(np.arange(len(rs)),[labels[mode_name(r)] for r in rs]);ax.invert_yaxis()
        ax.set(xlabel="Micro-F1 change vs integer core (pp)",title=w["label"])
    save(fig,"02_test_quality_effect")

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for ax,w in zip(axes,windows):
        rs=family(test,w);x=np.arange(len(rs));base=rs[0]
        for offset,key,label,color in [(-.16,"window_edges","Near core cuts",colors[2]),(.16,"window_interiors","Core interiors",colors[0])]:
            gain=[100*(r["quality"][key]["punctuation_micro"]["f1"]-base["quality"][key]["punctuation_micro"]["f1"]) for r in rs]
            ax.bar(x+offset,gain,width=.3,label=label,color=color)
        ax.set_xticks(x,[labels[mode_name(r)] for r in rs],rotation=25,ha="right")
        ax.axhline(0,color="#777777",linewidth=.8);ax.set(ylabel="Micro-F1 change vs integer core (pp)",title=w["label"])
        ax.legend(fontsize=8)
    save(fig,"03_boundary_effect")

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for ax,w in zip(axes,windows):
        rs=family(test,w);x=np.arange(len(rs))
        ax.bar(x,[r["median_seconds"] for r in rs],color=colors[:len(rs)],width=.65,
               yerr=[r["stdev_seconds"] for r in rs],capsize=3)
        ax.set_xticks(x,[labels[mode_name(r)] for r in rs],rotation=25,ha="right")
        ax.set(ylabel="End-to-end latency (s)",title=w["label"])
        for xx,r in zip(x,rs):ax.annotate(f"{r['median_seconds']:.3f}",(xx,r['median_seconds']),xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
    save(fig,"04_fusion_latency")

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for i,w in enumerate(windows):
        rs=family(test,w);x=np.arange(len(rs))
        axes[0,i].bar(x,[r["measurements"][0].get("d2h_probability_bytes",r["measurements"][0].get("d2h_integer_bytes",0))/2**20 for r in rs],color=colors[:len(rs)],width=.65)
        axes[0,i].set(ylabel="D2H payload (MiB)",title=w["label"])
        axes[1,i].bar(x,[statistics.median(s.get("aggregation_seconds",0) for s in r["measurements"]) for r in rs],color=colors[:len(rs)],width=.65)
        axes[1,i].set(ylabel="Probability validation + aggregation (s)")
        for ax in axes[:,i]:ax.set_xticks(x,[labels[mode_name(r)] for r in rs],rotation=25,ha="right")
    save(fig,"05_transfer_aggregation_cost")

    if optimization:
        lookup={r["condition"]["id"]:r for r in dev["conditions"]}
        rs=[r for r in optimization if lookup[r["condition"]]["condition"]["mode"] in ("uniform","weighted")]
        x=np.arange(len(rs));fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
        for offset,prefix,label,color in [(-.17,"before","Initial",colors[2]),(.17,"after","Optimized",colors[0])]:
            axes[0].bar(x+offset,[r[f"{prefix}_aggregation_seconds"] for r in rs],width=.32,label=label,color=color)
            axes[1].bar(x+offset,[100*(r[f"{prefix}_slowdown_vs_core"]-1) for r in rs],width=.32,label=label,color=color)
        for ax in axes:
            ax.set_xticks(x,[r["condition"].replace("median222-","Median\n").replace("original256-","Original\n") for r in rs],rotation=15,ha="right")
            ax.legend(fontsize=8)
        axes[0].set(ylabel="Validation + aggregation time (s)",title="(a) Window-local index optimization")
        axes[1].axhline(15,color="#777777",linestyle="--",linewidth=1)
        axes[1].set(ylabel="End-to-end overhead vs own core baseline (%)",title="(b) Baseline-normalized overhead")
        save(fig,"06_implementation_optimization")

    lines=["# 任务二：跨窗口概率融合实验", "", f"实验日期：2026-10-17。实验编号：`{experiment}`。", "",
           "## 1. 目的与实验设计", "",
           "本实验研究重叠窗口对同一字符的概率融合，能否降低固定并行窗口的交界误差，并评价相对现有整数标签输出的额外成本。使用任务一冻结的通用中文 BERT，不重新训练。", "",
           "两种窗口方案为 median222（总输入 222、核心 192、每侧上下文 14、batch=32，始终保持核心划分）和 original256（256/192/31、batch=16，短文可整窗）。输入预算含两个特殊 token，median222 的 tensor 补齐至 224。对比保持窗口位置、排序、模型输入、batch 和 BF16 一致。", "",
           "每种方案包含原整数标签核心输出、保留概率的核心控制、等权融合及中心加权融合。核心概率控制仅使用所属核心窗口的预测，用于分离保留概率的开销。开发集比较线性与平方中心权重；测试集保留各方案选定的权重。串行参考使用 256/192 的句号推进方法。", "",
           "窗口正文 token 数为 L，字符首个 token 的零基位置为 a、末尾边界为 b，距离 d=min(a,L−b)+1，中心权重 w=(d/ceil(L/2))^α；α=1 为线性，α=2 为平方。融合先计算 Σ(w·p)/Σw，再取 argmax；等权取 w=1。字符对齐采用首个 subtoken，权重距离使用完整字符的 token 边界。softmax 使用 FP32，主机聚合使用 FP64。", "",
           f"开发集共 {dev['documents']} 个匿名文档组、{dev['characters']:,} 字符；测试集 {test['documents']} 组、{test['characters']:,} 字符。模型和数据均沿用上一阶段。", "",
           "每条件预热一次，随机排序正式测量五轮。开发集按总体标点 micro-F1 选择加权参数；距最高分不超过 0.02 个百分点时优先较低中位耗时，完全相同则取较小 α。选择记录在测试前锁定，不根据测试分数继续调整。", "",
           "## 2. 开发集结果与参数选择", "",
           "![开发集融合质量](figures/01_development_quality.png)", "", "图 1：开发集各输出方式的标点 micro-F1。纵轴局部放大以展示细小差异。", "",
           "| 条件 | micro-F1/% | 边界 micro-F1/% | 中位耗时/s |", "| --- | ---: | ---: | ---: |"]
    for r in dev["conditions"]:
        lines.append(f"| {r['condition']['id']} | {r['quality']['overall']['punctuation_micro']['f1']*100:.3f} | {r['quality']['window_edges']['punctuation_micro']['f1']*100:.3f} | {r['median_seconds']:.4f} |")
    lines += ["", "各窗口选定的中心权重："]
    for w in windows:
        choice=selection["selected"][w["id"]];lines.append(f"- {w['id']}：α={choice['power']}，开发集 micro-F1={choice['development_micro_f1']*100:.3f}%。")
    lines += ["", "## 3. 测试集质量与效果量", "",
              "![测试集融合效果量](figures/02_test_quality_effect.png)", "", "图 2：相对同窗整数标签核心输出的 micro-F1 差值和文档级配对 bootstrap 95% 区间。", "",
              "| 条件 | micro-F1/% | macro-F1/% | 断句 F1/% | 句末 F1/% | 相对核心差值/百分点 | 配对 95% 区间 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in test["conditions"]:
        q=r["quality"]["overall"];comp=r.get("comparisons",{}).get("core")
        delta=f"{comp['delta_micro_f1_pp']:+.3f}" if comp else "—"
        ci=f"[{comp['paired_95ci_pp'][0]:+.3f}, {comp['paired_95ci_pp'][1]:+.3f}]" if comp else "—"
        lines.append(f"| {r['condition']['id']} | {q['punctuation_micro']['f1']*100:.3f} | {q['punctuation_macro_f1']*100:.3f} | {q['segmentation_any_punctuation']['f1']*100:.3f} | {q['sentence_boundary']['f1']*100:.3f} | {delta} | {ci} |")
    lines += ["", "断句 F1 合并所有标点位置；句末 F1 合并 。？！；标点 micro/macro-F1 保留七种类型，不包含 O。Bootstrap 以匿名文档组为单位进行 2,000 次配对重采样，未作多重比较校正。", "",
              "![边界与内部质量](figures/03_boundary_effect.png)", "", "图 3：相同核心交界左右各 8 字符及内部区域相对核心输出的质量变化。每个原文字符只计分一次。", "",
              "## 4. 效率与资源成本", "",
              "![融合耗时](figures/04_fusion_latency.png)", "", "图 4：测试集端到端中位耗时及五次计时标准差。", "",
              "| 条件 | 中位耗时/s | 标准差/s | 相对整数核心耗时 | 相对概率控制耗时 | 相对串行加速 | 峰值 allocated/GiB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in test["conditions"]:
        cs=r.get("comparisons",{});ratio=lambda key:f"{cs[key]['latency_ratio']:.3f}" if key in cs else "—"
        speed=f"{cs['serial']['speedup']:.2f}×" if "serial" in cs else "1.00×"
        lines.append(f"| {r['condition']['id']} | {r['median_seconds']:.4f} | {r['stdev_seconds']:.4f} | {ratio('core')} | {ratio('prob-core')} | {speed} | {r['peak_allocated_bytes']/2**30:.3f} |")
    lines += ["", "![传输和聚合成本](figures/05_transfer_aggregation_cost.png)", "", "图 5：D2H 数据量以及概率校验、加权求和、归一化和 argmax 的主机时间。整数核心方案未单独测量概率聚合，因此该项为零；其完整解码成本包含在端到端耗时中。", "",
              "端到端计时包含字符编码、构窗、排序、padding、传输、模型、概率计算、聚合、argmax 和文本恢复，排除模型/数据加载、文件校验、质量计分、指纹与缓存写出。所有概率模式保留相同的窗口概率数组，控制保留数组的成本。GPU 前向与概率计算事件分别记录；D2H 等待时间含此前 GPU 工作的同步等待，不与事件时间直接相加。", "",
              "## 5. 实用性判定", "",
              "预设工程判定要求：相对整数核心输出，测试总体 micro-F1 增益的配对区间下界大于零，且中位耗时增加不超过 15%。该判定同时考察质量与成本，不以开发集最高分单独决定是否采用。", "",
              "| 融合条件 | 质量区间下界>0 | 耗时增幅≤15% | 达到预设条件 |", "| --- | --- | --- | --- |"]
    for r in test["conditions"]:
        if r["condition"]["mode"] not in ("uniform","weighted"):continue
        comp=r["comparisons"]["core"];q=comp["paired_95ci_pp"][0]>0;t=comp["latency_ratio"]<=dev["config"]["practical_slowdown_limit"]
        lines.append(f"| {r['condition']['id']} | {'是' if q else '否'} | {'是' if t else '否'} | {'是' if q and t else '否'} |")
    lines += ["", "## 6. 核验与复现", "",
              "所有条件的重复预测、概率指纹和窗口对齐信息保存在原始结果中。核心概率控制须与整数核心输出完全一致；各融合方式须使用相同概率，避免将模型数值变化混入融合效果。独立核验从保存的概率缓存重算加权结果、覆盖次数和权重和，并核对字符级指标及配对区间。", "",
              "```bash", "cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2", "uv sync --locked", "uv run --locked --offline pytest -q",
              "uv run --locked --offline python scripts/fusion_study.py development", "uv run --locked --offline python scripts/fusion_study.py test",
              "uv run --locked --offline python analysis/fusion_report.py", "uv run --locked --offline python analysis/audit_fusion_study.py", "```", "",
              "已完成的实验阶段、图表报告及核验记录拒绝覆盖。重新实验须复制配置并修改 experiment，两个阶段均使用 --config 指定；图表脚本以 --experiment 指定新编号。", "",
              f"完整指标与原始计时为 [development.json](development.json)、[test.json](test.json)，汇总表为 [development.csv](development.csv)、[test.csv](test.csv)，参数选择见 [selection.json](selection.json)。详细预测、窗口、概率和聚合缓存位于 code/outputs/task2/{experiment}/，运行源码快照位于 code/outputs/task2/source-snapshots/{experiment}/。图像提供 300 dpi PNG、SVG 和 PDF。", "",
              "本实验只利用重叠窗口已有局部信息，不能恢复单个窗口之外的完整依赖。匿名重建长文保留片段连接位置，但无法恢复完整古籍排版。五次重复推理不视为五次独立模型训练。", ""]
    if optimization:
        lines += ["## 7. 聚合实现优化", "",
                  "初始开发集测量发现，每窗重复将整个文档的 token 前缀列表转换为数组，等权方案也进行了多余的距离计算。优化后只转换窗口内的索引片段，等权方案直接使用全一权重；公式、模型、窗口和候选参数不变。优化后完整复测开发集，再锁定参数进行本轮测试。", "",
                  "两次开发集的 11 个条件逐一核对，预测和概率指纹完全一致。初始结果及源码快照保留在 cclue-window-fusion-v0.4，比较表见 [implementation-comparison.json](implementation-comparison.json)。两轮计时并非同时进行，因此图中同时展示相对各自整数核心基线的额外耗时。", "",
                  "![实现优化前后](figures/06_implementation_optimization.png)", "",
                  "图 6：开发集聚合实现优化前后的主机耗时和相对各自核心基线的端到端额外成本。虚线表示预设 15% 成本界限。", ""]
    path=directory/"experiment.md"
    with path.open("x",encoding="utf-8") as f:f.write("\n".join(lines))
    write_json(directory/"report-bundle.json",{"analysis_script":{"path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":digest(Path(__file__))},
              "input_files_sha256":{p.name:digest(p) for p in (directory/"development.json",directory/"test.json",directory/"selection.json")},
              "generated_files_sha256":{str(p.relative_to(directory)):digest(p) for p in [*paths,path]},"matplotlib":matplotlib.__version__,"numpy":np.__version__})
    print({"report":str(path.relative_to(ROOT)),"figure_files":len(paths)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--experiment",default="cclue-window-fusion-v0.4-optimized")
    args=parser.parse_args()
    if Path(args.experiment).name!=args.experiment or args.experiment in (".",".."):parser.error("Invalid experiment ID")
    render(args.experiment)
