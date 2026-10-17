"""Render the selective-review study using completed measurements and cache audit."""
from pathlib import Path
import argparse
import os
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guwenbert_task2.common import ROOT,TASK,digest,read_json,write_json
import numpy as np


def render(experiment):
    directory=TASK/"reports"/experiment
    if (directory/"report-bundle.json").exists():raise FileExistsError("Report already completed")
    dev,test,selection,audit=[read_json(directory/f"{s}.json") for s in ("development","test","selection","validation")]
    assert audit["passed"] and test["selection_sha256"]==digest(directory/"selection.json")
    dr={r["condition"]["id"]:r for r in dev["conditions"]};tr={r["condition"]["id"]:r for r in test["conditions"]}
    chosen=selection["selected"]["condition"]["id"];random=tr[chosen]["comparisons"]["random"]["reference"]
    os.environ.setdefault("MPLCONFIGDIR",str(TASK/".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":.2,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42,"savefig.dpi":300})
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#666666","#E69F00"]
    names={"baseline":"Linear fusion","score-only":"Score only","all-review":"All cuts","serial":"Serial"}
    names.update({f"{s}-{q}":f"{s.title()} {q}%" for s in dev["config"]["selectors"] for q in (10,25,50)})
    figures=[];fd=directory/"figures";fd.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ("png","svg","pdf"):
            p=fd/f"{name}.{ext}"
            if p.exists():raise FileExistsError(p)
            fig.savefig(p,bbox_inches="tight",facecolor="white");figures.append(p)
        plt.close(fig)
    def f1(r,region="overall"):return 100*r["quality"][region]["punctuation_micro"]["f1"]
    def med(r,k):return statistics.median(s.get(k,0) for s in r["measurements"])

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for region,ax in zip(("overall","window_edges"),axes):
        for color,s in zip(colors,dev["config"]["selectors"]):
            rs=[dr[f"{s}-{int(q*100)}"] for q in dev["config"]["fractions"]]
            ax.plot([100*r["condition"]["fraction"] for r in rs],[f1(r,region) for r in rs],"o-",label=s.title(),color=color)
        ax.axhline(f1(dr["baseline"],region),color=colors[4],ls="--",label="Linear fusion")
        ax.axhline(f1(dr["all-review"],region),color=colors[3],ls=":",label="All cuts")
        ax.set(xlabel="Requested review fraction per document (%)",ylabel="Punctuation micro-F1 (%)",
               title="Overall" if region=="overall" else "Within 8 characters of first-pass cuts")
        ax.legend(fontsize=8)
    save(fig,"01_development_quality")

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for color,s in zip(colors,dev["config"]["selectors"]):
        rs=[dr[f"{s}-{int(q*100)}"] for q in dev["config"]["fractions"]]
        x=[r["review_windows"] for r in rs]
        axes[0].plot(x,[100*(r["comparisons"]["baseline"]["latency_ratio"]-1) for r in rs],"o-",color=color,label=s.title())
        axes[1].plot(x,[r["median_seconds"] for r in rs],"o-",color=color,label=s.title())
    axes[0].axhline(25,color=colors[4],ls="--",label="25% cost limit")
    for ax in axes:ax.set_xlabel("Actual extra review windows");ax.legend(fontsize=8)
    axes[0].set(ylabel="End-to-end overhead vs linear fusion (%)",title="Development cost")
    axes[1].set(ylabel="End-to-end median latency (s)",title="Matched review-window budgets")
    save(fig,"02_development_cost")

    rs=[tr[n] for n in ("score-only",chosen,random,"all-review","serial")]
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    for i,r in enumerate(rs):
        comp=r["comparisons"]["baseline"];lo,hi=comp["paired_95ci_pp"]
        ax.hlines(i,lo,hi,color=colors[i],lw=2);ax.scatter(comp["delta_micro_f1_pp"],i,color=colors[i],zorder=3)
    ax.axvline(0,color=colors[4],ls="--");ax.set_yticks(range(len(rs)),[names[r["condition"]["id"]] for r in rs]);ax.invert_yaxis()
    ax.set(xlabel="Micro-F1 change vs linear fusion (pp)",title="Test set: paired document bootstrap, 95% intervals")
    save(fig,"03_test_effect")

    rs=list(test["conditions"]);x=np.arange(len(rs));labels=[names[r["condition"]["id"]].replace(" ","\n",1) for r in rs]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    axes[0].bar(x,[r["median_seconds"] for r in rs],yerr=[r["stdev_seconds"] for r in rs],capsize=3,color=colors)
    axes[0].set_yscale("log");axes[0].set(ylabel="End-to-end latency (s, log scale)",title="Test latency (median; error = SD)")
    axes[1].bar(x,[f1(r) for r in rs],color=colors);ys=[f1(r) for r in rs]
    axes[1].set(ylabel="Punctuation micro-F1 (%)",ylim=(min(ys)-.08,max(ys)+.08),title="Test quality (expanded axis)")
    for ax in axes:ax.set_xticks(x,labels,fontsize=8)
    for i,r in enumerate(rs):
        axes[0].annotate(f"{r['median_seconds']:.3f}",(i,r['median_seconds']),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
        axes[1].annotate(f"{f1(r):.3f}",(i,f1(r)),xytext=(0,5),textcoords="offset points",ha="center",fontsize=8)
    save(fig,"04_test_tradeoff")

    rs=[r for r in test["conditions"] if r["condition"]["mode"]!="serial"];x=np.arange(len(rs));bottom=np.zeros(len(rs))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for color,key,label in zip(colors,("first_pass_seconds","selection_seconds","second_pass_seconds","review_apply_seconds"),
                               ("First pass + fusion","Scoring + selection","Review pass","Apply review")):
        y=np.array([med(r,key) for r in rs]);axes[0].bar(x,y,bottom=bottom,color=color,label=label);bottom+=y
    axes[0].set(ylabel="Sum of stage medians (s)",title="Test host-observed stages");axes[0].legend(fontsize=8)
    axes[1].bar(x,[r["first_forward_calls"] for r in rs],label="First pass",color=colors[0])
    axes[1].bar(x,[r["review_forward_calls"] for r in rs],bottom=[r["first_forward_calls"] for r in rs],label="Review pass",color=colors[1])
    axes[1].set(ylabel="Batched model forward calls",title="Batch = 16");axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,[names[r["condition"]["id"]].replace(" ","\n",1) for r in rs],fontsize=8)
    save(fig,"05_stage_cost")

    ids=[chosen,random,"all-review"];cs=[audit["changes"][f"test/{n}"] for n in ids];x=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for dx,k,label,color in [(-.17,"corrected_characters","Errors corrected",colors[0]),(.17,"newly_incorrect_characters","New errors",colors[1])]:
        axes[0].bar(x+dx,[c[k] for c in cs],width=.32,label=label,color=color)
    axes[0].set(ylabel="Character labels",title="Test review corrections and regressions");axes[0].legend(fontsize=8)
    axes[1].bar(x,[100*c["baseline_errors_in_reviewed_bands"]/c["reviewed_characters"] for c in cs],color=colors[:3])
    axes[1].set(ylabel="Baseline label-error rate in reviewed bands (%)",title="Post-selection diagnostic")
    for ax in axes:ax.set_xticks(x,[names[n].replace(" ","\n",1) for n in ids],fontsize=8)
    save(fig,"06_review_diagnostics")

    lines=["# 任务二：选择性局部复核实验","","实验日期：2026-10-17。实验编号：`"+experiment+"`。","",
        "## 1. 研究目的","",
        "本实验研究在固定窗口并行推理及线性概率融合后，能否通过少量第二遍推理改善窗口交界的标点预测，并将额外耗时控制在 25% 以内。模型沿用任务一冻结的通用中文 BERT，不进行训练。","",
        "## 2. 算法与实验设置","",
        "第一遍总输入预算为 256 token，核心区 192 token，每侧上下文 31 token，batch=16；保留概率并采用线性中心权重融合。候选位置为第一遍核心区之间的切点。第二遍以切点为中心构建最多 256 token 的输入，仅用其预测替换切点左右各 8 个字符的标签。复核窗口彼此独立，按长度排序后批量推理；未选中的字符保留第一遍预测。字符的全部 subtoken 始终位于同一输入窗口。","",
        "不确定性评分为交界局部区间内 1−(最大概率−次大概率) 的均值；分歧评分为该区间内不同覆盖窗口的 argmax 标签不一致的字符比例，分歧相同时优先不确定性较高者。随机对照使用固定种子。每文档分别选择 floor(q×候选数) 个切点，因此实际全局比例通常小于名义预算 q。第二遍区间如重叠，每个字符使用 token 距离最近切点对应的预测，等距时选较早切点。","",
        f"开发集包含 {dev['documents']} 个匿名文档组、{dev['characters']:,} 字符，测试集包含 {test['documents']} 组、{test['characters']:,} 字符。开发集比较三种筛选方法及 10%、25%、50% 三个名义预算，并设置线性融合基线、仅评分控制、全量交界复核和串行句号推进，共 13 个条件。每条件预热一次，再以随机顺序正式测量五轮。测试集在开发集锁定选择后运行六个条件，每个条件同样测量五轮。","",
        "选择规则为：在重复预测一致、耗时不超过基线 1.25 倍的不确定性或分歧方案中，优先总体标点 micro-F1；与最高分相差不超过 0.02 个百分点时优先低耗时。随机对照和全量复核不参与选择。测试前保存并校验参数、配置与源码指纹。","",
        "## 3. 开发集结果","","![开发集质量](figures/01_development_quality.png)","",
        "图 1：总体及第一遍交界左右各 8 字符的标点 micro-F1。全量复核是诊断对照，不预设为质量上界。","",
        "| 条件 | micro-F1/% | 交界 micro-F1/% | 中位耗时/s | 复核窗口数 | 相对基线耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in dev["conditions"]:lines.append(f"| {r['condition']['id']} | {f1(r):.4f} | {f1(r,'window_edges'):.4f} | {r['median_seconds']:.4f} | {r['review_windows']} | {r['comparisons']['baseline']['latency_ratio']:.3f}× |")
    lines += ["",f"开发集锁定方案为 **{chosen}**。成本约束满足情况：{'满足' if selection['selected']['cost_limit_satisfied'] else '不满足，按预注册规则退回质量优先选择'}。","",
        "![开发集成本](figures/02_development_cost.png)","","图 2：不同实际复核窗口数量对应的额外耗时与端到端中位耗时。虚线为 25% 额外成本界限。","",
        "## 4. 独立测试结果","",
        "| 条件 | micro-F1/% | macro-F1/% | 断句 F1/% | 句末 F1/% | 交界 micro-F1/% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in test["conditions"]:
        q=r["quality"]["overall"];lines.append(f"| {r['condition']['id']} | {f1(r):.4f} | {100*q['punctuation_macro_f1']:.4f} | {100*q['segmentation_any_punctuation']['f1']:.4f} | {100*q['sentence_boundary']['f1']:.4f} | {f1(r,'window_edges'):.4f} |")
    lines += ["","断句 F1 将所有标点位置合并；句末 F1 合并 。？！；标点 micro/macro-F1 保留七类标点，不计 O。交界区始终使用第一遍相同切点定义，避免因选区变化改变评价区域。","",
        "![测试效果量](figures/03_test_effect.png)","","图 3：相对线性融合的 micro-F1 差值及 2,000 次文档组配对 bootstrap 的 95% 区间。区间未作多重比较校正。","",
        "| 比较 | micro-F1 差值/百分点 | 配对 95% 区间 |",
        "| --- | ---: | --- |"]
    for name in (chosen,random,"all-review","serial"):
        c=tr[name]["comparisons"]["baseline"];lo,hi=c["paired_95ci_pp"];lines.append(f"| {name} − baseline | {c['delta_micro_f1_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    c=tr[chosen]["comparisons"]["random"];lo,hi=c["paired_95ci_pp"]
    lines.append(f"| {chosen} − {random} | {c['delta_micro_f1_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["","## 5. 工程成本","","![测试权衡](figures/04_test_tradeoff.png)","",
        "图 4：端到端中位耗时与标点 micro-F1。耗时轴使用对数刻度，误差线为五轮计时标准差；质量轴局部放大。","",
        "| 条件 | 中位耗时/s | 标准差/s | 相对基线增幅/% | 对串行加速 | 复核窗口/候选 | 复核字符 | 峰值 allocated/GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in test["conditions"]:
        lines.append(f"| {r['condition']['id']} | {r['median_seconds']:.4f} | {r['stdev_seconds']:.4f} | {100*(r['comparisons']['baseline']['latency_ratio']-1):+.2f} | {r['comparisons']['serial']['speedup']:.2f}× | {r['review_windows']}/{r['candidate_count']} | {r['reviewed_characters']} | {r['peak_allocated_bytes']/2**30:.3f} |")
    lines += ["","![分阶段成本](figures/05_stage_cost.png)","",
        "图 5：第一遍与融合、评分筛选、第二遍及标签替换的主机计时，以及两遍 batch 前向调用数。各阶段中位数之和不等于端到端耗时中位数；图中阶段不含全部编码、构窗和恢复文本成本。主机阶段时间包含相应 GPU 等待，不能再与 CUDA 事件时间相加。","",
        "端到端计时包含编码、构窗、模型前向、概率融合、评分筛选、批量复核、标签替换和文本恢复，排除模型及数据加载、来源校验、质量计分、指纹计算和文件写出。缓存和模型保持预热状态。跨阶段计时受运行时 GPU 状态影响，工程比较以本轮同条件对照为准。","",
        "## 6. 复核诊断与采用判定","","![复核诊断](figures/06_review_diagnostics.png)","",
        "图 6：复核修正的错误、引入的新错误，以及被选区间的基线标签错误率。以上诊断在选区确定后使用金标准计算，评分筛选过程不读取金标准。","",
        "| 条件 | 选区基线错误数 | 复核后错误数 | 修正字符 | 新错误字符 | 改变标签数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in (chosen,random,"all-review"):
        c=audit["changes"][f"test/{name}"];lines.append(f"| {name} | {c['baseline_errors_in_reviewed_bands']} | {c['errors_after_review']} | {c['corrected_characters']} | {c['newly_incorrect_characters']} | {c['changed_characters']} |")
    c=tr[chosen]["comparisons"]["baseline"];gain=c["paired_95ci_pp"][0]>0;cost=c["latency_ratio"]<=dev["config"]["practical_slowdown_limit"]
    lines += ["",f"预设采用条件要求相对线性融合的测试 micro-F1 增益区间下界大于零，且耗时增幅不超过 25%。本轮质量条件{'满足' if gain else '不满足'}，成本条件{'满足' if cost else '不满足'}，综合判定为**{'达到' if gain and cost else '未达到'}预设采用条件**。详细讨论见 [conclusions.md](conclusions.md)。","",
        "## 7. 复现与实验记录","",
        "```bash","cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2","uv sync --locked",
        "uv run --locked --offline pytest -q","uv run --locked --offline python scripts/review_study.py development",
        "uv run --locked --offline python scripts/review_study.py test","uv run --locked --offline python analysis/audit_review_study.py",
        "uv run --locked --offline python analysis/review_report.py","```","",
        "已完成的阶段和图表拒绝覆盖。重新实验应复制配置、修改 experiment，并为运行及核验脚本传入 --config，为绘图脚本传入 --experiment。先运行开发集，再运行测试集、独立核验和图表生成。","",
        f"独立核验通过缓存重新计算概率融合、候选评分、每文档预算、随机对照、复核归属和最终标签，重新计算混淆矩阵及配对区间。56 项单元测试通过，任务一和前期实验的保护文件保持一致。指标与计时见 [development.json](development.json)、[test.json](test.json)，参数见 [selection.json](selection.json)，核验见 [validation.json](validation.json)。详细预测与概率缓存在 code/outputs/task2/{experiment}/，源码快照在 code/outputs/task2/source-snapshots/{experiment}/。六幅图各提供 300 dpi PNG、SVG 与 PDF。","",
        "本实验只改变交界的局部观察位置，第二遍窗口仍受 BERT 上下文上限限制；不能解决任意距离的全文指代。匿名长文由原始片段重建，拼接位置保留在数据中，其分布与完整古籍仍有差异。实验使用一个冻结模型，推理重复用于衡量计时和确定性，不代表多次训练。测试数据沿用前几阶段，当前方案虽在开发集锁定后评价，整条研究路线已参考前期测试结果；本轮属于连续探索，最终泛化能力仍需新增外部留出集验证。",""]
    p=directory/"experiment.md"
    with p.open("x",encoding="utf-8") as f:f.write("\n".join(lines))
    write_json(directory/"report-bundle.json",{"passed":True,"inputs_sha256":{str((directory/f'{s}.json').relative_to(ROOT)):digest(directory/f'{s}.json') for s in ("development","test","selection","validation")},
        "files_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [*figures,p]},"analysis_script":{"path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":digest(Path(__file__))}})
    print({"figures":len(figures),"report":str(p),"selected":chosen})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--experiment",default="cclue-selective-review-v0.5")
    render(parser.parse_args().experiment)
