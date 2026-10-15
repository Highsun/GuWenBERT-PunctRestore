# 任务一 v2 向任务二交接

默认输入为 `models/task1/cclue-bert-base-v2/`，古文基座对照为 `models/task1/cclue-guwenbert-v2/`。两个模型独立保存，不覆盖彼此。任务二须核对 `freeze-manifest.json` 中主实验、代码标签、模型和数据校验值，不导入任务一源码或调用其内部脚本。

## 独立加载和字符对齐

使用 `AutoTokenizer.from_pretrained(path, local_files_only=True)` 和 `AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)`，不需要 trust_remote_code。模型目录包含标准权重、config、tokenizer，以及 `labels.json`、`inference_contract.json`、`reference_predictions.json`。

标签 ID 顺序为 O、逗号、句号、问号、叹号、顿号、冒号、分号，与上游原生标签一致。预测为每个字符后的一个标点；O 不插入，逐字符交错输出，正文不丢失、不重复、不改写。旧版本的标签顺序不适用于新版本。

对每个 Unicode 字符分别执行 `tokenizer.encode(character, add_special_tokens=False)`；空编码用 UNK，连接所有 subtoken，前后添加 CLS/SEP。解码取每个字符第一个 subtoken，其余 subtoken、特殊 token、padding 不产生输出。这个规则与整句分词不同，必须独立复现，不能直接调用普通 token-classification pipeline 替代。

训练窗长为 256 token（含两个特殊 token），窗口按真实 subtoken 数装箱，不切开单个字符。两种基座有效上限均为 512 token，RoBERTa 配置中的 514 含位置偏移。任务二改变窗长须记录设置与训练条件的差异。

## 数据与目标定义

直接使用 CCLUE 的 tokens 与 punc_tags，训练不规范化原文。原生标签 B-\ 为顿号，标点附着在前一个字符上；quote/book 是另外的标签列，本轮不训练引号或书名号恢复。

单窗演示去掉七类中文标点、“”‘’《》〈〉、空白和 Cc 控制字符，保留其他字符。不将 ASCII 标点自动映射成中文，也不改繁简或数值符号。任务二质量评测须优先直接使用相同的已制备测试输入，而不是再次清洗标签。

任意目标标点断句将七类标点都视作边界；句末边界只包括 。？！；标点类型 F1 要求位置和类型都正确。三者分别报告。padding、特殊 token 和后续 subtoken 不计分，macro 固定七类等权，零分母计 0，不人为补窗口末尾句号。

## 验证与保存

`reference_predictions.json` 含四个导出一致性案例；已通过时，验证记录保存在各实验报告目录。案例不是独立人工标注测试集。模型权重和原始/派生数据不随 Git 标签备份，需另行保存。

任务二只读 `data/splits/task1/cclue-punc-v2/test.jsonl` 和模型，新的长文本集合、配置和结果放在任务二目录。该上游测试包含匿名文档组，不能将片段记录数当作完整文章数，也不能声称完全排除了近重复或预训练重叠。
