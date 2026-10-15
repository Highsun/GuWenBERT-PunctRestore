# 任务一到任务二的交接说明

任务一交付版本化文件，任务二不导入 `task1/src/`，不调用任务一内部脚本作为运行时依赖。

## 文件与加载

微调模型位于 `models/task1/pd9801-guwenbert-v1/`，包括 `config.json`、`model.safetensors` 和 tokenizer 文件。可使用 `AutoTokenizer.from_pretrained(path, local_files_only=True)` 与 `AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)` 独立加载，不需要 `trust_remote_code`。

附加文件为 `labels.json` 和 `inference_contract.json`。标签顺序固定为 `O`、`，`、`。`、`？`、`！`、`；`、`：`、`、`，预测每个 Unicode 字符后的标点；O 不插入。逐字符交错插入标点，不能修改、丢失或重复正文字符。

上游预训练版本、数据划分、源文件及权重校验值见训练完成后生成的 `results.json` 和 `freeze-manifest.json`。先校验模型、tokenizer 和数据版本，原始及派生文件只读。

## 字符与 token 契约

不将整句直接交给 tokenizer。对每个字符分别执行 `tokenizer.encode(character, add_special_tokens=False)`，空编码用 `unk_token_id`，连接所有 subtoken 列表，前后添加 CLS/SEP。监督和解码均取每个字符的第一个 subtoken；后续 subtoken、CLS、SEP、padding 不产生输出。

训练窗最多 256 token，包含两个特殊 token。按 subtoken 实际长度切分，不能假定一个字符必定一个 token，也不能切开一个字符的 subtoken 列表。任务二可研究其他窗长，但不能超过模型配置的最大位置数，须记录与训练窗长的差异。

## 规范化与计分

行号、词性及组合词包装先解析，不属于正文。ASCII `,.?!;:` 映射到中文标点；空白及 Unicode Cc 控制字符删除并记录。数值内部 `.．/／`（左右相邻均为数字）及 `%％‰` 保留，其余非目标 Unicode P 类标点删除并计数。该策略不恢复原始排版或完整引号括号。

连续目标标点只生成一个标签：句末 `。？！` 优先于非句末标点，同优先级保留最后一个。开头没有可附着字符的标点删除并记录。任务二须沿用同一目标定义；改变规范化规则时另建数据版本，明确区分实验。

按字符位置计分，固定七类标点 micro/macro-F1；含 O 准确率单列。句末边界将 `。？！` 合并后计二元 precision/recall/F1。零分母计 0，macro 固定七类等权；不同标点错分产生相应 FP 和 FN，漏判为 O 产生 FN。窗口尾保留原文标签，不强制句末。

任务二可以只读复用 `data/splits/task1/pd9801-v1/test.jsonl`，但不能用它重新选择任务一模型；其新增长文本集合和运行结果另存。任务一逐窗预测和字符区间索引保存在本地 `outputs/task1/`。
