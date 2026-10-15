# CCLUE 古文标点数据

当前仅保留 CCLUE 的固定版本原始文件和任务一派生数据；旧语料及派生文件已经移除。

- `raw/cclue/<commit>/`：官方 train/dev/test CoNLL、加载器、构建 notebook、README、LICENSE 和评测脚本。下载后保留字节，不执行上游脚本。
- `manifest.json`：仓库、固定提交、文件路径、大小、下载链接及 SHA-256，纳入 Git。
- `processed/task1/cclue-punc-v2/`：处理摘要与移除训练记录的审计。
- `splits/task1/cclue-punc-v2/`：从原生字符/标签构造的 train/validation/test JSONL 和校验清单。

从 `task1/` 执行 `uv run --locked python scripts/run.py fetch-data` 下载，随后执行 `prepare`。重建当前数据版本应得到相同校验值；改变来源、划分或处理规则须另建版本。

五列含义是 character、seg_tags、punc_tags、quote_tags、book_tags。标签附着在标点前的字符，直接采用 punc_tags，`B-\` 是顿号。训练不转码、不改字符、不重新推断标点；quote/book 标签保留，但本轮不训练这两项任务。

官方非空记录数为 26,932/4,074/3,992。清理后训练 26,890 条；官方开发和测试集顺序及内容保留。移除训练集中的 16 条与留出集相同正文、26 条其余训练重复正文；开发和测试内部各保留 2 条重复正文。只做精确正文去重，不保证近重复完全消除。

来源、原生标签、实际文件统计及划分局限见 [核验记录](../docs/corpus-inventory.md)。Git 仅跟踪数据说明、元数据清单及目录标记，原始和派生语料不上传。CCLUE 仓库标示 Apache 2.0，项目 MIT 许可证不替代上游授权。
