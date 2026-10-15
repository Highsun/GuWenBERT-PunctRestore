# 项目协作约定

- 项目根目录及唯一 Git 仓库根目录为本文件所在的 `code/`；所有项目代码、配置、缓存和输出均放在此目录内。
- 用户已授权实施任务一，包括数据制备、模型下载、微调、独立测试和实验报告。任务二仍仅保留初始化骨架，不实施其滑窗与批量推理逻辑。
- Python 使用 uv 和 `.venv/`，版本约定见 `.python-version`；新增依赖更新 `pyproject.toml` 和 `uv.lock`。
- 任务一和任务二分别位于 `task1/`、`task2/`，各自拥有源码、配置、脚本、测试、uv 项目、锁文件、虚拟环境和报告。不建共享锁文件的 uv workspace，不直接跨任务导入源码。根目录环境仅用于初始化。
- 任务一完成后按 `docs/task-boundaries.md` 冻结代码和实验依据。任务二只读任务一的冻结模型、标签及数据版本，不修改任务一的源码、配置、依赖锁文件、报告和产物；必要修正另建任务一新版本。当前尚未完成或冻结任何任务。
- 换源后的六份原始语料保存在 `data/raw/`，均为 UTF-8。不得覆盖或静默转码；编码、控制字符审计和校验值见 `docs/corpus-inventory.md` 与 `data/manifest.json`。派生数据写入 `data/processed/`，划分结果写入 `data/splits/`，并记录处理规则。
- 不提交语料、虚拟环境、缓存、模型权重、凭据和生成结果。模型写入 `models/task1/` 或 `models/task2/`，运行输出写入 `outputs/task1/` 或 `outputs/task2/`；报告和精简的实验元数据写入对应任务的 `reports/` 并纳入 Git。
- 用户已授权初始化后的首次提交和同步，指定 `origin` 为 `https://github.com/Highsun/GuWenBERT-PunctRestore.git`。后续推送须处于用户授权范围，不改用其他远程，也不强制推送。
- 文档以中文为主，实验结论必须有真实运行证据。没有执行的工作须明确标为待办。
