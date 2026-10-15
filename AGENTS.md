# 项目协作约定

- 项目根目录及唯一 Git 仓库根目录为本文件所在的 `code/`；所有项目代码、配置、缓存和输出均放在此目录内。
- 当前仅获授权进行初始化。用户明确授权下一阶段后，再开展相应的数据处理、模型下载、训练、推理或测试，不自动扩大任务范围。
- Python 使用 uv 和 `.venv/`，版本约定见 `.python-version`；新增依赖更新 `pyproject.toml` 和 `uv.lock`。
- 换源后的六份原始语料保存在 `data/raw/`，均为 UTF-8。不得覆盖或静默转码；编码、控制字符审计和校验值见 `docs/corpus-inventory.md` 与 `data/manifest.json`。派生数据写入 `data/processed/`，划分结果写入 `data/splits/`，并记录处理规则。
- 不提交语料、虚拟环境、缓存、模型权重、凭据和生成结果。模型写入 `models/`，运行输出写入 `outputs/`。
- 用户已授权初始化后的首次提交和同步，指定 `origin` 为 `https://github.com/Highsun/GuWenBERT-PunctRestore.git`。后续推送须处于用户授权范围，不改用其他远程，也不强制推送。
- 文档以中文为主，实验结论必须有真实运行证据。没有执行的工作须明确标为待办。
