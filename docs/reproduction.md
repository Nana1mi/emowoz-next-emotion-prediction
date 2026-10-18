# 复现说明

## 环境

原实验运行于 Windows、Python 3.11.15、PyTorch 2.11.0+cu128、RTX 5080 Laptop GPU（约 16 GiB 显存），其他核心版本见根目录 `requirements.txt`。这是历史记录，不代表这些版本适用于所有硬件。PyTorch 请按 [官方安装说明](https://pytorch.org/get-started/locally/) 选择支持本机 GPU 的构建。

公开版保留全部训练 Python 文件原样；五个 PowerShell 入口只将个人机器的 Python 绝对路径改为当前环境 `python`，不改实验参数。应先激活目标环境，并从仓库根目录运行命令。

```powershell
conda create -n emowoz-project python=3.11
conda activate emowoz-project
# 根据设备安装 PyTorch 后：
python -m pip install -r requirements.txt
python run_experiments.py check
python run_experiments.py smoke --device cpu
```

依赖版本来自原机器记录；本次公开整理没有在全新环境重新完成所有训练。原始实验最早一轮未锁定模型 revision；后续 MiniLM 微调的记录 revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。脚本仍沿用原下载逻辑，因此未来下载不同 revision 可能带来复现差异。

## 顺序与依赖

下列是完整重建流程，不会因为阅读 README 而自动执行。每一步都应先检查输出与完成状态，再进入下一步；同名目录中已有结果时不要随意更改参数。GPU 微调中断后的恢复规则不同于原始轻量训练，见末节。

### 1. 原始冻结编码器实验

```powershell
python run_experiments.py prepare --out runs/full
python run_experiments.py encode --out runs/full --encode-batch 32
python run_experiments.py train --out runs/full --seeds 42 43 44 --epochs 15
python run_experiments.py summarize --out runs/full
```

`prepare` 从作者 Zenodo 发布页下载 `emowoz-multiwoz.json`、`emowoz-dialmage.json`、`data-split.json` 到 `data/raw/`。下载器检查分段范围、完整 JSON，并支持临时文件续传。也可自行下载这些文件放入 `data/raw/`。`encode` 缓存后续实验共同使用的句向量与处理后划分。模型与数据下载需要网络；仓库本身没有附带它们。

### 2. 类别不平衡与语义辅助目标

```powershell
python verify_controlled.py --device cpu
python run_controlled_experiments.py --out runs/controlled_v2 --reuse-from runs/full --route all --device cuda --seeds 42 43 44
```

### 3. 事后识别诊断

```powershell
python run_posthoc_diagnostic.py --source runs/full --reference runs/controlled_v2 --out runs/posthoc_diagnostic --device cuda
python analyze_error_cases.py
```

这些探针部分读取真实未来句，只用于解释前瞻预测的信息限制。不能把它们当作部署模型。`analyze_error_cases.py` 需要前两轮以及事后诊断的完整本地结果。

### 4. 可训练编码器强基线

```powershell
python verify_encoder_baselines.py --device cpu
python run_encoder_baselines.py --stage all --suite minilm --source runs/full --out runs/encoder_minilm_v1 --allow-download --precision auto --micro-batch 16 --accumulation 4 --eval-batch 32
```

默认 8 轮、早停 patience=3、encoder LR=2e-5、head LR=1e-3。先 seed 42 验证筛选，再进行三个种子的确认和测试。RoBERTa 入口虽然保留在代码中，**没有在本项目汇总中完成该系列正式实验**。

### 5. 情绪蒸馏与角色交互

```powershell
python verify_emotion_distillation.py
python run_emotion_distillation.py --stage all --source runs/full --reference runs/encoder_minilm_v1 --out runs/emotion_kd_v1 --teacher-device cpu
python verify_role_interaction.py
python run_role_interaction.py --stage all --source runs/full --reference runs/encoder_minilm_v1 --out runs/role_interaction_v1
```

教师在 CPU 上运行，不与学生同时占用 GPU。角色模型是在一次共享上下文编码后做分角色池化，不是各话语独立编码的状态空间模型。

### 6. 对话级统计审计

```powershell
python verify_statistical_robustness.py
python analyze_statistical_robustness.py --out audit-reproduced --bootstrap 10000 --permutations 10000 --random-seed 20261015
```

统计脚本只使用 CPU，但需要前述六轮的完整本地预测，以及原始数据和划分。只下载本仓库里的聚合 JSON，**无法重建逐对话 bootstrap 分布或置换 p 值**；需先完整重跑产生逐条预测。本仓库提供已完成审计的汇总值用于阅读核对，并不把聚合表伪装成重采样所需原始输入。

## 显存与恢复

原微调设置的 micro-batch=16、accumulation=4，有效批量为 64。显存需求还取决于精度、序列长度、驱动和其他进程，不能保证所有 16 GiB 设备都可直接运行。若需要减少 micro-batch，应在新输出目录运行；保持有效批量也不保证浮点计算与原结果逐位相同。不要同时运行多组 GPU 训练。

编码器、KD、角色训练的已完成 epoch 检查点可按脚本规则恢复：使用完全相同的命令与目录重跑。原始 `run_experiments.py train` 不支持优化器进度续训，同名结果可能重新训练。不要把新参数运行当成旧检查点的无缝延续。
