# OpenCLIP 项目结构与运行指南 / Project Structure & Run Guide

## 目录结构概览 / Directory Overview
- `src/open_clip/`: 核心库代码，包含模型构建、tokenizer、损失函数等实现。
- `src/open_clip_train/`: 训练脚本与数据流水线，封装分布式训练、评测逻辑。
- `docs/` 与 `tutorials/`: 文档与示例笔记本，展示用法与扩展案例。
- `scripts/`: 常用工具脚本，例如权重转换、评测辅助脚本。
- `tests/`: 单元测试与回归测试集合。
- `requirements*.txt`: 运行、训练、测试所需依赖列表。

## 核心模块说明 / Key Modules
- `factory.py`: 负责模型实例化、权重加载、预处理管线构建。
- `model.py` 与 `transformer.py`: 定义视觉/文本编码器、投影层以及联合前向逻辑。
- `tokenizer.py`: 提供OpenCLIP默认BPE分词器与HuggingFace分词器封装。
- `loss.py`: 实现对比损失、CoCa联合损失等训练目标。
- `open_clip_train/main.py`: 命令行入口，整合参数解析、数据加载与训练循环。

## 快速上手步骤 / Getting Started
1. **安装依赖 / Install dependencies**
   ```bash
   pip install -e .
   pip install -r requirements-training.txt  # 可选：完整训练依赖
   ```
2. **下载或指定权重 / Acquire checkpoints**
   - 使用 `open_clip.create_model_and_transforms` 自动下载官方权重。
   - 或者通过 `--pretrained` 参数指定本地路径。
3. **运行推理 / Run inference**
   ```python
   import open_clip
   model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
   model.eval()
   ```
   使用 `preprocess` 处理图像后，调用 `model.encode_image` / `model.encode_text` 获取特征。
4. **启动训练 / Launch training**
   ```bash
   python -m open_clip_train.main \
       --model ViT-B-32 --pretrained laion400m_e32 --train-data <data_path> \
       --warmup 2000 --batch-size 256 --epochs 10
   ```
   根据数据类型选择 `--train-data` 格式（WebDataset/CSV/JSON）。
5. **零样本评测 / Zero-shot evaluation**
   ```bash
   python -m open_clip_train.zero_shot --model ViT-B-32 --pretrained laion2b_s34b_b79k --imagenet-val <path>
   ```

## 调试与扩展建议 / Tips for Extension
- 修改或扩展tokenizer时，请参考 `tokenizer.py` 中 `SimpleTokenizer` 与 `HFTokenizer` 的注释。
- 新增模型配置后，在 `src/open_clip/model_configs` 中补充JSON文件，并在 `factory.py` 注册。
- 训练大模型时建议开启混合精度 (`--amp`) 与梯度累积 (`--accum-freq`) 以节省显存。

