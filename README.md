# T5 问答生成微调项目

基于 Mengzi-T5-base 模型，在 DuReaderQG 数据集上微调实现**阅读理解问答生成**任务。

## 项目结构

```
.
├── dataloader.py   # 数据加载器，将 JSONL 数据转为模型输入格式
├── train.py        # 训练脚本，基于 HuggingFace Trainer 微调模型
├── predict.py      # 推理脚本，支持 Trainer 和 DataLoader 两种方式
└── data/
    └── DuReaderQG/
        ├── train.json   # 训练集
        ├── dev.json     # 验证集
        └── test.json    # 测试集
```

## 数据格式

每行为一条 JSON，包含三个字段：

```json
{"context": "...", "question": "...", "answer": "..."}
```

模型输入：`阅读理解: 内容:{context}, 问题:{question}` → 预测 `{answer}`

## 环境依赖

```
torch
transformers
numpy
tensorboard
```

## 使用方法

### 训练

```bash
python train.py
```

- 基座模型：`langboat/mengzi-t5-base`
- 训练输出保存到 `./results/mengzi-t5-base-finetuned`
- TensorBoard 日志写入 `./logs`

### 预测

```bash
python predict.py
```

默认使用 DataLoader + `model.generate()` 方式进行 beam search 推理，输出预测结果与真实标签对比。

## 训练配置

| 参数 | 值 |
|------|-----|
| 学习率 | 1e-5 |
| batch size | 3 |
| 最大训练步数 | 5000 |
| 评估间隔 | 每 50 步 |
| 最大输入长度 | 512 |
| 最大输出长度 | 128 |