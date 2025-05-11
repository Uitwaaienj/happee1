# happee1
# 工业轴承TE数据集故障检测系统

本项目使用PyTorch实现了多种深度学习模型，用于工业轴承TE数据集的故障检测。系统支持RNN、GRU、LSTM、CNN-LSTM和CNN-GRU等多种模型架构。

## 项目结构

```
.
├── data_processor.py    # 数据处理模块
├── model_factory.py     # 模型工厂和模型定义
├── trainer.py          # 训练和评估模块
├── main.py             # 主程序
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- 其他依赖见 requirements.txt

## 安装

1. 克隆项目
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python main.py --model_type lstm --batch_size 32 --window_size 50 --hidden_size 128 --num_layers 2 --learning_rate 0.001 --num_epochs 50
```

参数说明：
- `--model_type`: 模型类型，可选值：rnn, gru, lstm, cnn_lstm, cnn_gru
- `--batch_size`: 批次大小
- `--window_size`: 时间窗口大小
- `--stride`: 滑动窗口步长
- `--hidden_size`: 隐藏层大小
- `--num_layers`: RNN层数
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数

### 输出结果

训练过程中会输出：
1. 每个epoch的训练和验证损失、准确率
2. 训练结束后的详细评估指标（准确率、精确率、召回率、F1分数）
3. 混淆矩阵图
4. 训练历史图（损失和准确率曲线）
5. 模型参数和评估结果保存在results目录下

## 模型说明

1. RNN：基础循环神经网络
2. GRU：门控循环单元
3. LSTM：长短期记忆网络
4. CNN-LSTM：卷积神经网络与LSTM的组合
5. CNN-GRU：卷积神经网络与GRU的组合

## 评估指标

系统提供以下评估指标：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-Score）
- 混淆矩阵（Confusion Matrix） 
>>>>>>> master
