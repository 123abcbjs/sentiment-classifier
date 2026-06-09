# 四分类中文评论情感分析

这是一个中文评论文本分类实验项目。项目使用 DeepSeek 辅助生成并标注评论，
将评论分为正面、中性、负面和反讽四类，并比较 TF-IDF 基线与中文 BERT 模型。

## 类别与评价对象

- 情感类别：`positive`、`neutral`、`negative`、`sarcasm`
- 评价对象：剧情、画质、配音、节奏、角色
- 目标规模：2,000 条，每个“情感 × 评价对象”组合 100 条

## 环境

```powershell
conda activate pytorch
pip install -r requirements.txt
```

项目读取用户环境变量 `DEEPSEEK_API_KEY`，也支持在项目目录创建 `.env`。
仓库已包含本次实验使用的清洗数据和固定划分，因此复现实验时可跳过数据生成步骤。

## 运行流程

生成数据，脚本会按已有数据断点续跑：

```powershell
python generate_data.py
```

清洗、质检并生成固定的数据划分：

```powershell
python validate_data.py
```

训练 TF-IDF 基线：

```powershell
python train_baseline.py
```

微调中文 BERT：

```powershell
python train_bert.py
```

预测单条评论：

```powershell
python predict.py "这剧情真精彩，不开二倍速都对不起导演"
python predict.py "画面清晰，色彩也很舒服" --model bert
```

## 输出

- `data/comments.csv`：清洗后的完整数据
- `data/review_sample.csv`：供人工抽查的 200 条样本
- `data/quality_report.json`：数据质量统计
- `data/splits/`：固定训练、验证和测试集
- `models/`：训练后的模型
- `outputs/`：指标、混淆矩阵与错误样本

## 本次实验结果

本次实验使用固定随机种子和 `1600/200/200` 的训练、验证、测试划分。

| 模型 | 测试集 Accuracy | 测试集 Macro-F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 0.7900 | 0.7891 |
| bert-base-chinese | 0.9250 | 0.9246 |

BERT 在中性和负面评论上的 F1 均为 `0.9412`，反讽类别 F1 为 `0.9109`。
错误样本显示，模型仍容易混淆隐含反讽、缺少明确情感词的负面评论，以及简短的客观描述。

## 项目局限

- 数据由 DeepSeek 辅助标注，不代表真实平台评论。
- 反讽表达边界模糊，需要人工抽样复核。
- 测试集与训练集来自相同的数据生成流程，指标可能高于真实业务场景。
- 当前指标来自一次固定划分实验，尚未进行交叉验证或真实平台数据外部测试。
