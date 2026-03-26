# Multimodal-Stock-Forecasting
**多模态股票预测框架 | 图像模态 + 时间序列模态融合**

本项目基于股票K线图，使用 MAE / ViT 模型进行图像特征提取与对比实验，最终用于多模态股票收益率预测。

---

## 📌 已完成工作
1. 股票混合图 / 分图生成
2. 基于 MAE 和 ViT 的图像特征提取
3. 特征质量对比（方差、相似度、MSE、MAE）
4. 最优特征方案验证

---

## 🔎 实验结论
| 特征类型 | 方差 | 相似度 | MSE | MAE |
|---------|------|--------|-----|-----|
| 分图-ViT | 2.4979 | 0.9793 | 0.0047 | 0.0469 |
| 分图-MAE | 0.3508 | 0.9947 | 0.0046 | 0.0467 |
| 混合图-ViT | 2.6066 | 0.9767 | 0.0006 | 0.0163 |
| **混合图-MAE** | **2.6171** | **0.9708** | **0.0006** | **0.0167** |

✅ **最优方案：混合图 + MAE 特征提取**

---

## 📁 当前 GitHub 已上传内容
### code/
- 混合图绘制：`mix_generate_figure.ipynb`
- 分图绘制：`generate_separate_figure.ipynb`
- MAE 特征提取：`mae_features_npz.ipynb`
- ViT 特征提取：`vit_features_npz.ipynb`
- 分图特征提取：`f_s_mae.ipynb`、`f_s_vit.ipynb`
- 对比实验：`混合图vs分图.ipynb`、`mae_vs_vit.ipynb`、`f_s_vit_vs_mae.ipynb`
- 模型下载脚本：`hf_download.ipynb`

### data/
- 原始股票数据：`日个股数据2.0.csv`

---

## 📦 未上传大文件（待发布至 HuggingFace）
由于文件体积过大，以下内容**未上传至 GitHub**，未来会统一发布在 HuggingFace 等数据集平台：

1. **models/**
   - MAE 预训练模型权重
   - 包括：`model.safetensors`、`config.json`

2. **figures/**
   - 混合图（K线+成交量+均线）
   - 分图（单独K线/量/指标图）

3. **results/**
   - MAE / ViT 提取的 npz 图像特征
   - 实验对比图 `mix_vs_split.png`

---

## 🚀 未来计划
- 时间序列模态特征提取
- 多模态特征融合
- 模型训练与回测
- 数据集与模型权重发布至 HuggingFace

---

## 📬 说明
GitHub 仅存放**代码与说明**，**图像、特征、模型权重等大文件**将在 HuggingFace 单独发布。
