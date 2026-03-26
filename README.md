# Multimodal-Stock-Forecasting
**多模态股票预测框架 | 图像模态 + 时间序列模态融合**

本项目基于股票K线图，使用MAE/ViT模型进行图像特征提取与对比实验，最终用于多模态股票收益率预测。

---

## 已完成工作
1. 股票混合图与分图生成
2. 基于MAE和ViT的图像特征提取
3. 特征质量对比（方差、相似度、MSE、MAE）
4. 最优特征方案验证

---

## 实验结论
| 特征类型 | 方差 | 相似度 | MSE | MAE |
|---------|------|--------|-----|-----|
| 分图-ViT | 2.4979 | 0.9793 | 0.0047 | 0.0469 |
| 分图-MAE | 0.3508 | 0.9947 | 0.0046 | 0.0467 |
| 混合图-ViT | 2.6066 | 0.9767 | 0.0006 | 0.0163 |
| **混合图-MAE** | **2.6171** | **0.9708** | **0.0006** | **0.0167** |

**最优方案：混合图 + MAE 特征提取**

---

## GitHub 已上传内容
### code/
- 混合图绘制：`mix_generate_figure.ipynb`
- 分图绘制：`generate_separate_figure.ipynb`
- MAE特征提取：`mae_features_npz.ipynb`
- ViT特征提取：`vit_features_npz.ipynb`
- 分图特征提取：`f_s_mae.ipynb`、`f_s_vit.ipynb`
- 对比实验：`混合图vs分图.ipynb`、`mae_vs_vit.ipynb`、`f_s_vit_vs_mae.ipynb`
- 模型下载脚本：`hf_download.ipynb`

### data/
- 原始股票数据：`日个股数据2.0.csv`

---

## 未上传大文件（待发布至HuggingFace）
由于文件体积限制，以下内容未上传至GitHub，未来将统一发布在HuggingFace等数据集平台：
1. **models/**：MAE预训练模型权重（`model.safetensors`、`config.json`）
2. **figures/**：混合图、分图图像数据
3. **results/**：提取完成的npz特征文件、实验对比图`mix_vs_split.png`

---

## 完整复现步骤
以下步骤可完整复现本项目所有图像、特征与对比实验结果：

### 1. 下载MAE模型
- 代码：`code/hf_download.ipynb`
- 输入：无（从huggingface-mirror自动下载）
- 输出：`models/new_mae_model/`
- 功能：获取预训练MAE模型权重，为后续特征提取提供模型支撑

### 2. 生成股票混合图
- 代码：`code/mix_generate_figure.ipynb`
- 输入：`data/日个股数据2.0.csv`
- 输出：`figures/`
- 功能：生成包含K线、成交量热力图与均线指标的混合图（时间窗口为60天），每支股票对应一个文件夹存储多张时序图像

### 3. 生成股票分图
- 代码：`code/generate_separate_figure.ipynb`
- 输入：`data/日个股数据2.0.csv`
- 输出：`figure_separate/`
- 功能：生成K线、成交量、技术指标分离的分图数据，每只股票对应独立存储

### 4. 混合图特征提取
#### MAE特征提取
- 代码：`code/mae_features_npz.ipynb`
- 输入：`figures/` + `models/new_mae_model/`
- 输出：`results/mae_new_npz_features/`
- 标签：收益率（已正确标注）

#### ViT特征提取
- 代码：`code/vit_features_npz.ipynb`
- 输入：`figures/`
- 输出：`results/vit_new_npz_features/`

### 5. 分图特征提取
#### MAE特征提取
- 代码：`code/f_s_mae.ipynb`
- 输入：`figure_separate/`
- 输出：`results/f_s_mae_features/`

#### ViT特征提取
- 代码：`code/f_s_vit.ipynb`
- 输入：`figure_separate/`
- 输出：`results/f_s_vit_features/`

### 6. 特征对比实验
#### 混合图 vs 分图对比
- 代码：`code/混合图vs分图.ipynb`
- 输入：所有提取完成的特征文件
- 输出：方差、相似度、MSE、MAE四项指标对比结果
- 结论：混合图-MAE特征表现最优

#### 分图内部对比（ViT vs MAE）
- 代码：`code/f_s_vit_vs_mae.ipynb`
- 功能：验证分图场景下不同模型的特征质量差异

#### 混合图内部对比（MAE vs ViT）
- 代码：`code/mae_vs_vit.ipynb`
- 功能：验证混合图场景下不同模型的特征质量差异

---

## 最终产出
- 股票混合图与分图图像数据
- MAE/ViT提取的图像特征（npz格式）
- 4种特征组合的质量对比指标
- **最优特征方案：混合图 + MAE**

---

## 未来工作
1. 时间序列模态特征提取
2. 多模态特征融合
3. 收益率预测模型训练与回测
4. 数据集与模型权重发布至HuggingFace

---

## 说明
GitHub仓库仅存放代码与项目说明，图像、特征、模型权重等大体积文件将在HuggingFace平台单独发布。
