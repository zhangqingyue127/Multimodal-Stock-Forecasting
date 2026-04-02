# Multimodal Stock Forecasting
**Multimodal Stock Prediction Framework | Image Modality + Time Series Modality Fusion**

This project constructs stock candlestick chart datasets (mixed and separate formats) based on historical trading data, employs MAE and ViT models for image feature extraction, and conducts comprehensive feature quality comparisons using variance, similarity, MSE, and MAE metrics. The results establish an optimal multimodal feature scheme for stock return prediction.

---

## Completed Work
1. Generation of mixed and separate stock candlestick charts
2. Image feature extraction based on MAE and ViT architectures
3. Quantitative feature quality evaluation across four metrics
4. Validation and selection of the optimal feature extraction scheme

---

## Experimental Conclusions
| Feature Type         | Variance | Similarity | MSE    | MAE    |
|----------------------|----------|------------|--------|--------|
| Separate-ViT         | 2.4979   | 0.9793     | 0.0047 | 0.0469 |
| Separate-MAE         | 0.3508   | 0.9947     | 0.0046 | 0.0467 |
| Mixed-ViT            | 2.6066   | 0.9767     | 0.0006 | 0.0163 |
| **Mixed-MAE**        | **2.6171** | **0.9708** | **0.0006** | **0.0167** |

**Optimal Scheme: Mixed Chart + MAE Feature Extraction**

---

## GitHub Repository Contents
### `code/`
- Mixed chart generation: `mix_generate_figure.ipynb`
- Separate chart generation: `generate_separate_figure.ipynb`
- Mixed chart MAE feature extraction: `mae_features_npz.ipynb`
- Mixed chart ViT feature extraction: `vit_features_npz.ipynb`
- Separate chart MAE feature extraction: `f_s_mae.ipynb`
- Separate chart ViT feature extraction: `f_s_vit.ipynb`
- Main comparative experiment: `混合图vs分图.ipynb`
- Mixed chart model comparison: `mae_vs_vit.ipynb`
- Separate chart model comparison: `f_s_vit_vs_mae.ipynb`
- MAE model download script: `hf_download.ipynb`

### `data/`
- Raw stock trading data: `日个股数据2.0.csv`

---

## Dataset and Model Resources (Hugging Face)
All large files (images, features, model weights) have been published on Hugging Face for academic use:

- **Full Dataset (Images + Extracted Features)**:  
  https://huggingface.co/datasets/zhangqingyue127/Multimodal-Stock-Forecasting-Dataset

- **Pre-trained MAE Model Weights**:  
  https://huggingface.co/zhangqingyue127/Multimodal-Stock-Forecasting-MAE

---

## Complete Reproduction Steps
Two reproduction paths are provided:

### Path 1: Quick Reproduction (Recommended)
Download pre-generated resources directly from Hugging Face and run comparative experiments:
1. Download the dataset and model from the links above
2. Run `code/混合图vs分图.ipynb` to reproduce all experimental results

### Path 2: Full Reproduction from Scratch
1. **Download MAE Model**
   - Code: `code/hf_download.ipynb`
   - Output: `models/new_mae_model/`

2. **Generate Mixed Stock Charts**
   - Code: `code/mix_generate_figure.ipynb`
   - Input: `data/日个股数据2.0.csv`
   - Output: `figures/` (60-day time window, includes candlesticks, volume heatmaps, and moving averages)

3. **Generate Separate Stock Charts**
   - Code: `code/generate_separate_figure.ipynb`
   - Input: `data/日个股数据2.0.csv`
   - Output: `figure_separate/` (isolated candlestick, volume, and indicator charts)

4. **Extract Features from Mixed Charts**
   - MAE: `code/mae_features_npz.ipynb` (labeled with stock returns)
   - ViT: `code/vit_features_npz.ipynb`

5. **Extract Features from Separate Charts**
   - MAE: `code/f_s_mae.ipynb`
   - ViT: `code/f_s_vit.ipynb`

6. **Run Comparative Experiments**
   - Mixed vs. Separate Charts: `code/混合图vs分图.ipynb`
   - Model Comparison (Mixed Charts): `code/mae_vs_vit.ipynb`
   - Model Comparison (Separate Charts): `code/f_s_vit_vs_mae.ipynb`

---

## Final Outputs
- Mixed and separate stock candlestick chart datasets
- MAE/ViT extracted image features (npz format)
- Quantitative feature quality comparison metrics
- **Optimal feature scheme: Mixed Chart + MAE**

---

## Future Work
1. Time series modality feature extraction
2. Multimodal feature fusion architecture design
3. Stock return prediction model training and backtesting
4. Extension to larger stock market datasets

---

## Citation
If you use this dataset or code in your research, please cite:
```
@misc{zhang2026multimodal,
  author = {Zhang, Qingyue},
  title = {Multimodal Stock Forecasting: Image Modality Feature Extraction and Comparison},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/zhangqingyue127/Multimodal-Stock-Forecasting}
}
```

---

## Note
This GitHub repository contains only code and documentation. All large files are hosted on Hugging Face for accessibility and reproducibility.
