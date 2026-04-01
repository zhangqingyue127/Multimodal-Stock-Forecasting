# Multimodal Stock Forecasting  
**Multimodal Stock Prediction Framework | Image Modality + Time Series Modality Fusion**

This project is based on stock candlestick charts and employs MAE/ViT models for image feature extraction and comparative experiments, ultimately facilitating multimodal stock return prediction.

---

## Completed Work
1. Generation of mixed and separate stock charts  
2. Image feature extraction based on MAE and ViT  
3. Feature quality comparison (variance, similarity, MSE, MAE)  
4. Validation of optimal feature scheme  

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

## GitHub Uploaded Content
### code/
- Mixed chart generation: `mix_generate_figure.ipynb`
- Separate chart generation: `generate_separate_figure.ipynb`
- MAE feature extraction: `mae_features_npz.ipynb`
- ViT feature extraction: `vit_features_npz.ipynb`
- Separate chart feature extraction: `f_s_mae.ipynb`, `f_s_vit.ipynb`
- Comparative experiments: `混合图vs分图.ipynb`, `mae_vs_vit.ipynb`, `f_s_vit_vs_mae.ipynb`
- Model download script: `hf_download.ipynb`

### data/
- Raw stock data: `日个股数据2.0.csv`

---

## Large Files Not Uploaded (To Be Released on HuggingFace)
Due to file size limitations, the following contents have not been uploaded to GitHub and will be published on platforms such as HuggingFace in the future:  
1. **models/**: MAE pre-trained model weights (`model.safetensors`, `config.json`)  
2. **figures/**: Mixed and separate chart image data  
3. **results/**: Extracted npz feature files, experimental comparison plots `mix_vs_split.png`

---

## Complete Reproduction Steps
The following steps enable full reproduction of all images, features, and comparative experimental results in this project:

### 1. Download MAE Model
- Code: `code/hf_download.ipynb`
- Input: None (automatically downloaded from huggingface-mirror)
- Output: `models/new_mae_model/`
- Description: Obtains pre-trained MAE model weights to support subsequent feature extraction.

### 2. Generate Mixed Stock Charts
- Code: `code/mix_generate_figure.ipynb`
- Input: `data/日个股数据2.0.csv`
- Output: `figures/`
- Description: Generates mixed charts incorporating candlesticks, volume heatmaps, and moving average indicators (time window: 60 days). Each stock corresponds to a folder storing multiple sequential images.

### 3. Generate Separate Stock Charts
- Code: `code/generate_separate_figure.ipynb`
- Input: `data/日个股数据2.0.csv`
- Output: `figure_separate/`
- Description: Generates separate chart data where candlesticks, volume, and technical indicators are isolated. Each stock is stored independently.

### 4. Feature Extraction from Mixed Charts
#### MAE Feature Extraction
- Code: `code/mae_features_npz.ipynb`
- Input: `figures/` + `models/new_mae_model/`
- Output: `results/mae_new_npz_features/`
- Label: Returns (correctly annotated)

#### ViT Feature Extraction
- Code: `code/vit_features_npz.ipynb`
- Input: `figures/`
- Output: `results/vit_new_npz_features/`

### 5. Feature Extraction from Separate Charts
#### MAE Feature Extraction
- Code: `code/f_s_mae.ipynb`
- Input: `figure_separate/`
- Output: `results/f_s_mae_features/`

#### ViT Feature Extraction
- Code: `code/f_s_vit.ipynb`
- Input: `figure_separate/`
- Output: `results/f_s_vit_features/`

### 6. Feature Comparison Experiments
#### Mixed vs. Separate Charts
- Code: `code/混合图vs分图.ipynb`
- Input: All extracted feature files
- Output: Comparison results of four metrics: variance, similarity, MSE, MAE
- Conclusion: Mixed-MAE features achieve the best performance.

#### Separate Chart Comparison (ViT vs. MAE)
- Code: `code/f_s_vit_vs_mae.ipynb`
- Description: Validates the difference in feature quality between models in the separate chart scenario.

#### Mixed Chart Comparison (MAE vs. ViT)
- Code: `code/mae_vs_vit.ipynb`
- Description: Validates the difference in feature quality between models in the mixed chart scenario.

---

## Final Outputs
- Mixed and separate stock chart image data  
- Image features extracted via MAE/ViT (npz format)  
- Quality comparison metrics for four feature combinations  
- **Optimal feature scheme: Mixed Chart + MAE**

---

## Future Work
1. Time series modality feature extraction  
2. Multimodal feature fusion  
3. Return prediction model training and backtesting  
4. Dataset and model weights release on HuggingFace

---

## Note
The GitHub repository contains only code and project documentation. Large files such as images, features, and model weights will be separately published on the HuggingFace platform.
