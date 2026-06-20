# Result files

本目录只保存可审阅的轻量实验汇总，不保存生成图、逐股票 NPZ 特征、模型权重或日志。

- `aligned_feature_comparison.json`: mixed/separate 与 ViT/MAE 四组对比。
- `three_layer_results.json`: 特征诊断、简单探针和多模态下游评估。
- `factor_selection_backtest_summary.json`: 因子筛选、横截面标准化和回测摘要。
- `iterative_*_results.json`: 排序、融合、强模型与滚动验证实验。
- `multi_metric_tuning_results.json`: 多模型调参与回归结果汇总。
- `focused_auc_results.json`: 多预测期辅助实验及回归参考结果。
- `gpu_auc_search_results.json`: GPU 模型搜索结果。
- `ranker_auc_results.json`: XGBoost Ranker 搜索结果。

README 中的主表均取自这些文件。由于生成特征未入库，完整复现需先运行仓库根目录 README 中的图像生成与特征提取步骤。
