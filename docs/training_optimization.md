# 地质 GNN 训练优化方案（中文版）

## 当前状况与痛点
- 缩小模型后测试指标大幅下降：Accuracy ~0.28，F1(Macro) ~0.32。
- 先前较大模型曾达到 Accuracy ~0.53 / F1(Macro) ~0.55。
- 对图连通性和模型容量高度敏感：过小易欠拟合，过深且无合适边权处理会过平滑。

## 立即优化动作（按优先级）
1) 数据与预处理
- 类别再平衡：继续使用 class weights，并对 focal gamma 试 {1.5, 2.0, 2.5}。
- 特征核查：确认 `feature_cols` 覆盖；已添加深度分箱，可加每口钻孔的岩性频率编码。
- 异常值：缩尾/裁剪深度与特征的极端 z-score（>3）再做标准化。

2) 图构建
- KNN 搜索：k 取 {20, 30, 40}，监控平均度数和连通分量。
- 距离上限：将 `max_distance` 设为两两距离的 75% 分位，截断远距离噪声边。
- 边权：保持 `exp(-d/mean_d)`；分支试验反距离 `1/(d+eps)`。

3) 模型结构
- 主力：EnhancedGeoGNN（GATv2 + edge_attr），hidden {192, 256}，layers {4, 6}，heads=4，dropout=0.3。
- 基线：GraphSAGE（残差+LayerNorm）做稳定性对照；hidden {128, 192}，layers {4, 5}。
- 正则：分层 dropout 0.3；若加深，可尝试 drop path 0.05。

4) 训练策略
- 学习率：LR finder 后用 cosine_restart；起始 lr {0.0015, 0.002}，weight_decay 1e-4。
- 损失：默认 focal；若不稳定，换 CE + label smoothing 0.05。
- 早停：patience 60–80，min_lr 1e-6。
- 梯度裁剪：clip_norm=1.0，抑制注意力梯度尖峰。

5) 评估与诊断
- 追踪按类 F1 与支持数，关注零体积岩性。
- 验证/测试集混淆矩阵，观察易混类别（如各类砂岩）。
- 校准：可靠性曲线 / ECE，若过度自信可做温度缩放。

## 推荐实验矩阵
- A1: EnhancedGeoGNN，hidden 256，layers 6，k=30，lr 0.002，focal gamma 2.0。
- A2: EnhancedGeoGNN，hidden 192，layers 4，k=30，lr 0.0015，focal gamma 2.0。
- A3: EnhancedGeoGNN，hidden 256，layers 6，k=40，lr 0.002，focal gamma 1.5。
- B1: GraphSAGE，hidden 192，layers 5，k=30，lr 0.0015，label smoothing 0.05。
- B2: GraphSAGE，hidden 128，layers 4，k=20，lr 0.0015，focal gamma 2.5。
记录：val F1 macro、分类别 F1、训练稳定性（loss 曲线）、推理延迟。

## 日志与复现
- 固定种子：random_seed=42，并在每次运行记录 seed。
- 每次运行保存配置快照（超参、git 提交哈希）。
- 保存验证 F1 最优的前 3 个 checkpoint，而非仅最优一个。

## 训练后处理
- 用温度缩放导出校准后的概率，用于体积统计。
- 生成混淆矩阵、分类报告、按类支持数图。
- 可视化：体素置信度直方图，定位高不确定区域。

## 论文写作建议（面向地学+ML 期刊/会议）
- 问题动机：稀疏且不规则的钻孔采样，需要基于图的空间插值。
- 方法亮点：GATv2 + 距离感知边特征；深度/质心等工程特征。
- 消融：k（20/30/40）、hidden（192/256）、深度（4/6）、损失（focal vs label smoothing）、边权方案（exp vs 反距离）。
- 基线：克里金、IDW、表格 MLP、GCN/GraphSAGE。
- 指标：Accuracy、F1 macro/weighted、若可行加入按类体积 IoU；校准指标 ECE。
- 图示：3D 栅栏图、预测/真实剖面对比、不确定性热力图、岩性体积柱状图。

## 接下来要做的事（短清单）
- 先跑 A1 和 A2，择稳者继续。
- 若小类仍为 0，降 gamma 到 1.5，并启用 max_distance 上限。
- 若 loss 有尖峰，加梯度裁剪 1.0 再跑。
- 为论文留图：混淆矩阵、学习曲线、体素置信度分布。
