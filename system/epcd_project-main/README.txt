EPCD：基于子图传播路径的可解释性网络暴力检测模型
本仓库是论文 “An Explainable Cyberbullying Detection Model Based on Subgraph Propagation Paths” 的官方 PyTorch 实现。

该模型从群体极化理论出发，将网络暴力检测从传统的文本分类升级为结构化传播路径推理。通过显式提取社交网络中的信息传播链条，联合建模结构依赖（GCN）与情绪演化（BiLSTM），并引入核心‑背景子图划分与可解释性监督，最终在输出检测结果的同时，自动定位关键传播路径与核心用户节点，为平台治理提供可落地的决策依据。