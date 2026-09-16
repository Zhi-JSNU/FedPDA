[README.md](https://github.com/user-attachments/files/32296095/README.md)
<div align="center">

# FedPDA

### Personalized Federated Learning Based on Attribute Similarity Migration

<p>
  <a href="https://www.sciencedirect.com/journal/information-sciences">
    <img src="https://img.shields.io/badge/Published%20in-Information%20Sciences-0B4F6C?style=for-the-badge&logo=elsevier&logoColor=white" alt="Published in Information Sciences" />
  </a>
  <a href="https://doi.org/10.1016/j.ins.2025.122553">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.ins.2025.122553-D35400?style=for-the-badge&logo=doi&logoColor=white" alt="DOI: 10.1016/j.ins.2025.122553" />
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Volume-720-153E75?style=flat-square" alt="Volume 720" />
  <img src="https://img.shields.io/badge/Article-122553-153E75?style=flat-square" alt="Article 122553" />
  <img src="https://img.shields.io/badge/Year-2025-153E75?style=flat-square" alt="Year 2025" />
  <a href="#citation--引用"><img src="https://img.shields.io/badge/Cite-BibTeX-7B2CBF?style=flat-square" alt="Cite this work" /></a>
</p>

<p><b>Official implementation of FedPDA</b> &mdash; a personalized federated learning framework that migrates knowledge across clients according to attribute similarity.</p>

[📄 Read the paper](https://doi.org/10.1016/j.ins.2025.122553) &nbsp;·&nbsp; [🚀 Quick start](#quick-start--快速开始) &nbsp;·&nbsp; [📚 Citation](#citation--引用)

</div>

---

> **Published paper / 已发表论文**  
> **FedPDA: Personalized Federated Learning Based on Attribute Similarity Migration**  
> *Information Sciences*, **Volume 720**, Article 122553, 2025.  
> DOI: [10.1016/j.ins.2025.122553](https://doi.org/10.1016/j.ins.2025.122553)

## Overview / 项目概述

**FedPDA** is a personalized federated learning framework based on attribute-similarity migration. It dynamically adapts knowledge transfer and aggregation according to client data distributions, model behavior, and resource conditions.

**FedPDA** 是一个基于属性相似性迁移的个性化联邦学习框架。该方法根据客户端的数据分布、模型行为和资源条件，自适应地调整知识迁移与聚合策略，以提升异构环境下的联邦学习性能。

## Highlights / 核心特点

| English | 中文 |
| :-- | :-- |
| **Attribute-similarity migration** combines data distribution, model behavior, and resource attributes. | **属性相似性迁移**：综合数据分布、模型行为与资源属性进行客户端间迁移。 |
| **Personalized dynamic aggregation** adapts to heterogeneous client conditions. | **个性化动态聚合**：适应不同客户端的异构条件。 |
| Supports **IID** and **Dirichlet-based Non-IID** data partitions. | 支持 **IID** 与基于 Dirichlet 分布的 **Non-IID** 数据划分。 |
| Includes MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100; with SimpleCNN, ResNet-18/50, and VGG-16 backbones. | 包含 MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100 数据集，以及 SimpleCNN、ResNet-18/50、VGG-16 模型。 |

## Repository Structure / 项目结构

```text
FedPDA/
├── main.py                 # experiment entry point / 实验入口
├── client.py               # local client training and evaluation / 客户端本地训练与评估
├── fmi_server.py           # federated server / 联邦服务器
├── SimilarityManager.py    # attribute similarity and migration / 属性相似性与迁移
├── dataset.py              # dataset loading and partitioning / 数据加载与划分
└── models.py               # model definitions / 模型定义
```

## Environment / 运行环境

- Python 3.8 or later / Python 3.8 或更高版本
- PyTorch 1.8.0 or later / PyTorch 1.8.0 或更高版本
- CUDA is optional; the code automatically uses CUDA when available. / CUDA 为可选项；可用时程序会自动使用 GPU。

Install the required packages / 安装依赖：

```bash
pip install torch torchvision numpy scikit-learn pandas
```

> Please install the PyTorch build appropriate for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) when using a GPU.  
> 如使用 GPU，请根据 CUDA 版本在 [pytorch.org](https://pytorch.org/get-started/locally/) 安装相匹配的 PyTorch。

## Quick Start / 快速开始

The datasets are downloaded automatically to `./data` on first use. Results and logs are written to `./output`.

首次运行时，数据集会自动下载到 `./data`；实验结果和日志将保存至 `./output`。

```bash
# CIFAR-10, Non-IID setting (default configuration)
# CIFAR-10，Non-IID 设置（默认配置）
python main.py --dataset cifar10 --num_clients 10 --num_rounds 100 --model_type resnet18
```

### Example Configurations / 示例配置

```bash
# CIFAR-10 with IID client data / CIFAR-10 的 IID 客户端数据
python main.py --dataset cifar10 --distribution iid --num_clients 5 --num_rounds 100

# Fashion-MNIST with Dirichlet Non-IID client data
# Fashion-MNIST 的 Dirichlet Non-IID 客户端数据
python main.py --dataset fashion_mnist --distribution non_iid --alpha 0.5 --num_clients 10 --num_rounds 150

# CIFAR-100 (set the correct number of classes explicitly)
# CIFAR-100（请显式设置正确的类别数）
python main.py --dataset cifar100 --num_classes 100 --model_type resnet18
```

## Main Arguments / 主要参数

| Argument | Description / 说明 | Default |
| :-- | :-- | :-- |
| `--dataset` | Dataset: `cifar10`, `cifar100`, `mnist`, or `fashion_mnist` / 数据集 | `cifar10` |
| `--model_type` | Model: `simple_cnn`, `resnet18`, `resnet50`, or `vgg16` / 模型 | `resnet18` |
| `--num_clients` | Number of federated clients / 联邦客户端数量 | `10` |
| `--num_rounds` | Number of communication rounds / 通信轮数 | `100` |
| `--epochs` | Local epochs per round / 每轮本地训练轮数 | `5` |
| `--batch_size` | Local batch size / 本地批次大小 | `32` |
| `--learning_rate` | Local learning rate / 本地学习率 | `0.003` |
| `--distribution` | Data partition: `iid` or `non_iid` / 数据划分方式 | `non_iid` |
| `--alpha` | Dirichlet concentration for Non-IID partitioning / Non-IID 划分的 Dirichlet 参数 | `0.5` |
| `--global_aggregation` | Enable periodic global aggregation / 启用周期性全局聚合 | disabled / 未启用 |
| `--global_agg_rounds` | Period of global aggregation / 全局聚合周期 | `5` |

### Similarity Weights / 相似度权重

FedPDA evaluates client affinity using three complementary attributes. The three weights should normally sum to 1.

FedPDA 使用以下三类属性评估客户端间的亲和性。通常建议三个权重之和为 1。

| Argument | Attribute / 属性 | Default |
| :-- | :-- | :-- |
| `--data_similarity_weight` | Data-distribution similarity / 数据分布相似度 | `0.33` |
| `--behavior_similarity_weight` | Model-behavior similarity / 模型行为相似度 | `0.33` |
| `--resource_similarity_weight` | Resource similarity / 资源相似度 | `0.34` |

## Outputs / 输出说明

- `output/*.log`: training logs / 训练日志
- `output/*.csv`: per-round client metrics, including accuracy, loss, and F1 score / 每轮客户端指标，包括准确率、损失与 F1 分数

## Notes / 使用提示

- Smaller `--alpha` values produce more heterogeneous Non-IID partitions. / 更小的 `--alpha` 会产生更不均衡的 Non-IID 数据划分。
- For CIFAR-100, pass `--num_classes 100`. / 使用 CIFAR-100 时，请传入 `--num_classes 100`。
- The experiment seed is set to 42 in `main.py`. / `main.py` 中的实验随机种子设置为 42。

## Citation / 引用

If you find this repository useful in your research, please cite our paper.

如果本项目对您的研究有所帮助，请引用以下论文：

```bibtex
@article{zhou2025fedpda,
  title   = {FedPDA: Personalized federated learning based on attribute similarity migration},
  author  = {Zhou, Xiang and Zhi, Qiang and Liu, Ziyang and Han, Dongyi and Liu, Nan},
  journal = {Information Sciences},
  volume  = {720},
  pages   = {122553},
  year    = {2025},
  doi     = {10.1016/j.ins.2025.122553}
}
```

## License / 许可证

No license has been specified for this repository yet. Please contact the authors for permission before using the code beyond the scope permitted by applicable law.

本仓库暂未指定开源许可证。除适用法律允许的范围外，如需使用本代码，请先联系作者获取许可。

