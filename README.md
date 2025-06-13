# FedPDA: Federated Learning Framework with Personalized Dynamic Aggregation

Project Overview

FedPDA (Federated Learning with Personalized Dynamic Aggregation) is a federated learning framework based on personalized dynamic aggregation. It can dynamically adjust aggregation strategies according to client data distribution, model behavior, and resource conditions to improve federated learning performance.

Key Features

- Support for multiple datasets: MNIST, Fashion-MNIST, CIFAR10
- Support for multiple deep learning models: ResNet18, VGG16, SimpleCNN, etc.
- Support for IID and Non-IID data distribution scenarios
- Similarity-based dynamic model migration strategy
- Configurable global aggregation mechanism

Environment Requirements

```bash
python >= 3.8
torch >= 1.8.0
torchvision
numpy
scikit-learn
pandas
matplotlib
```

Usage Guide

1. Install Dependencies

```bash
pip install -r requirements.txt
```

2. Run Experiments

Basic usage:

```bash
python main.py --dataset cifar10 --num_clients 5 --num_rounds 100 --model_type resnet18
```

Key Parameters

- `--dataset`: Choose dataset (cifar10/mnist/fashion_mnist)
- `--num_clients`: Number of clients
- `--num_rounds`: Number of training rounds
- `--model_type`: Model type (simple_cnn/resnet18/resnet50/vgg16)
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size
- `--distribution`: Data distribution type (iid/non_iid)
- `--alpha`: Dirichlet parameter for Non-IID distribution
- `--global_aggregation`: Whether to enable global aggregation
- `--global_agg_rounds`: Global aggregation period

Similarity Weight Configuration

- `--data_similarity_weight`: Data distribution similarity weight
- `--behavior_similarity_weight`: Model behavior similarity weight
- `--resource_similarity_weight`: Resource similarity weight

Examples

1. Run CIFAR10 experiment (IID distribution):
```bash
python main.py --dataset cifar10 --distribution iid --num_clients 5 --num_rounds 100
```

2. Run Fashion-MNIST experiment (Non-IID distribution):
```bash
python main.py --dataset fashion_mnist --distribution non_iid --alpha 0.5 --num_clients 10 --num_rounds 150
```


Output Description

- Training logs are saved in the `output/` directory
- Experiment results (accuracy, loss, etc.) are saved as CSV files
- Visualization results are saved in the `plots/` directory

Notes

1. For Non-IID experiments, the alpha parameter controls the degree of data distribution imbalance, with smaller values indicating more uneven distribution
2. It is recommended to adjust similarity weight configuration according to specific tasks
3. Datasets will be automatically downloaded on first run
