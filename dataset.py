import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import logging
import os
import time

class DatasetManager:
    def __init__(self):
        self.transform = {
            'MNIST': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomErasing(p=0.2)
            ]),
            'CIFAR10': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
                transforms.RandomErasing(p=0.2)
            ]),
            'FashionMNIST': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            'CIFAR100': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        }
        
        self.test_transform = {
            'MNIST': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'CIFAR10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        }
        
        self.dataset_configs = {
            'MNIST': {'num_classes': 10, 'channels': 1},
            'FashionMNIST': {'num_classes': 10, 'channels': 1},
            'CIFAR10': {'num_classes': 10, 'channels': 3},
            'CIFAR100': {'num_classes': 100, 'channels': 3}
        }
        
    def load_dataset(self, dataset_name, train=True):
        """加载数据集"""
        dataset_name = dataset_name.upper()
        if dataset_name not in self.transform:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        dataset_class = getattr(datasets, dataset_name)
        return dataset_class(
            root='./data',
            train=train,
            download=True,
            transform=self.transform[dataset_name] if train else self.test_transform[dataset_name]
        )

    def create_client_datasets(self, dataset, num_clients, is_iid=True, alpha=0.5):
        np.random.seed(int(time.time()))
        """创建客户端数据集"""
        if is_iid:
            # IID分布
            num_items = len(dataset)
            indices = np.random.permutation(num_items)
            batch_indices = np.array_split(indices, num_clients)
            client_datasets = [Subset(dataset, indices) for indices in batch_indices]
        else:
            # 使用Dirichlet分布创建Non-IID数据集
            if hasattr(dataset, 'targets'):
                labels = np.array(dataset.targets)
            elif hasattr(dataset, 'train_labels'):
                labels = np.array(dataset.train_labels)
            else:
                labels = np.array([target for _, target in dataset])
            
            num_classes = len(np.unique(labels))
            
            # 为每个类别生成Dirichlet分布
            client_proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)
            
            # 按类别划分数据索引
            class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
            
            # 为每个客户端分配数据
            client_idxs = [[] for _ in range(num_clients)]
            
            # 根据Dirichlet分布分配数据
            for c_idx, c_idxs in enumerate(class_idxs):
                np.random.shuffle(c_idxs)
                
                # 按比例分配给客户端
                proportions = client_proportions[c_idx]
                cumsum = np.cumsum(proportions)
                cumsum = (cumsum * len(c_idxs)).astype(int)[:-1]
                client_portions = np.split(c_idxs, cumsum)
                
                # 分配给对应的客户端
                for client_id, portion in enumerate(client_portions):
                    client_idxs[client_id].extend(portion)
            
            # 对每个客户端的数据再次打乱
            for client_id in range(num_clients):
                np.random.shuffle(client_idxs[client_id])
            
            # 创建子数据集并记录分布情况
            client_datasets = []
            for i, idxs in enumerate(client_idxs):
                client_datasets.append(Subset(dataset, idxs))
                # 记录每个客户端的数据分布
                client_labels = labels[idxs]
                dist = np.bincount(client_labels, minlength=num_classes) / len(idxs)
                logging.info(f"\n客户端 {i} 的数据分布:")
                for c, p in enumerate(dist):
                    logging.info(f"类别 {c}: {p:.3f}")
        
        return client_datasets
        
    def _create_dirichlet_datasets(self, dataset, num_clients, alpha):
        """使用Dirichlet分布创建Non-IID数据集"""
        # 获取所有标签
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'train_labels'):
            labels = np.array(dataset.train_labels)
        else:
            # 如果是Subset类型，尝试从dataset.dataset获取
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
                labels = np.array(dataset.dataset.targets)
            else:
                raise ValueError("无法获取数据集标签")
    
        num_classes = len(np.unique(labels))
        
        # 为每个类别生成Dirichlet分布
        client_proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)
        
        # 按类别划分数据索引
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
        
        # 为每个客户端分配数据
        client_idxs = [[] for _ in range(num_clients)]
        
        # 根据Dirichlet分布分配数据
        for c_idx, c_idxs in enumerate(class_idxs):
            # 打乱每个类别的数据索引
            np.random.shuffle(c_idxs)
            
            # 按比例分配给客户端
            proportions = client_proportions[c_idx]
            cumsum = np.cumsum(proportions)
            cumsum = (cumsum * len(c_idxs)).astype(int)[:-1]
            client_portions = np.split(c_idxs, cumsum)
            
            # 分配给对应的客户端
            for client_id, portion in enumerate(client_portions):
                client_idxs[client_id].extend(portion)
        
        # 对每个客户端的数据再次打乱
        for client_id in range(num_clients):
            np.random.shuffle(client_idxs[client_id])
        
        # 创建子数据集
        client_datasets = []
        for idxs in client_idxs:
            client_datasets.append(Subset(dataset, idxs))
        
        return client_datasets

    def _create_iid_datasets(self, dataset, num_clients):
        """创建IID数据分布的客户端数据集"""
        num_items = len(dataset)
        indices = torch.randperm(num_items)
        batch_indices = np.array_split(indices, num_clients)
        client_datasets = [Subset(dataset, indices) for indices in batch_indices]
        
        return client_datasets
        
    def _create_non_iid_datasets(self, dataset, num_clients, alpha):
        """创建Non-IID数据分布的客户端数据集"""
        labels = self._get_dataset_labels(dataset)
        num_classes = len(np.unique(labels))
        
        # 为每个类别生成Dirichlet分布
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        # 按类别分配数据
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
        client_idxs = [[] for _ in range(num_clients)]
        
        # 为每个类别分配样本
        for c, fracs in enumerate(label_distribution):
            for i, idx in enumerate(np.split(class_idxs[c], 
                (np.cumsum(fracs)[:-1] * len(class_idxs[c])).astype(int))):
                client_idxs[i].extend(idx)
        
        client_datasets = [Subset(dataset, idxs) for idxs in client_idxs]
        
        # 记录分配情况
        for i, dataset in enumerate(client_datasets):
            labels = self._get_dataset_labels(dataset)
            dist = np.bincount(labels, minlength=num_classes) / len(labels)
            logging.info(f"\nClient {i} data distribution:")
            for c, p in enumerate(dist):
                logging.info(f"Class {c}: {p:.3f}")
                
        return client_datasets
        
    def _get_dataset_labels(self, dataset):
        """获取数据集的标签"""
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'train_labels'):
            labels = dataset.train_labels
        else:
            raise ValueError("无法获取数据集标签")
            
        if isinstance(dataset, Subset):
            return np.array(labels)[dataset.indices]
        return np.array(labels)

    def create_data_loaders(self, datasets, batch_size, shuffle=True, generator=None):
        """创建数据加载器"""
        loaders = []
        for dataset in datasets:
            # 确保训练和测试使用相同的配置
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=True,
                generator=generator,
                drop_last=True  # 训练和测试都使用 drop_last=True
            )
            loaders.append(loader)
        return loaders

    def _find_largest_divisor(self, n, max_value):
        """找到不超过max_value的最大除数"""
        for i in range(max_value, 0, -1):
            if n % i == 0:
                return i
        return 1


    def get_distribution_similarity(self, dist1, dist2):
        """计算两个分布之间的相似度"""
        if dist1 is None or dist2 is None:
            return 0.0
            
        # 使用余弦相似度
        dot_product = np.dot(dist1, dist2)
        norm1 = np.linalg.norm(dist1)
        norm2 = np.linalg.norm(dist2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    # 在DatasetManager类中添加IID和Non-IID数据分区方法
    
    def create_iid_partition(self, dataset, num_clients):
        """创建IID数据分区（独立同分布）"""
        num_items = len(dataset)
        items_per_client = num_items // num_clients
        client_data_indices = {}
        
        # 随机打乱索引
        all_indices = list(range(num_items))
        np.random.shuffle(all_indices)
        
        # 均匀分配给每个客户端
        for i in range(num_clients):
            start_idx = i * items_per_client
            end_idx = (i + 1) * items_per_client if i < num_clients - 1 else num_items
            client_data_indices[f'client_{i}'] = all_indices[start_idx:end_idx]
        
        return client_data_indices
    
    def create_non_iid_partition(self, dataset, num_clients, alpha=0.5):
        """创建Non-IID数据分区（非独立同分布）
        
        Args:
            dataset: 数据集
            num_clients: 客户端数量
            alpha: 狄利克雷分布参数，越小数据越不均衡
        """
        # 获取标签
        labels = []
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            # 如果数据集没有直接提供标签，则遍历数据集获取
            for _, label in dataset:
                labels.append(label)
        
        labels = np.array(labels)
        num_classes = len(np.unique(labels))
        
        # 为每个客户端分配类别比例
        client_proportions = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
        
        # 按类别分组索引
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        
        # 为每个客户端分配数据
        client_data_indices = {f'client_{i}': [] for i in range(num_clients)}
        
        # 按类别分配数据
        for c, indices in enumerate(class_indices):
            np.random.shuffle(indices)
            
            # 计算每个客户端应分配的该类别数据量
            proportions = client_proportions[c]
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)
            
            # 分配数据
            start = 0
            for i in range(num_clients):
                end = proportions[i]
                client_data_indices[f'client_{i}'].extend(indices[start:end])
                start = end
        
        return client_data_indices
