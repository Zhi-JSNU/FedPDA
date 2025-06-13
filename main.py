import torch
import logging
import argparse
import numpy as np
import os
from datetime import datetime
from dataset import DatasetManager
from models import get_model
from client import Client
from fmi_server import FMIServer
import random
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(description='个性化联邦学习参数')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--num_rounds', type=int, default=100, help='训练轮数')
    parser.add_argument('--epochs', type=int, default=5, help='每轮本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'],
                       help='数据集名称 (cifar10, cifar100, mnist, fashion_mnist)')
    parser.add_argument('--model_type', type=str, default='resnet18', 
                       choices=['simple_cnn', 'resnet18', 'resnet50', 'vgg16'], 
                       help='模型类型')
    parser.add_argument('--learning_rate', type=float, default=0.003, 
                       help='学习率')
    parser.add_argument('--distribution', type=str, default='non_iid', 
                       choices=['iid', 'non_iid'], help='数据分布类型')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Non-IID分布的alpha参数')
    parser.add_argument('--behavior_similarity_weight', type=float, default=0.33,
                       help='模型行为相似度权重')
    parser.add_argument('--data_similarity_weight', type=float, default=0.33,
                       help='数据分布相似度权重')
    parser.add_argument('--resource_similarity_weight', type=float, default=0.34,
                       help='资源相似度权重')
    parser.add_argument('--global_aggregation', action='store_true', default=False,
                       help='是否启用周期性全局聚合')
    parser.add_argument('--global_agg_rounds', type=int, default=5,
                       help='全局聚合的周期（每多少轮执行一次）')
    return parser.parse_args()

def setup_logging():
    """设置日志"""
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'output/fedp_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def main():

    def setup_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    setup_seed()
    # 初始化
    args = parse_args()
    setup_logging()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载数据集
    dataset_manager = DatasetManager()
    train_dataset = dataset_manager.load_dataset(args.dataset, train=True)
    test_dataset = dataset_manager.load_dataset(args.dataset, train=False)
    
    # 创建客户端数据集
    is_iid = (args.distribution == 'iid')
    train_datasets = dataset_manager.create_client_datasets(
        train_dataset, 
        args.num_clients, 
        is_iid=is_iid,  # 训练集分布根据命令行参数决定
        alpha=args.alpha
    )
    test_datasets = dataset_manager.create_client_datasets(
        test_dataset, 
        args.num_clients, 
        is_iid=True,  # 测试集始终使用IID分布
        alpha=1.0
    )
    
    
    # 初始化服务器
    server = FMIServer(device=device, 
                      num_classes=args.num_classes, 
                      enable_global_agg=args.global_aggregation,
                      global_agg_rounds=args.global_agg_rounds,
                      epochs=args.epochs,  # 添加epochs参数
                      batch_size=args.batch_size,
                      dataset_type=args.dataset)  
    
    # 修改相似度权重设置
    server.similarity_manager.similarity_weights = {
        'data': args.data_similarity_weight,
        'resource': args.resource_similarity_weight,
        'model': args.behavior_similarity_weight  # 修正权重名称
    }
    
    # 初始化客户端
    clients = []
    for i in range(args.num_clients):
        # 创建数据加载器
        train_loader = dataset_manager.create_data_loaders([train_datasets[i]], args.batch_size)[0]
        test_loader = dataset_manager.create_data_loaders([test_datasets[i]], args.batch_size, shuffle=False)[0]
        
        # 根据数据集类型设置输入通道数
        if args.dataset in ['mnist', 'fashion_mnist']:
            in_channels = 1  # 改为 in_channels
            input_size = 28
        else:  # cifar10, cifar100
            in_channels = 3  # 改为 in_channels
            input_size = 32
            
        # 创建模型
        model = get_model(
            model_type=args.model_type,
            num_classes=args.num_classes,
            in_channels=in_channels,  # 使用命名参数更清晰
            input_size=input_size
        )
        
        def init_weights(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(init_weights)
        
        # 创建客户端
        client_id = f"client_{i}"
        client = Client(client_id, model, device, train_loader, test_loader, 
                       learning_rate=args.learning_rate, model_type=args.model_type,
                       epochs=args.epochs)
        
        # 注册客户端到服务器
        server.register_client(client)
        clients.append(client)
        logging.info(f"注册客户端: {client_id}")
    
    # 添加这一行：初始化联邦学习系统
    logging.info("初始化联邦学习系统...")
    
    # 添加以下代码：更新客户端信息并计算相似度矩阵
    for client in clients:
        # 获取数据分布 - 使用客户端自己的方法
        data_distribution = client.get_data_distribution()
        
        # 更新客户端信息
        server.similarity_manager.update_client_info(
            client.client_id,
            model=client.model,
            data_distribution=data_distribution,
            resources=client.attributes
        )
        
        # 标记客户端初始化完成
        logging.info(f"客户端 {client.client_id} 初始化完成")
    
    # 计算相似度矩阵
    server.similarity_manager.compute_similarity_matrix()
    
    # 标记系统初始化完成
    logging.info("联邦学习系统初始化完成")
    logging.info("完成联邦学习系统初始化")
    
    # 添加DataFrame用于记录训练过程
    import pandas as pd
    training_records = {
        'round': [],
        'client_id': [],
        'accuracy': [],
        'loss': [],
        'f1_score': [],
        'avg_accuracy': [],
        'avg_loss': [],
        'avg_f1': []
    }
    
    # 训练过程
    for round_idx in range(args.num_rounds):
        server.train_round(round_idx)
        
        # 收集并记录性能指标
        round_accuracies = []
        round_losses = []
        round_f1_scores = []
        
        for client in clients:
            metrics = client.evaluate()
            round_accuracies.append(metrics['accuracy'])
            round_losses.append(metrics['loss'])
            round_f1_scores.append(metrics['f1_score'])
            
            # 记录每个客户端的指标
            training_records['round'].append(round_idx + 1)
            training_records['client_id'].append(client.client_id)
            training_records['accuracy'].append(metrics['accuracy'])
            training_records['loss'].append(metrics['loss'])
            training_records['f1_score'].append(metrics['f1_score'])
            training_records['avg_accuracy'].append(np.mean(round_accuracies))
            training_records['avg_loss'].append(np.mean(round_losses))
            training_records['avg_f1'].append(np.mean(round_f1_scores))
        
        # 打印本轮平均性能
        avg_accuracy = np.mean(round_accuracies)
        avg_loss = np.mean(round_losses)
        avg_f1 = np.mean(round_f1_scores)
        
        logging.info(f"\n轮次 {round_idx + 1} 完成:")
        logging.info(f"平均准确率: {avg_accuracy:.4f}")
        logging.info(f"平均损失: {avg_loss:.4f}")
        logging.info(f"平均F1分数: {avg_f1:.4f}")
        
        for i, client in enumerate(clients):
            logging.info(f"客户端 {client.client_id} - 准确率: {round_accuracies[i]:.4f}, 损失: {round_losses[i]:.4f}, F1: {round_f1_scores[i]:.4f}")
        
        # 检查是否收敛（移动到循环内部）
        if server.similarity_manager.check_convergence({client.client_id: client.metrics for client in clients}):
            logging.info(f"训练在第 {round_idx + 1} 轮收敛，但将继续训练")
    
    # 保存训练记录到CSV文件
    df = pd.DataFrame(training_records)
    
    # 创建基于实验参数的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.dataset}_c{args.num_clients}_r{args.num_rounds}_a{args.alpha}"
    weights_info = f"d{args.data_similarity_weight}_b{args.behavior_similarity_weight}_r{args.resource_similarity_weight}"
    
    # 确保output目录存在
    os.makedirs('output', exist_ok=True)
    
    # 保存CSV文件到output目录
    csv_path = f'output/{experiment_name}_{weights_info}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"\n训练记录已保存到: {csv_path}")
    
    # 最终评估
    final_metrics = {
        'average_accuracy': np.mean(round_accuracies),
        'average_loss': np.mean(round_losses),
        'average_f1': np.mean(round_f1_scores)
    }
    
    logging.info("\n训练完成!")
    logging.info(f"最终平均准确率: {final_metrics['average_accuracy']:.4f}")
    logging.info(f"最终平均损失: {final_metrics['average_loss']:.4f}")
    logging.info(f"最终平均F1分数: {final_metrics['average_f1']:.4f}")
    logging.info("\n训练完成!")


# 保存结果
#result_file = f'output/fedp_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
#with open(result_file, 'w') as f:
#    f.write(f"数据集: {args.dataset}\n")
#    f.write(f"模型类型: {args.model_type}\n")
#    f.write(f"学习率: {args.learning_rate}\n")
#    f.write(f"相似度权重配置:\n")
#    f.write(f"  - 数据相似度: {args.data_similarity_weight}\n")
#    f.write(f"  - 资源相似度: {args.resource_similarity_weight}\n")
#    f.write(f"  - 行为相似度: {args.behavior_similarity_weight}\n")
#    f.write(f"客户端数量: {args.num_clients}\n")
#    f.write(f"训练轮数: {args.num_rounds}\n")
#    f.write(f"数据分布: {args.distribution} (alpha={args.alpha})\n")
#    f.write(f"最终平均准确率: {final_metrics['average_accuracy']:.2f}%\n")
#    f.write(f"最终平均F1分数: {final_metrics['average_f1']:.2f}\n")
#    f.write("\n各客户端准确率:\n")
#    for client_id, accuracy in final_metrics['client_accuracies'].items():
#        f.write(f"{client_id}: {accuracy:.2f}%\n")

#logging.info(f"结果已保存到 {result_file}")

if __name__ == '__main__':
    main()