import torch
import numpy as np
import logging
from SimilarityManager import SimilarityManager
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import copy  # 添加这一行
import torch.nn as nn  # 添加这一行
from models import get_model

class FMIServer:
    def __init__(self, device, num_classes, enable_global_agg, global_agg_rounds, epochs=5, batch_size=64, dataset_type='mnist', 
                 data_similarity_weight=0.6, behavior_similarity_weight=0.3, resource_similarity_weight=0.1):  # 添加权重参数
        self.device = device
        self.num_classes = num_classes
        self.clients = {}
        # 添加全局模型
        self.global_model = None
        # 添加全局聚合相关参数
        self.enable_global_agg = enable_global_agg
        self.global_agg_rounds = global_agg_rounds
        self.similarity_manager = SimilarityManager(
            server=self,
            data_weight=data_similarity_weight,
            behavior_weight=behavior_similarity_weight,
            resource_weight=resource_similarity_weight
        )
        
        # 服务器状态
        self.current_round = 0
        
        # 训练配置
        self.epochs = epochs  # 使用传入的参数
        self.batch_size = batch_size  # 使用传入的参数
        
        # 性能追踪 - 现在主要由SimilarityManager管理
        self.client_history = defaultdict(lambda: {
            'performance_history': [],  # 历史性能记录
            'improvement_rate': [],     # 性能提升率
            'migration_history': []     # 迁移历史
        })
        
        # 设置随机输入维度，用于模型行为相似度计算
        self.dataset_type = dataset_type  
        self.setup_model_behavior_params()
        
        # 添加启发式搜索相关参数
        self.exploration_rate = 0.3  # 初始探索率
        self.max_migration_models = 5  # 最大迁移模型数量

    
    def setup_model_behavior_params(self):
        """设置模型行为相似度计算参数"""
        # 根据数据集类型设置随机输入维度和形状
        if self.dataset_type in ['mnist', 'fashion_mnist']:
            in_channels = 1
            input_size = 28
            self.similarity_manager.random_input_dim = in_channels * input_size * input_size
            self.similarity_manager.input_shape = (1, in_channels, input_size, input_size)
        elif self.dataset_type in ['cifar10', 'cifar100']:
            in_channels = 3
            input_size = 32
            self.similarity_manager.random_input_dim = in_channels * input_size * input_size
            self.similarity_manager.input_shape = (1, in_channels, input_size, input_size)
        elif self.dataset_type == 'imagenet':
            in_channels = 3
            input_size = 224
            self.similarity_manager.random_input_dim = in_channels * input_size * input_size
            self.similarity_manager.input_shape = (1, in_channels, input_size, input_size)
        else:
            # 默认值 - 尝试从客户端数据中检测输入维度
            if len(self.clients) > 0:
                first_client_id = list(self.clients.keys())[0]
                client = self.clients[first_client_id]
                
                # 从数据中获取输入维度
                try:
                    sample_data = next(iter(client.train_data))[0]
                    in_channels = sample_data.size(1)
                    input_size = sample_data.size(-1)
                    
                    self.similarity_manager.random_input_dim = in_channels * input_size * input_size
                    self.similarity_manager.input_shape = (1, in_channels, input_size, input_size)
                    
                    logging.info(f"从客户端数据检测到输入维度: 通道数={in_channels}, 尺寸={input_size}")
                except:
                    # 如果无法从数据中获取，使用默认值
                    self.similarity_manager.input_shape = (1, 3, 28, 28)
                    self.similarity_manager.random_input_dim = 3 * 28 * 28
                    logging.warning("无法从客户端数据检测输入维度，使用默认值")
            else:
                self.similarity_manager.input_shape = (1, 3, 28, 28)
                self.similarity_manager.random_input_dim = 3 * 28 * 28
                logging.warning("没有客户端，使用默认输入维度")
        
        # 设置随机输入数量
        self.similarity_manager.num_random_inputs = 50  # 减少数量以提高效率
        
        logging.info(f"设置模型行为相似度计算参数: 输入形状={self.similarity_manager.input_shape}, 随机输入数量={self.similarity_manager.num_random_inputs}")
    
    def train_round(self, round_idx):
        """执行一轮训练"""
        self.current_round = round_idx
        logging.info(f"\n开始第 {round_idx + 1} 轮训练")
        
        # 修改判断条件：使用round_idx + 1来判断是否是聚合轮次
        do_global_agg = self.enable_global_agg and round_idx > 0 and (round_idx + 1) % self.global_agg_rounds == 0
        
        # 添加明确的日志记录
        logging.info(f"全局聚合状态: enable_global_agg={self.enable_global_agg}, round_idx={round_idx}, 是否执行全局聚合={do_global_agg}")
    
        if round_idx == 0:
            logging.info("第 1 轮 - 仅进行本地训练")
            for client_id, client in self.clients.items():
                logging.info(f"训练客户端 {client_id}")
                client.train(epochs=self.epochs, after_global_agg=False)
                logging.info(f"客户端 {client_id} 训练完成")
        elif do_global_agg:
            # 执行全局聚合
            logging.info(f"第 {round_idx + 1} 轮 - 执行全局模型聚合")
            
            # 保存聚合前的模型和性能指标
            pre_agg_models = {}
            pre_agg_metrics = {}
            for client_id, client in self.clients.items():
                pre_agg_models[client_id] = {k: v.clone() for k, v in client.model.state_dict().items()}
                pre_agg_metrics[client_id] = client.metrics.copy()
            
            # 执行全局聚合
            self._perform_global_aggregation()
            
            # 全局聚合后训练并评估
            for client_id, client in self.clients.items():
                logging.info(f"训练客户端 {client_id}")
                client.train(epochs=self.epochs, after_global_agg=True)
                logging.info(f"客户端 {client_id} 训练完成")
                
                # 检查聚合后性能是否提升
                if client.metrics['accuracy'] < pre_agg_metrics[client_id]['accuracy'] - 1.0:  # 如果准确率下降超过1%
                    logging.info(f"客户端 {client_id} 全局聚合后性能下降 "
                                f"({pre_agg_metrics[client_id]['accuracy']:.2f}% -> {client.metrics['accuracy']:.2f}%)，"
                                f"恢复聚合前模型")
                    # 恢复聚合前的模型和性能指标
                    client.model.load_state_dict(pre_agg_models[client_id])
                    client.metrics = pre_agg_metrics[client_id]
                else:
                    logging.info(f"客户端 {client_id} 全局聚合后性能变化: "
                                f"{pre_agg_metrics[client_id]['accuracy']:.2f}% -> {client.metrics['accuracy']:.2f}%")
        else:
            # 从第二轮开始执行模型迁移
            logging.info(f"第 {round_idx + 1} 轮 - 开始模型迁移")
            
            # 获取上一轮的客户端性能指标
            client_performances = {
                client_id: {
                    'accuracy': client.metrics['accuracy'],
                    'loss': client.metrics['loss'],
                    'f1_score': client.metrics.get('f1_score', 0),
                    'train_accuracy': client.metrics.get('train_accuracy', 0)
                }
                for client_id, client in self.clients.items()
            }
            
            # 使用SimilarityManager生成迁移策略
            migration_strategies, _ = self.similarity_manager.generate_migration_strategy(
                performance_metrics=client_performances,
                round_num=round_idx
            )
            
            # 执行模型迁移
            self._execute_model_migration(migration_strategies)
            logging.info("模型迁移完成，开始本轮训练")
            
            # 训练所有客户端
            for client_id, client in self.clients.items():
                logging.info(f"训练客户端 {client_id}")
                client.train(epochs=self.epochs, after_global_agg=False)
                logging.info(f"客户端 {client_id} 训练完成")
        
        # 获取客户端性能指标并更新
        client_performances = {
            client_id: {
                'accuracy': client.metrics['accuracy'],
                'loss': client.metrics['loss'],
                'f1_score': client.metrics.get('f1_score', 0),
                'train_accuracy': client.metrics.get('train_accuracy', 0)
            }
            for client_id, client in self.clients.items()
        }
        
        # 收集客户端信息
        self._collect_client_info()
        
        # 计算平均性能
        metrics = self.get_round_metrics()
        logging.info(f"\n轮次 {round_idx + 1} 完成:")
        logging.info(f"平均准确率: {metrics['average_accuracy']:.2f}%")
        
        # 检查是否收敛
        # 修改收敛检查部分
        is_converged = self.similarity_manager.check_convergence(client_performances)
        if is_converged:
            logging.info("检测到训练收敛，但将继续训练以完成所有轮次")
        
        # 始终返回False，确保训练继续进行
        return False
    
    def _execute_model_migration(self, migration_strategies):
        """执行模型迁移策略"""
        logging.info("执行模型迁移策略")
        
        migration_count = 0
        for client_id, strategy in migration_strategies.items():
            if not strategy['sources']:
                logging.info(f"客户端 {client_id} 没有迁移源，跳过迁移")
                continue
                
            logging.info(f"客户端 {client_id} 接收来自 {len(strategy['sources'])} 个客户端的模型")
            
            # 将自身权重信息传递给客户端
            self_weight = strategy.get('self_weight', self.similarity_manager.self_weight)
            self.clients[client_id].self_weight = self_weight  # 设置客户端的自身权重
            
            # 逐个发送模型和权重
            for i, source_id in enumerate(strategy['sources']):
                if source_id not in self.clients or client_id not in self.clients:
                    logging.warning(f"迁移失败: 客户端 {source_id} 或 {client_id} 不存在")
                    continue
                    
                source_model = self.clients[source_id].model
                weight = strategy['weights'][i]
                self.clients[client_id].receive_model(source_model, weight)
                migration_count += 1
                
                # 记录迁移历史
                if client_id in self.similarity_manager.client_history:
                    self.similarity_manager.client_history[client_id]['migration_history'].append({
                        'round': self.current_round,
                        'source': source_id,
                        'weight': weight
                    })
        
        logging.info(f"本轮共执行 {migration_count} 次模型迁移")
    
    def initialize_federation(self):
        """初始化联邦学习系统"""
        logging.info("初始化联邦学习系统...")
        
        # 1. 如果启用全局聚合，初始化全局模型
        if self.enable_global_agg and len(self.clients) > 0:
        
            # 根据数据集类型确定输入通道数           
            if self.dataset_type in ['mnist', 'fashion_mnist']:
                in_channels = 1  # MNIST和Fashion-MNIST是灰度图像
                input_size = 28  # MNIST和Fashion-MNIST是28x28
            else:  # cifar10, cifar100等
                in_channels = 3  # CIFAR-10是彩色图像
                input_size = 32  # CIFAR-10是32x32
                
            logging.info(f"根据数据集类型 {self.dataset_type} 设置输入通道数: {in_channels}, 输入尺寸: {input_size}")
            
            # 创建全局模型
            self.global_model = get_model(
                model_type='resnet18',  # 可以根据需要选择不同的模型类型
                num_classes=self.num_classes,
                in_channels=in_channels,
                input_size=input_size
            )
            
            # 初始化模型参数
            self._reinitialize_model_parameters(self.global_model)
            self.global_model.to(self.device)
            logging.info(f"初始化全局模型完成（使用{self.dataset_type}数据集参数）")
            
            # 移除全局模型分发部分，让客户端保持各自的初始模型
            logging.info("跳过全局模型分发，保留客户端各自的初始模型")
        
        # 2. 更新客户端信息到相似度管理器前，确保客户端模型的输入通道数正确
        for client_id, client in self.clients.items():
            # 检查客户端模型的第一个卷积层
            first_conv_layer = None
            for name, module in client.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    first_conv_layer = module
                    logging.info(f"客户端 {client_id} - 找到第一个卷积层: {name}, 输入通道数: {module.in_channels}")
                    break
            
            # 获取客户端数据的输入通道数
            sample_data = next(iter(client.train_data))[0]
            input_channels = sample_data.size(1)
            input_size = sample_data.size(-1)
            
            logging.info(f"客户端 {client_id} - 数据输入通道数: {input_channels}, 输入尺寸: {input_size}")
            
            # 如果模型的输入通道数与数据不匹配，重新创建模型
            if first_conv_layer is not None and first_conv_layer.in_channels != input_channels:
                logging.warning(f"客户端 {client_id} - 模型输入通道数 ({first_conv_layer.in_channels}) 与数据输入通道数 ({input_channels}) 不匹配，重新创建模型")
                
                # 获取模型类型
                model_type = client.model_type if hasattr(client, 'model_type') else 'resnet18'
                
                # 创建新模型
                new_model = get_model(
                    model_type=model_type,
                    num_classes=self.num_classes,
                    in_channels=input_channels,
                    input_size=input_size
                )
                
                # 将新模型移动到与原模型相同的设备上
                new_model.to(client.device)
                
                # 更新客户端的模型
                client.model = new_model
                
                # 重新初始化优化器
                client.optimizer = torch.optim.Adam(
                    client.model.parameters(),
                    lr=client.learning_rate
                )
                
                logging.info(f"客户端 {client_id} - 已重新创建模型，输入通道数: {input_channels}")
            
            # 更新客户端信息到相似度管理器
            self.similarity_manager.update_client_info(
                client_id=client_id,
                model=client.model,  # 使用客户端自己的模型
                data_distribution=client.get_data_distribution(),
                resources={
                    'compute_power': client.attributes['compute_power'],
                    'device_resources': client.attributes['device_resources']
                }
            )
            
            logging.info(f"客户端 {client_id} 初始化完成")
        
        # 3. 初始化相似度矩阵
        self.similarity_manager.calculate_similarity_matrix()
        
        # 4. 初始化搜索空间
        for client_id in self.clients.keys():
            if client_id not in self.similarity_manager.search_spaces:
                self.similarity_manager.search_spaces[client_id] = self.similarity_manager._initialize_search_space(client_id)
        
        logging.info("联邦学习系统初始化完成")
    
    def _reinitialize_model_parameters(self, model):
        """重新初始化模型参数"""
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
    
    def get_round_metrics(self):
        """获取当前轮次的性能指标"""
        metrics = {
            'average_accuracy': np.mean([client.metrics['accuracy'] for client in self.clients.values()]),
            'average_f1': np.mean([client.metrics.get('f1_score', 0) for client in self.clients.values()]),
            'client_accuracies': {cid: client.metrics['accuracy'] for cid, client in self.clients.items()}
        }
        return metrics

    def register_client(self, client):
        """注册客户端到服务器"""
        self.clients[client.client_id] = client
        client.server = self  # 设置客户端的服务器引用

    def _collect_client_info(self):
        """收集客户端信息"""
        for client_id, client in self.clients.items():
            # 更新客户端信息到相似度管理器
            self.similarity_manager.update_client_info(
                client_id=client_id,
                resources={
                    'compute_power': client.attributes['compute_power'],
                    'device_resources': client.attributes['device_resources']
                },
                data_distribution=client.get_data_distribution(),
                model=client.model
            )
    
    def run_federation(self, num_rounds=10):
        """运行联邦学习过程"""
        logging.info(f"开始联邦学习，计划执行 {num_rounds} 轮")
        
        # 初始化联邦系统
        self.initialize_federation()
        
        # 执行多轮训练
        for round_idx in range(num_rounds):
            is_converged = self.train_round(round_idx)
            
            # 如果收敛，提前结束
            if is_converged:
                logging.info(f"训练在第 {round_idx + 1} 轮收敛，提前结束")
                break
        
        logging.info("联邦学习完成")
        
        # 返回最终性能指标
        return self.get_round_metrics()
    
    def get_similarity_matrix(self):
        """获取相似度矩阵"""
        return self.similarity_manager.similarity_matrix
    
    def get_migration_strategies(self):
        """获取最近一轮的迁移策略"""
        return self.similarity_manager.migration_strategies
    
    def get_client_performance_history(self):
        """获取客户端性能历史"""
        return {
            client_id: history['performance_history'] 
            for client_id, history in self.similarity_manager.client_history.items()
        }
    
    def set_dataset_type(self, dataset_type):
        """设置数据集类型，用于调整模型行为相似度计算参数"""
        self.dataset_type = dataset_type
        self.setup_model_behavior_params()
        logging.info(f"设置数据集类型为 {dataset_type}，并更新了模型行为相似度计算参数")
    
    def set_similarity_weights(self, model_weight=0.2, data_weight=0.7, resource_weight=0.1):
        """设置相似度权重"""
        self.similarity_manager.similarity_weights = {
            'model': model_weight,
            'data': data_weight,
            'resource': resource_weight
        }
        logging.info(f"设置相似度权重: 模型={model_weight}, 数据={data_weight}, 资源={resource_weight}")
        
        # 如果已经计算过相似度矩阵，则重新计算
        if self.similarity_manager.similarity_matrix is not None:
            self.similarity_manager.calculate_similarity_matrix()

    def set_exploration_rate(self, rate):
        """设置探索率"""
        self.exploration_rate = rate
        # 更新所有客户端的探索率
        for client_id in self.clients.keys():
            if client_id in self.similarity_manager.search_spaces:
                self.similarity_manager.search_spaces[client_id]['exploration_rate'] = rate
        logging.info(f"设置探索率为 {rate}")
    
    def set_max_migration_models(self, max_models):
        """设置最大迁移模型数量"""
        self.max_migration_models = max_models
        self.similarity_manager.max_migration_models = max_models
        logging.info(f"设置最大迁移模型数量为 {max_models}")
    
    def get_search_space_info(self):
        """获取搜索空间信息"""
        info = {}
        for client_id, search_space in self.similarity_manager.search_spaces.items():
            info[client_id] = {
                'exploration_rate': search_space.get('exploration_rate', self.exploration_rate),
                'best_reward': search_space.get('best_reward', -float('inf')),
                'history_length': len(search_space.get('history', [])),
                'has_best_strategy': search_space.get('best_strategy') is not None
            }
        return info

    def apply_migration_strategy(self, client_id, strategy):
        """应用迁移策略到客户端模型"""
        if not strategy['sources']:
            logging.info(f"客户端 {client_id} 没有迁移源，跳过迁移")
            return False
        
        client = self.clients[client_id]
        client_model = client.model
        
        # 从策略中获取自身权重，如果没有则从相似度管理器获取默认值
        self_weight = strategy.get('self_weight', self.similarity_manager.self_weight)
        
        # 创建模型参数的副本
        own_state_dict = {k: v.clone() for k, v in client_model.state_dict().items()}
        
        # 应用迁移策略
        with torch.no_grad():
            # 先将模型重置为自身权重，只对浮点类型参数应用权重
            for key, param in own_state_dict.items():
                # 只对浮点类型参数应用权重
                if param.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                    own_state_dict[key] *= self_weight
                
            # 然后添加迁移模型的权重
            for i, source_id in enumerate(strategy['sources']):
                if source_id not in self.clients:
                    logging.warning(f"迁移源 {source_id} 不存在，跳过")
                    continue
                    
                source_model = self.clients[source_id].model
                source_weight = strategy['weights'][i]
                
                # 获取源模型的状态字典
                source_state_dict = source_model.state_dict()
                
                # 合并参数，只对浮点类型参数应用权重
                for key in own_state_dict:
                    if key in source_state_dict:
                        # 只对浮点类型参数应用权重
                        if own_state_dict[key].dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                            own_state_dict[key] += source_state_dict[key] * source_weight
                        else:
                            # 对于非浮点类型参数，直接使用源模型的参数
                            if self_weight < 0.5:  # 如果自身权重小于0.5，则使用源模型的参数
                                own_state_dict[key] = source_state_dict[key]
        
        # 加载合并后的参数
        client_model.load_state_dict(own_state_dict)
        
        # 记录迁移信息
        logging.info(f"客户端 {client_id} 完成迁移，自身权重: {self_weight:.2f}, 迁移源: {strategy['sources']}, 迁移权重: {[f'{w:.2f}' for w in strategy['weights']]}")
        
        return True

    def _perform_global_aggregation(self):
        """执行全局模型聚合"""
        logging.info("开始执行全局模型聚合...")
        
        # 获取所有客户端的模型参数
        client_states = []
        client_weights = []
        
        # 根据客户端性能分配权重
        total_accuracy = 0
        for client_id in self.clients:
            total_accuracy += self.clients[client_id].metrics['accuracy']
        
        # 如果总准确率为0，则使用均匀权重
        if total_accuracy == 0:
            weights = [1.0 / len(self.clients)] * len(self.clients)
        else:
            # 根据准确率分配权重
            weights = []
            for client_id in self.clients:
                weight = self.clients[client_id].metrics['accuracy'] / total_accuracy
                weights.append(weight)
                client_states.append(self.clients[client_id].model.state_dict())
                client_weights.append(weight)
        
        logging.info(f"客户端权重分配: {['%.4f' % w for w in weights]}")
        
        # 初始化全局模型状态
        global_state = {}
        for key in client_states[0].keys():
            # 创建与第一个客户端相同类型和形状的零张量
            global_state[key] = torch.zeros_like(client_states[0][key])
            
        # 聚合模型参数
        for client_idx, state_dict in enumerate(client_states):
            weight = client_weights[client_idx]
            for key in global_state:
                # 根据参数类型选择不同的聚合方式
                if global_state[key].dtype == torch.long:
                    # 对于长整型参数，使用加权投票
                    if client_idx == 0:
                        global_state[key] = state_dict[key].clone().float() * weight
                    else:
                        global_state[key] += state_dict[key].float() * weight
                    # 最后转回长整型
                    if client_idx == len(client_states) - 1:
                        global_state[key] = global_state[key].long()
                else:
                    # 对于浮点型参数，使用加权平均
                    global_state[key] += state_dict[key] * weight
        
        # 更新全局模型
        if self.global_model is not None:
            self.global_model.load_state_dict(global_state)
        
        # 更新所有客户端的模型
        for client_id in self.clients:
            self.clients[client_id].model.load_state_dict(global_state)

    