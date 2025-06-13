import numpy as np
import logging
from sklearn.neighbors import KDTree
import torch
import copy
from collections import defaultdict
import random
from models import SimpleCNN,ResNet18
import torch.nn as nn

class SimilarityManager:
    def __init__(self, server=None, data_weight=0.6, behavior_weight=0.2, resource_weight=0.2):
        self.server = server
        
        # 添加客户端相关的字典初始化
        self.client_models = {}             # 存储客户端模型
        self.client_model_behaviors = {}    # 存储模型行为
        self.client_data_distributions = {} # 存储数据分布
        self.client_resources = {}          # 存储资源信息
        
        # 添加权重相关的属性初始化
        self.self_weight = 0.7      # 自身权重
        self.migration_weight = 0.3  # 迁移权重
        
        # 相似度权重
        # 使用传入的权重值
        self.similarity_weights = {
            'model': behavior_weight,    # 模型行为相似度权重
            'data': data_weight,         # 数据分布相似度权重
            'resource': resource_weight  # 资源相似度权重
        }
        self.similarity_matrix = None
        
        # 添加资源相关参数
        self.communication_budget = 1000  # 全局通信预算
        self.computation_budget = 100     # 全局计算预算
        self.model_size = 1.0             # 模型大小(MB)
        self.max_n = None                 # 最大迁移数量
        self.search_spaces = {}           # 搜索空间
        self.rewards = {}                 # 奖励记录
        # 修改收敛判断参数
        self.convergence_threshold = 0.001  # 降低阈值，允许更小的改进
        self.patience = 10  # 增加耐心值，给更多训练机会
        self.min_accuracy = 70.0  # 添加最小准确率要求
        self.best_global_loss = float('inf')
        self.non_improving_rounds = 0
        self.client_comm_budgets = {}     # 客户端通信预算
        self.client_comp_budgets = {}     # 客户端计算预算

        # 从FMIServer移植的参数
        self.min_confidence_threshold = 0.7  # 最小置信度阈值，设置为0.7
        self.max_confidence_threshold = 1.0  # 最大置信度阈值
        self.max_migration_models = 5     # 最大迁移模型数量

        # 客户端历史记录
        self.client_history = defaultdict(lambda: {
            'performance_history': [],  # 历史性能记录
            'improvement_rate': [],     # 性能提升率
            'migration_history': []     # 迁移历史
        })
        
        # 模型行为相似度计算参数
        self.num_random_inputs = 100     # 用于生成模型行为的随机输入数量
        self.random_input_dim = 784      # 随机输入维度 (例如 MNIST 28x28=784)
        self.random_inputs = None        # 存储随机生成的输入

    def update_client_info(self, client_id, model=None, data_distribution=None, resources=None):
        """更新客户端信息"""
        logging.info(f"更新客户端 {client_id} 信息")
        
        if model is not None:
            # 存储模型状态字典而不是模型对象
            self.client_models[client_id] = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            logging.info(f"- 更新了模型信息")
            
            # 计算模型行为特征向量
            self._update_model_behavior(client_id, model)
        
        if data_distribution is not None:
            self.client_data_distributions[client_id] = data_distribution
            logging.info(f"- 更新了数据分布信息: {data_distribution}")
        
        if resources is not None:
            self.client_resources[client_id] = resources
            logging.info(f"- 更新了资源信息: {resources}")
        
        # 如果所有信息都已更新，重新计算相似度矩阵
        if client_id in self.client_models and client_id in self.client_data_distributions and client_id in self.client_resources:
            logging.info(f"客户端 {client_id} 所有信息已更新，准备重新计算相似度矩阵")
        
        return True
    
    def _update_model_behavior(self, client_id, model):
        """更新模型行为特征向量"""
        # 获取全局设备
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 创建模型的深拷贝，避免修改原始模型
        model_copy = copy.deepcopy(model)
        model_copy.to(global_device)  # 显式移动到全局设备
        model_copy.eval()
         # 获取模型所在设备
        device = next(model_copy.parameters()).device
        logging.info(f"模型当前在设备: {device}")
        # 获取模型类型
        model_type = model_copy.__class__.__name__
        logging.info(f"处理模型类型: {model_type}")

        # 获取正确的输入通道数
        if client_id in self.server.clients:
            client = self.server.clients[client_id]
            in_channels = client.input_channels
            input_size = client.input_size
        else:
            # 如果无法从客户端获取，则从模型中获取
            first_conv = None
            for module in model_copy.modules():
                if isinstance(module, torch.nn.Conv2d):
                    first_conv = module
                    break
            in_channels = first_conv.in_channels if first_conv else 1
            input_size = 28  # 默认值

        logging.info(f"使用输入通道数: {in_channels}, 输入尺寸: {input_size}")

        # 检查模型的第一个卷积层
        first_conv_layer = None
        first_conv_name = None
        for name, module in model_copy.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv_layer = module
                logging.info(f"找到第一个卷积层: {name}, 输入通道数: {module.in_channels}, 权重形状: {module.weight.shape}")
                break

        # 只有在通道数不匹配时才进行修复
        if first_conv_layer is not None and first_conv_layer.in_channels != in_channels:
            logging.warning(f"检测到通道数不匹配: 当前为 {first_conv_layer.in_channels}，需要 {in_channels}")
            
            # 保存原始参数
            old_weight = first_conv_layer.weight.data
            out_channels = old_weight.shape[0]
            kernel_size = old_weight.shape[2:]
            
            # 创建新的卷积层，使用正确的输入通道数
            new_conv = torch.nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size[0],
                stride=first_conv_layer.stride,
                padding=first_conv_layer.padding,
                bias=first_conv_layer.bias is not None
            )

            # 初始化新权重
            if in_channels > first_conv_layer.in_channels:
                # 如果需要更多通道，复制现有通道
                new_weight = torch.zeros(out_channels, in_channels, *kernel_size)
                for i in range(in_channels):
                    new_weight[:, i] = old_weight[:, i % first_conv_layer.in_channels]
            else:
                # 如果需要更少通道，取平均值或选择部分通道
                new_weight = old_weight[:, :in_channels]
                
            new_conv.weight.data = new_weight
            if first_conv_layer.bias is not None:
                new_conv.bias.data = first_conv_layer.bias.data

            # 根据模型类型替换第一个卷积层
            if hasattr(model_copy, 'conv1'):
                model_copy.conv1 = new_conv
                logging.info(f"已修复模型的conv1层，新权重形状: {new_conv.weight.shape}")
            elif model_type.lower().startswith('resnet'):
                model_copy.conv1 = new_conv
                logging.info(f"已修复ResNet的conv1层，新权重形状: {new_conv.weight.shape}")
            elif model_type.lower().startswith('vgg'):
                if hasattr(model_copy, 'features') and len(model_copy.features) > 0:
                    for i, layer in enumerate(model_copy.features):
                        if isinstance(layer, torch.nn.Conv2d):
                            model_copy.features[i] = new_conv
                            logging.info(f"已修复VGG的第一个卷积层，新权重形状: {new_conv.weight.shape}")
                            break

        # 处理全连接层的输入维度
        try:
            # 使用正确通道数的示例输入，并确保在正确的设备上
            dummy_input = torch.randn(1, in_channels, input_size, input_size).to(device)  # 修改这里，确保在正确设备上
            
            # 获取特征提取部分
            feature_extractor = None
            if model_type == 'SimpleCNN':
                if hasattr(model_copy, 'pool2'):
                    feature_extractor = nn.Sequential(
                        model_copy.conv1,
                        model_copy.relu1,
                        model_copy.pool1,
                        model_copy.conv2,
                        model_copy.relu2,
                        model_copy.pool2
                    ).to(device)
            elif model_type.lower().startswith('ResNet'):
                # 为 ResNet 模型创建特征提取器
                try:
                    # 尝试直接使用模型的前向传播到 avgpool 之前
                    # 这是一个安全的方法，不需要直接访问 conv1
                    class FeatureExtractor(nn.Module):
                        def __init__(self, model):
                            super(FeatureExtractor, self).__init__()
                            self.model = model
                            
                        def forward(self, x):
                            # 对于 ResNet 模型，我们只需要获取特征，不需要分类层
                            # 这种方法适用于各种 ResNet 变体
                            x = self.model.conv1(x)
                            x = self.model.bn1(x)
                            x = self.model.relu(x)
                            x = self.model.maxpool(x)
                            
                            x = self.model.layer1(x)
                            x = self.model.layer2(x)
                            x = self.model.layer3(x)
                            x = self.model.layer4(x)
                            
                            x = self.model.avgpool(x)
                            return x
                            
                    feature_extractor = FeatureExtractor(model_copy).to(device)
                except Exception as e:
                    logging.warning(f"创建 ResNet 特征提取器失败: {e}")
                    # 使用更通用的方法
                    feature_extractor = None
            elif model_type.lower().startswith('vgg'):
                if hasattr(model_copy, 'features'):
                    feature_extractor = model_copy.features.to(device)

            # 计算特征大小并更新全连接层
            if feature_extractor is not None:
                with torch.no_grad():
                    features = feature_extractor(dummy_input)
                    feature_size = features.view(1, -1).size(1)
                    logging.info(f"计算得到的特征大小: {feature_size}")

                    # 更新相应模型的全连接层
                    if model_type == 'SimpleCNN' and hasattr(model_copy, 'fc1'):
                        old_fc1 = model_copy.fc1
                        model_copy.fc1 = nn.Linear(feature_size, old_fc1.out_features).to(device)
                    elif model_type.lower().startswith('ResNet') and hasattr(model_copy, 'fc'):
                        old_fc = model_copy.fc
                        model_copy.fc = nn.Linear(feature_size, old_fc.out_features).to(device)
                    elif model_type.lower().startswith('vgg') and hasattr(model_copy, 'classifier'):
                        if isinstance(model_copy.classifier, nn.Sequential):
                            for i, layer in enumerate(model_copy.classifier):
                                if isinstance(layer, nn.Linear):
                                    old_fc = layer
                                    model_copy.classifier[i] = nn.Linear(feature_size, old_fc.out_features).to(device)
                                    break
        except Exception as e:
            logging.error(f"更新全连接层失败: {e}")

        # 生成随机输入
        self.random_inputs = [torch.randn(1, in_channels, input_size, input_size).to(device) 
                            for _ in range(self.num_random_inputs)]

        # 获取模型行为
        behavior_vector = []
        device = next(model_copy.parameters()).device

        with torch.no_grad():
            for random_input in self.random_inputs:
                try:
                    # 确保输入在正确的设备上
                    random_input = random_input.to(device)  # 添加这行
                    output = model_copy(random_input)
                    _, predicted = torch.max(output, 1)
                    behavior_vector.append(predicted.item())
                except Exception as e:
                    logging.error(f"模型预测失败: {e}")
                    behavior_vector.append(-1)

        # 存储行为向量
        self.client_model_behaviors[client_id] = np.array(behavior_vector)
        logging.info(f"已更新模型行为特征向量，长度: {len(behavior_vector)}")
  
    
    def _generate_random_inputs(self):
        """生成用于测试模型行为的随机输入"""
        logging.info(f"生成 {self.num_random_inputs} 个随机输入用于模型行为分析")
        
        self.random_inputs = []
        
        # 获取第一个客户端的模型来确定输入维度
        if self.server and hasattr(self.server, 'clients') and len(self.server.clients) > 0:
            first_client = list(self.server.clients.values())[0]
            model = first_client.model
            
            # 获取模型的输入维度
            in_channels = None
            input_size = None
            
            # 尝试从模型的第一层获取输入维度
            first_layer = None
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    first_layer = module
                    break
            
            if first_layer is not None:
                if isinstance(first_layer, torch.nn.Conv2d):
                    in_channels = first_layer.in_channels
                    input_size = 28 if in_channels == 1 else 32
                elif isinstance(first_layer, torch.nn.Linear):
                    in_channels = 1
                    input_size = int(np.sqrt(first_layer.in_features))
            
            # 如果无法从模型结构获取，则根据数据集类型设置
            if in_channels is None:
                if hasattr(self.server, 'dataset_type'):
                    in_channels = 1 if self.server.dataset_type in ['mnist', 'fashion_mnist'] else 3
                    input_size = 28 if self.server.dataset_type in ['mnist', 'fashion_mnist'] else 32
                else:
                    in_channels = 3
                    input_size = 32
            
            logging.info(f"使用输入通道数: {in_channels}, 输入尺寸: {input_size}")
            
            # 生成随机输入
            for _ in range(self.num_random_inputs):
                random_input = torch.randn(1, in_channels, input_size, input_size)
                self.random_inputs.append(random_input)
        else:
            logging.warning("无法获取模型信息，使用默认输入维度")
            # 使用默认维度
            for _ in range(self.num_random_inputs):
                random_input = torch.randn(1, 3, 32, 32)
                self.random_inputs.append(random_input)

    def calculate_model_similarity(self, client_id1, client_id2):
        """计算两个客户端模型行为之间的相似度"""
        if client_id1 not in self.client_model_behaviors or client_id2 not in self.client_model_behaviors:
            logging.warning(f"计算模型行为相似度失败: 客户端 {client_id1} 或 {client_id2} 的模型行为信息不存在")
            return 0.0
        
        behavior1 = self.client_model_behaviors[client_id1]
        behavior2 = self.client_model_behaviors[client_id2]
        
        # 使用KD树计算行为相似度
        try:
            # 将行为向量重塑为KD树所需的形状
            behavior1_reshaped = behavior1.reshape(1, -1)
            behavior2_reshaped = behavior2.reshape(1, -1)
            
            # 构建KD树
            tree = KDTree(behavior1_reshaped)
            
            # 计算距离
            dist, _ = tree.query(behavior2_reshaped)
            
            # 将距离转换为相似度 (0-1范围)
            # 确保最大距离合理设置
            max_dist = self.num_random_inputs  # 最大可能距离
            
            # 确保相似度在[0,1]范围内
            similarity = max(0.0, min(1.0, 1 - (dist[0][0] / max_dist)))
            
            return float(similarity)
        except Exception as e:
            logging.error(f"KD树计算模型行为相似度失败: {e}")
            
            # 退化为直接计算匹配率
            matches = np.sum(behavior1 == behavior2)
            similarity = matches / len(behavior1)
            return float(similarity)

    def calculate_data_similarity(self, client_id1, client_id2):
        """计算两个客户端数据分布之间的相似度 (使用余弦相似度)"""
        if client_id1 not in self.client_data_distributions or client_id2 not in self.client_data_distributions:
            logging.warning(f"计算数据相似度失败: 客户端 {client_id1} 或 {client_id2} 的数据分布信息不存在")
            return 0.0
    
        dist1 = self.client_data_distributions[client_id1]
        dist2 = self.client_data_distributions[client_id2]
    
        # 检查数据分布格式并尝试转换
        if not isinstance(dist1, (dict, list, np.ndarray)) or not isinstance(dist2, (dict, list, np.ndarray)):
            # 尝试将数据转换为数组
            try:
                if hasattr(dist1, '__iter__'):
                    dist1 = np.array(list(dist1))
                else:
                    logging.warning(f"数据分布格式错误: 客户端 {client_id1} 的数据分布无法转换为数组")
                    return 0.0
            
                if hasattr(dist2, '__iter__'):
                    dist2 = np.array(list(dist2))
                else:
                    logging.warning(f"数据分布格式错误: 客户端 {client_id2} 的数据分布无法转换为数组")
                    return 0.0
            except Exception as e:
                logging.warning(f"数据分布格式错误: 无法转换为可比较的格式 - {e}")
                return 0.0
    
        # 如果是字典格式，转换为列表
        if isinstance(dist1, dict) and isinstance(dist2, dict):
            # 确保两个字典有相同的键
            keys = set(dist1.keys()).intersection(set(dist2.keys()))
            if not keys:
                logging.warning(f"数据分布没有共同的键: 客户端 {client_id1} 和 {client_id2}")
                return 0.0
        
            # 提取值
            vec1 = np.array([dist1[k] for k in keys])
            vec2 = np.array([dist2[k] for k in keys])
        else:
            # 如果是列表或数组格式，直接使用
            vec1 = np.array(dist1) if not isinstance(dist1, np.ndarray) else dist1
            vec2 = np.array(dist2) if not isinstance(dist2, np.ndarray) else dist2
        
            # 确保两个向量长度相同
            min_len = min(len(vec1), len(vec2))
            if min_len == 0:
                logging.warning(f"数据分布为空: 客户端 {client_id1} 或 {client_id2}")
                return 0.0
            
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
    
        # 计算余弦相似度
        try:
            # 计算点积
            dot_product = np.dot(vec1, vec2)
        
            # 计算向量范数
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
        
            # 避免除零错误
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            
            # 计算余弦相似度
            cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        
            # 确保结果在[0,1]范围内
            return max(0.0, min(1.0, cosine_similarity))
        except Exception as e:
            logging.warning(f"计算数据相似度失败: {e}")
            return 0.0    

    def calculate_resource_similarity(self, client_id1, client_id2):
        """计算两个客户端资源之间的相似度 (使用绝对值之差)"""
        if client_id1 not in self.client_resources or client_id2 not in self.client_resources:
            logging.warning(f"计算资源相似度失败: 客户端 {client_id1} 或 {client_id2} 的资源信息不存在")
            return 0.0
        
        res1 = self.client_resources[client_id1]
        res2 = self.client_resources[client_id2]
        
        # 确保资源信息是字典格式
        if not isinstance(res1, dict) or not isinstance(res2, dict):
            logging.warning(f"资源信息格式错误: 应为字典")
            return 0.0
        
        # 提取资源特征
        similarities = []
        
        # 处理计算能力
        if 'compute_power' in res1 and 'compute_power' in res2:
            cp1 = res1['compute_power']
            cp2 = res2['compute_power']
            # 计算绝对值之差并转换为相似度
            diff = abs(cp1 - cp2)
            # 假设最大差异为100
            max_diff = 100
            sim = 1 - (diff / max_diff)
            similarities.append(sim)
        
        # 处理设备资源
        if 'device_resources' in res1 and 'device_resources' in res2:
            dr1 = res1['device_resources']
            dr2 = res2['device_resources']
            # 计算绝对值之差并转换为相似度
            diff = abs(dr1 - dr2)
            # 假设最大差异为100
            max_diff = 100
            sim = 1 - (diff / max_diff)
            similarities.append(sim)
        
        # 如果没有提取到特征，返回0
        if not similarities:
            return 0.0
        
        # 返回平均相似度
        return max(0.0, min(1.0, np.mean(similarities)))

    def compute_similarity_matrix(self):
        """计算客户端之间的相似度矩阵"""
        client_ids = list(self.client_models.keys())
        n_clients = len(client_ids)
        
        # 添加调试日志
        logging.info(f"开始计算相似度矩阵，共 {n_clients} 个客户端")
        
        # 初始化相似度矩阵
        self.similarity_matrix = np.zeros((n_clients, n_clients))
        
        # 计算各种相似度并组合
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    self.similarity_matrix[i][j] = 1.0  # 自己与自己的相似度为1
                    continue
                    
                client_i = client_ids[i]
                client_j = client_ids[j]
                
                # 计算模型行为相似度
                model_sim = self.calculate_model_similarity(client_i, client_j)
                
                # 计算数据分布相似度
                data_sim = self.calculate_data_similarity(client_i, client_j)
                
                # 计算资源相似度
                resource_sim = self.calculate_resource_similarity(client_i, client_j)
                
                # 加权组合相似度
                combined_sim = (
                    self.similarity_weights['model'] * model_sim +
                    self.similarity_weights['data'] * data_sim +
                    self.similarity_weights['resource'] * resource_sim
                )
                
                self.similarity_matrix[i][j] = combined_sim
                
                # 添加调试日志
                logging.info(f"客户端 {client_i} 和 {client_j} 的相似度: {combined_sim:.4f} (模型:{model_sim:.4f}, 数据:{data_sim:.4f}, 资源:{resource_sim:.4f})")
        
        logging.info("相似度矩阵计算完成")
        return self.similarity_matrix

    def optimize_migration_params(self, performance_metrics=None):
        """优化每个客户端的迁移参数 Ni 和 Ci"""
        params = {}
        client_ids = list(self.client_models.keys())
        
        for client_id in client_ids:
            if client_id not in self.client_history:
                self.client_history[client_id] = {
                    'performance_history': [],
                    'improvement_rate': [],
                    'migration_history': []
                }
                
            if performance_metrics and client_id in performance_metrics:
                current_performance = performance_metrics[client_id]['accuracy'] / 100
                history = self.client_history[client_id]
                
                # 更新历史记录
                history['performance_history'].append(current_performance)
                
                # 计算性能提升率
                if len(history['performance_history']) > 1:
                    improvement = (current_performance - history['performance_history'][-2])
                    history['improvement_rate'].append(improvement)
                
                # 动态调整置信度阈值
                Ci = self.adaptive_confidence_threshold(
                    current_performance,
                    history['improvement_rate'],
                    history['migration_history']
                )
                
                # 动态调整迁移数量
                Ni = self.adaptive_migration_number(
                    current_performance,
                    self.client_resources.get(client_id, {}).get('device_resources', 50) / 100,
                    history['improvement_rate']
                )
                
                # 获取相似度排序
                client_idx = client_ids.index(client_id)
                similarities = self.similarity_matrix[client_idx] if client_idx < len(self.similarity_matrix) else np.zeros(len(client_ids))
                sorted_indices = np.argsort(similarities)[::-1]
                
                params[client_id] = {
                    'Ni': Ni,
                    'Ci': Ci,
                    'sorted_similarities': sorted_indices
                }
            else:
                # 默认参数
                params[client_id] = {
                    'Ni': 1,  # 默认迁移源数量
                    'Ci': 0.5,  # 默认置信度阈值，设置为0.5
                    'sorted_similarities': np.argsort(self.similarity_matrix[client_ids.index(client_id)])[::-1] if client_ids.index(client_id) < len(self.similarity_matrix) else []
                }
        
        return params

    def adaptive_confidence_threshold(self, current_performance, improvement_history, migration_history):
        """自适应调整置信度阈值"""
        # 基础阈值
        base_threshold = self.min_confidence_threshold  # 使用0.7作为基础阈值
        
        if len(improvement_history) > 0:
            # 根据最近的性能提升趋势调整
            recent_improvements = improvement_history[-3:] if len(improvement_history) >= 3 else improvement_history
            avg_improvement = np.mean(recent_improvements)
            
            # 性能提升显著时，提高阈值要求
            if avg_improvement > 0.05:
                threshold_delta = 0.1
            # 性能提升停滞时，降低阈值要求
            elif avg_improvement < 0.01:
                threshold_delta = -0.1  # 降低阈值
            else:
                threshold_delta = 0.0  # 保持阈值不变
                
            # 根据当前性能调整基础阈值
            performance_factor = current_performance * 0.1  # 降低性能因素的影响
            
            final_threshold = base_threshold + threshold_delta + performance_factor
            
            # 确保在合理范围内
            return np.clip(final_threshold, self.min_confidence_threshold, self.max_confidence_threshold)
        
        return base_threshold

    def adaptive_migration_number(self, current_performance, resources, improvement_history):
        """自适应调整迁移数量"""
        # 基础迁移数量
        base_number = int(self.max_migration_models * resources)
        
        if len(improvement_history) > 0:
            # 计算最近的平均提升率
            recent_improvements = improvement_history[-3:] if len(improvement_history) >= 3 else improvement_history
            avg_improvement = np.mean(recent_improvements)
            
            # 根据提升率调整迁移数量
            if avg_improvement < 0.01:  # 性能提升较小
                number_factor = 1.5  # 增加迁移数量
            elif avg_improvement > 0.05:  # 性能提升显著
                number_factor = 0.8  # 减少迁移数量
            else:
                number_factor = 1.0
                
            # 考虑当前性能的影响
            performance_factor = 1 - current_performance  # 性能越差，迁移越多
            
            final_number = int(base_number * number_factor * (0.5 + performance_factor))
            
            # 确保在合理范围内
            return max(1, min(final_number, self.max_migration_models))
        
        return max(1, base_number)  # 确保至少有1个迁移源

    def generate_migration_strategy(self, performance_metrics=None, round_num=0):
        """使用启发式搜索算法生成迁移策略"""
        # 确保相似度矩阵已计算
        if self.similarity_matrix is None:
            logging.info("相似度矩阵未计算，正在计算...")
            self.calculate_similarity_matrix()
            
        # 检查相似度矩阵是否为空
        if self.similarity_matrix is None or len(self.similarity_matrix) == 0:
            logging.warning("相似度矩阵为空，无法生成迁移策略")
            return {}, False
        
        client_ids = list(self.client_models.keys())
        n_clients = len(client_ids)
        
        # 记录客户端数量
        logging.info(f"生成迁移策略，共 {n_clients} 个客户端")
        
        self.migration_strategies = {}
        
        # 为每个客户端生成迁移策略
        for client_id in client_ids:
            # 初始化搜索空间
            if client_id not in self.search_spaces:
                self.search_spaces[client_id] = self._initialize_search_space(client_id)
                
            # 使用启发式搜索算法找到最优迁移策略
            best_strategy = self._heuristic_search(
                client_id, 
                self.search_spaces[client_id], 
                performance_metrics.get(client_id, {}) if performance_metrics else {},
                round_num
            )
            
            # 动态调整迁移权重，防止性能下降
            if performance_metrics and client_id in performance_metrics:
                best_strategy = self.adjust_migration_weights(
                    client_id, 
                    best_strategy, 
                    performance_metrics.get(client_id, {})
                )
            
            self.migration_strategies[client_id] = best_strategy
            
            # 记录迁移信息
            logging.info(f"客户端 {client_id} 的迁移策略:")
            logging.info(f"- 迁移模型数量: {len(best_strategy['sources'])}")
            logging.info(f"- 置信度阈值: {best_strategy['confidence_threshold']:.2f}")
            logging.info(f"- 源客户端: {best_strategy['sources']}")
            logging.info(f"- 权重: {[f'{w:.2f}' for w in best_strategy['weights']]}")
        
        # 检查终止条件
        is_converged = self.check_convergence(performance_metrics)
        
        return self.migration_strategies, is_converged
    
    def _initialize_search_space(self, client_id):
        """初始化客户端的搜索空间"""
        client_ids = list(self.client_models.keys())
        client_idx = client_ids.index(client_id)
        
        # 获取相似度排序
        similarities = self.similarity_matrix[client_idx] if client_idx < len(self.similarity_matrix) else np.zeros(len(client_ids))
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 过滤掉自己
        sorted_indices = [idx for idx in sorted_indices if client_ids[idx] != client_id]
        
        # 初始化搜索空间
        search_space = {
            'source_candidates': [client_ids[idx] for idx in sorted_indices],
            'threshold_candidates': np.linspace(self.min_confidence_threshold, self.max_confidence_threshold, 5),
            'max_sources_candidates': list(range(1, min(self.max_migration_models + 1, len(sorted_indices) + 1))),
            'best_reward': -float('inf'),
            'best_strategy': None,
            'exploration_rate': 0.3,  # 探索率
            'history': []  # 历史搜索记录
        }
        
        return search_space
    
    def _heuristic_search(self, client_id, search_space, performance_metrics, round_num):
        """使用启发式搜索算法找到最优迁移策略"""
        client_ids = list(self.client_models.keys())
        
        # 如果是第一轮或没有性能指标，使用基于相似度的初始策略
        if round_num <= 1 or not performance_metrics:
            return self._generate_initial_strategy(client_id, search_space)
        
        # 获取当前性能
        current_performance = performance_metrics.get('accuracy', 0) / 100
        
        # 更新客户端历史记录
        if client_id not in self.client_history:
            self.client_history[client_id] = {
                'performance_history': [],
                'improvement_rate': [],
                'migration_history': []
            }
        
        history = self.client_history[client_id]
        history['performance_history'].append(current_performance)
        
        # 计算性能提升率
        if len(history['performance_history']) > 1:
            improvement = current_performance - history['performance_history'][-2]
            history['improvement_rate'].append(improvement)
        else:
            improvement = 0
            
        # 高性能模型保护机制：如果性能已经很高，考虑不进行迁移
        if current_performance > 0.9:  # 90%以上的准确率
            # 如果性能下降，完全停止迁移
            if improvement < 0:
                logging.info(f"客户端 {client_id} 性能已经很高 ({current_performance:.2f}) 且出现下降趋势，停止迁移")
                return {'sources': [], 'weights': [], 'confidence_threshold': 0.5}
            
            # 即使性能提升，也大幅降低探索率
            search_space['exploration_rate'] = max(0.15, search_space['exploration_rate'] * 0.2)  # 修改：最小值从0.01改为0.15
            logging.info(f"客户端 {client_id} 性能已经很高 ({current_performance:.2f})，降低探索率至 {search_space['exploration_rate']:.2f}")
            
        # 根据历史性能调整探索率
        if len(history['improvement_rate']) > 0:
            avg_improvement = np.mean(history['improvement_rate'][-3:] if len(history['improvement_rate']) >= 3 else history['improvement_rate'])
            # 如果性能提升停滞，增加探索率
            if avg_improvement < 0.01:
                search_space['exploration_rate'] = min(0.5, search_space['exploration_rate'] + 0.1)
            # 如果性能提升显著，减少探索率
            elif avg_improvement > 0.05:
                search_space['exploration_rate'] = max(0.15, search_space['exploration_rate'] - 0.1)  # 修改：最小值从0.1改为0.15
        
        # 如果当前性能已经很高，大幅降低探索率
        if current_performance > 0.95:  # 95%以上的准确率
            search_space['exploration_rate'] = max(0.15, search_space['exploration_rate'] * 0.5)  # 修改：最小值从0.05改为0.15
            logging.info(f"客户端 {client_id} 性能已经很高 ({current_performance:.2f})，降低探索率至 {search_space['exploration_rate']:.2f}")
        
        # 如果性能下降，减少探索
        if improvement < -0.05:  # 性能下降超过5%
            search_space['exploration_rate'] = max(0.15, search_space['exploration_rate'] * 0.5)  # 修改：最小值从0.05改为0.15
            logging.info(f"客户端 {client_id} 性能下降 ({improvement:.2f})，降低探索率至 {search_space['exploration_rate']:.2f}")
        
        # 决定是探索还是利用
        if np.random.random() < search_space['exploration_rate']:
            # 探索：尝试新的策略
            strategy = self._explore_new_strategy(client_id, search_space)
            logging.info(f"客户端 {client_id} 正在探索新策略")
        else:
            # 利用：使用历史最佳策略
            if search_space['best_strategy'] is not None:
                strategy = search_space['best_strategy']
                logging.info(f"客户端 {client_id} 正在利用历史最佳策略")
            else:
                strategy = self._explore_new_strategy(client_id, search_space)
                logging.info(f"客户端 {client_id} 没有历史最佳策略，正在探索")
        
        # 计算当前策略的奖励
        reward = self._calculate_reward(client_id, strategy, performance_metrics, improvement)
        
        # 更新历史记录
        search_space['history'].append({
            'round': round_num,
            'strategy': strategy,
            'reward': reward,
            'performance': current_performance
        })
        
        # 更新最佳策略
        if reward > search_space['best_reward']:
            search_space['best_reward'] = reward
            search_space['best_strategy'] = strategy
            logging.info(f"客户端 {client_id} 更新了最佳策略，奖励: {reward:.4f}")
        
        # 返回当前选择的策略
        return strategy
    
    def _generate_initial_strategy(self, client_id, search_space):
        """生成基于相似度的初始策略"""
        client_ids = list(self.client_models.keys())
        
        # 选择置信度阈值
        threshold = self.min_confidence_threshold
        
        # 选择迁移源数量
        max_sources = min(3, len(search_space['source_candidates']))
        
        # 选择满足置信度阈值的相似客户端
        sources = []
        weights = []
        
        for source_id in search_space['source_candidates']:
            if len(sources) >= max_sources:
                break
                
            client_idx = client_ids.index(client_id)
            source_idx = client_ids.index(source_id)
            
            if client_idx < len(self.similarity_matrix) and source_idx < len(self.similarity_matrix[client_idx]):
                similarity = self.similarity_matrix[client_idx][source_idx]
                if similarity >= threshold:
                    sources.append(source_id)
                    weights.append(similarity)
        
        # 确保至少有一个迁移源
        if not sources and len(search_space['source_candidates']) > 0:
            best_source = search_space['source_candidates'][0]
            sources.append(best_source)
            
            client_idx = client_ids.index(client_id)
            source_idx = client_ids.index(best_source)
            
            if client_idx < len(self.similarity_matrix) and source_idx < len(self.similarity_matrix[client_idx]):
                similarity = self.similarity_matrix[client_idx][source_idx]
                weights.append(max(0.1, similarity))
            else:
                weights.append(0.1)
        
        # 归一化权重
        if weights:
            total = sum(weights)
            weights = [w/total for w in weights]
        
        return {
            'sources': sources,
            'weights': weights,
            'confidence_threshold': threshold
        }
    
    def _explore_new_strategy(self, client_id, search_space):
        """探索新的迁移策略"""
        # 随机选择置信度阈值
        threshold = np.random.choice(search_space['threshold_candidates'])
        
        # 随机选择迁移源数量
        max_sources = np.random.choice(search_space['max_sources_candidates'])
        
        # 随机选择迁移源
        available_sources = search_space['source_candidates']
        if len(available_sources) > max_sources:
            # 优先选择相似度高的客户端
            sources = available_sources[:max_sources]
            # 随机替换一些源以增加多样性
            if np.random.random() < 0.5 and len(available_sources) > max_sources + 1:
                replace_idx = np.random.randint(0, max_sources)
                new_idx = np.random.randint(max_sources, len(available_sources))
                sources[replace_idx] = available_sources[new_idx]
        else:
            sources = available_sources.copy()
        
        # 计算权重
        weights = []
        client_ids = list(self.client_models.keys())
        
        for source_id in sources:
            client_idx = client_ids.index(client_id)
            source_idx = client_ids.index(source_id)
            
            if client_idx < len(self.similarity_matrix) and source_idx < len(self.similarity_matrix[client_idx]):
                similarity = self.similarity_matrix[client_idx][source_idx]
                weights.append(max(0.1, similarity))
            else:
                weights.append(0.1)
        
        # 归一化权重
        if weights:
            total = sum(weights)
            weights = [w/total for w in weights]
        
        return {
            'sources': sources,
            'weights': weights,
            'confidence_threshold': threshold
        }
    
    def _calculate_reward(self, client_id, strategy, performance_metrics, improvement):
        """计算迁移策略的奖励值"""
        # 基础奖励：性能提升
        if improvement >= 0:
            reward = improvement * 10  # 放大性能提升的影响
        else:
            # 对性能下降进行更严厉的惩罚
            reward = improvement * 30  # 更大的惩罚系数
        
        # 如果性能已经很高，且出现下降，额外惩罚
        if 'accuracy' in performance_metrics:
            current_performance = performance_metrics['accuracy'] / 100
            if current_performance > 0.9 and improvement < 0:
                reward -= abs(improvement) * 20  # 额外惩罚
        
        # 奖励因子：迁移源数量
        # 迁移源越少越好（通信开销小）
        source_factor = 1.0 - (len(strategy['sources']) / self.max_migration_models)
        reward += source_factor * 0.2
        
        # 奖励因子：相似度
        # 相似度越高越好
        similarity_sum = 0
        client_ids = list(self.client_models.keys())
        client_idx = client_ids.index(client_id)
        
        for source_id in strategy['sources']:
            source_idx = client_ids.index(source_id)
            if client_idx < len(self.similarity_matrix) and source_idx < len(self.similarity_matrix[client_idx]):
                similarity_sum += self.similarity_matrix[client_idx][source_idx]
        
        if strategy['sources']:
            avg_similarity = similarity_sum / len(strategy['sources'])
            reward += avg_similarity * 0.3
        
        # 奖励因子：当前性能
        # 性能越低，奖励越高（鼓励帮助性能差的客户端）
        if 'accuracy' in performance_metrics:
            current_performance = performance_metrics['accuracy'] / 100
            # 如果性能已经很高，减少迁移的动力
            if current_performance > 0.95:  # 95%以上的准确率
                reward -= 0.5  # 减少迁移动力
            else:
                performance_factor = 1.0 - current_performance
                reward += performance_factor * 0.1
        
        # 奖励因子：资源消耗
        # 资源消耗越低越好
        if client_id in self.client_resources:
            resources = self.client_resources[client_id]
            if 'device_resources' in resources:
                resource_factor = resources['device_resources'] / 100  # 假设资源范围是0-100
                reward += resource_factor * 0.1
        
        return reward
        
    def check_convergence(self, client_performances):
        """检查是否收敛 - 但不会真正停止训练"""
        # 计算当前轮次的平均性能
        current_accuracy = np.mean([perf['accuracy'] for perf in client_performances.values()])
        current_loss = np.mean([perf['loss'] for perf in client_performances.values()])
    
        # 记录最佳损失
        if current_loss < self.best_global_loss:
            self.best_global_loss = current_loss
            self.non_improving_rounds = 0
        else:
            self.non_improving_rounds += 1
    
        # 检查收敛条件
        if (self.non_improving_rounds >= self.patience and 
            current_accuracy >= self.min_accuracy):
            logging.info(f"检测到训练收敛（连续{self.non_improving_rounds}轮无改善，"
                        f"当前准确率: {current_accuracy:.2f}%），但将继续训练")
            # 返回False以确保训练继续进行
            return False
    
        return False


    def adjust_migration_weights(self, client_id, strategy, performance_metrics):
        """动态调整迁移权重，防止性能下降"""
        # 如果没有性能指标或没有历史记录，直接返回修改后的策略
        if not strategy['sources']:
            return strategy
            
        # 使用集中定义的权重
        self_weight = self.self_weight
        migration_weight = self.migration_weight
        
        # 调整迁移源的权重，保持它们之间的相对比例
        original_weights = strategy['weights']
        if sum(original_weights) > 0:
            # 归一化原始权重
            normalized_weights = [w/sum(original_weights) for w in original_weights]
            # 应用迁移总权重
            adjusted_weights = [w * migration_weight for w in normalized_weights]
        else:
            # 如果原始权重和为0，平均分配迁移权重
            adjusted_weights = [migration_weight / len(strategy['sources']) for _ in strategy['sources']]
        
        # 创建新策略，包含自身权重信息
        adjusted_strategy = {
            'sources': strategy['sources'],
            'weights': adjusted_weights,
            'confidence_threshold': strategy['confidence_threshold'],
            'self_weight': self_weight  # 添加自身权重
        }
        
        logging.info(f"客户端 {client_id} 应用固定权重分配: 自身 {self_weight:.2f}, 迁移 {migration_weight:.2f}")
        return adjusted_strategy

    def get_migration_models(self, client_id, strategy):
        """根据迁移策略获取迁移模型"""
        migration_models = []
        
        # 如果没有迁移源，返回空列表
        if not strategy['sources']:
            return migration_models
            
        # 获取迁移源模型
        for i, source_id in enumerate(strategy['sources']):
            if source_id in self.client_models:
                # 获取模型状态字典
                model_state_dict = self.client_models[source_id]
                
                # 获取权重
                weight = strategy['weights'][i] if i < len(strategy['weights']) else 0.0
                
                # 添加到迁移模型列表
                migration_models.append({
                    'source_id': source_id,
                    'model_state_dict': model_state_dict,
                    'weight': weight
                })
            else:
                logging.warning(f"迁移源 {source_id} 的模型不存在")
        
        return migration_models

    def apply_migration(self, client_id, client_model, strategy):
        """应用迁移策略，更新客户端模型"""
        # 如果没有迁移源，直接返回原模型
        if not strategy['sources']:
            logging.info(f"客户端 {client_id} 没有迁移源，保持原模型不变")
            return client_model
            
        # 获取迁移模型
        migration_models = self.get_migration_models(client_id, strategy)
        
        # 如果没有获取到迁移模型，直接返回原模型
        if not migration_models:
            logging.warning(f"客户端 {client_id} 没有获取到迁移模型，保持原模型不变")
            return client_model
            
        # 获取自身权重
        self_weight = strategy.get('self_weight', self.self_weight)
        
        # 创建新模型
        new_model = copy.deepcopy(client_model)
        new_model.to('cpu')  # 确保在CPU上进行模型融合
        
        # 获取原模型状态字典
        client_state_dict = client_model.state_dict()
        
        # 创建融合后的状态字典
        fused_state_dict = {}
        
        # 对每个参数进行加权平均
        for key in client_state_dict:
            # 初始化为自身权重 * 自身参数
            fused_state_dict[key] = self_weight * client_state_dict[key].clone().detach().cpu()
            
            # 添加迁移模型的贡献
            for migration_model in migration_models:
                weight = migration_model['weight']
                model_state_dict = migration_model['model_state_dict']
                
                if key in model_state_dict:
                    # 确保参数在CPU上
                    param = model_state_dict[key].clone().detach().cpu()
                    
                    # 检查形状是否匹配
                    if param.shape == fused_state_dict[key].shape:
                        fused_state_dict[key] += weight * param
                    else:
                        logging.warning(f"参数 {key} 的形状不匹配: {param.shape} vs {fused_state_dict[key].shape}")
        
        # 加载融合后的状态字典
        new_model.load_state_dict(fused_state_dict)
        
        # 将模型移回原设备
        device = next(client_model.parameters()).device
        new_model.to(device)
        
        logging.info(f"客户端 {client_id} 完成模型迁移，自身权重: {self_weight:.2f}, 迁移源数量: {len(migration_models)}")
        
        return new_model

    def update_client_budgets(self, client_id, strategy):
        """更新客户端的通信和计算预算"""
        # 如果客户端不在预算字典中，初始化
        if client_id not in self.client_comm_budgets:
            self.client_comm_budgets[client_id] = self.communication_budget
            
        if client_id not in self.client_comp_budgets:
            self.client_comp_budgets[client_id] = self.computation_budget
            
        # 计算通信开销
        comm_cost = len(strategy['sources']) * self.model_size
        
        # 计算计算开销
        comp_cost = 1.0  # 基础计算开销
        
        # 更新预算
        self.client_comm_budgets[client_id] -= comm_cost
        self.client_comp_budgets[client_id] -= comp_cost
        
        # 检查预算是否耗尽
        comm_exhausted = self.client_comm_budgets[client_id] <= 0
        comp_exhausted = self.client_comp_budgets[client_id] <= 0
        
        if comm_exhausted:
            logging.warning(f"客户端 {client_id} 通信预算耗尽")
            
        if comp_exhausted:
            logging.warning(f"客户端 {client_id} 计算预算耗尽")
            
        return not (comm_exhausted or comp_exhausted)

    def reset(self):
        """重置相似度管理器的状态"""
        # 重置相似度矩阵
        self.similarity_matrix = None
        
        # 重置搜索空间
        self.search_spaces = {}
        
        # 重置奖励记录
        self.rewards = {}
        
        # 重置收敛判断参数
        self.best_global_loss = float('inf')
        self.non_improving_rounds = 0
        
        # 重置客户端预算
        self.client_comm_budgets = {}
        self.client_comp_budgets = {}
        
        # 重置随机输入
        self.random_inputs = None
        
        logging.info("相似度管理器已重置")
