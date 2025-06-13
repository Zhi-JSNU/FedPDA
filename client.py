import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim.lr_scheduler
import numpy as np
import logging
from sklearn.metrics import f1_score
import copy

class Client:
    def __init__(self, client_id, model, device, train_data, test_data, val_data=None, learning_rate=0.01, model_type=None, epochs=5):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.epochs = epochs
        
        # 初始化基本属性
        self.attributes = {
            'device_type': np.random.choice(['high', 'medium', 'low']),
            'compute_power': 0.0,
            'device_resources': np.random.randint(30, 100),
            'data_distribution': None
        }
        
        # 初始化模型迁移相关属性
        self.received_models = []
        self.migration_weights = []
        
        # 初始化服务器引用
        self.server = None
        
        # 初始化性能指标
        self.metrics = {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'loss': 0.0,
            'converged': False
        }
        
        # 初始化优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,  # 添加动量
            weight_decay=1e-4,  # 保持相同的权重衰减
            nesterov=True  # 使用Nesterov动量
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-5,   # 设置最小学习率，避免学习率过低
            verbose=True
        )
        
        # 计算计算能力
        self.attributes['compute_power'] = self._estimate_compute_power()
        
        # 获取数据集特征
        sample_data = next(iter(train_data))[0]
        self.input_size = sample_data.size(-1)
        self.input_channels = sample_data.size(1)

        

    def _train_epoch(self, epoch, epochs):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_data):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            
            # 收集所有预测和目标
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        # 在整个epoch结束后计算准确率
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = 100. * np.mean(all_preds == all_targets)
        avg_loss = total_loss / len(self.train_data)

        # 使用scheduler调整学习率
        self.scheduler.step(avg_loss)
        
        return avg_loss, accuracy

    def train(self, epochs=None, after_global_agg=False):
        """训练方法"""
        if epochs is None:
            epochs = self.epochs
            
        self.model = self.model.to(self.device)
        logging.info(f"\n客户端 {self.client_id} 开始训练...")
        
        if not after_global_agg and self.received_models:
            self._aggregate_received_models()
        
        if after_global_agg:
            self._reinitialize_optimizer()
        
        train_losses = []
        train_accs = []
        
        for epoch in range(epochs):
            epoch_loss, epoch_acc = self._train_epoch(epoch, epochs)
            eval_metrics = self.evaluate()
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            logging.info(f"客户端 {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, F1: {eval_metrics['f1_score']:.4f}")
        
        self.metrics.update(eval_metrics)
        self.metrics['train_accuracy'] = train_accs[-1]
        
        logging.info(f"客户端 {self.client_id} 评估 - 损失: {eval_metrics['loss']:.4f}, "
                    f"准确率: {eval_metrics['accuracy']:.2f}%, "
                    f"F1分数: {eval_metrics['f1_score']:.4f}")
        
        self.received_models = []
        self.migration_weights = []
        
        return self.metrics

    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.view(-1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = test_loss / len(self.test_data)
        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'loss': avg_loss
        }

    def receive_model(self, source_model, weight):
        """接收其他客户端的模型"""
        # 深拷贝并确保在正确的设备上
        model_copy = copy.deepcopy(source_model)
        # 注意：这里不立即转换设备，因为可能会占用大量GPU内存
        # 在聚合时再转换设备
        self.received_models.append(model_copy)
        self.migration_weights.append(weight)
        logging.info(f"客户端 {self.client_id} 接收到新模型，当前共有 {len(self.received_models)} 个待聚合模型")

    def _aggregate_received_models(self):
        """聚合接收到的模型"""
        if not self.received_models:
            return
            
        try:
            self_weight = 0.7
            total_weight = sum(self.migration_weights)
            normalized_weights = [w/(total_weight + self_weight) for w in self.migration_weights]
            normalized_self_weight = self_weight/(total_weight + self_weight)
            
            logging.info(f"客户端 {self.client_id} 开始聚合模型，自身权重: {normalized_self_weight:.2f}, 迁移模型数量: {len(self.received_models)}")
            
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            own_state_dict = {}
            
            # 先复制自己的模型参数
            for k, v in self.model.state_dict().items():
                own_state_dict[k] = v.clone().to(self.device) * normalized_self_weight
            
            # 逐个聚合其他模型
            for i, (model, weight) in enumerate(zip(self.received_models, normalized_weights)):
                # 确保模型在正确的设备上
                model = model.to(self.device)
                state_dict = model.state_dict()
                
                for key in own_state_dict:
                    if key in state_dict:
                        # 确保张量在同一设备上
                        tensor = state_dict[key].to(self.device)
                        
                        if own_state_dict[key].dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                            own_state_dict[key] += tensor * weight
                        else:
                            if weight > normalized_self_weight:
                                own_state_dict[key] = tensor
            
            # 加载聚合后的参数
            self.model.load_state_dict(own_state_dict)
            logging.info(f"客户端 {self.client_id} 完成模型聚合")
            
        except Exception as e:
            logging.error(f"客户端 {self.client_id} 模型聚合失败: {str(e)}")
        
        self.received_models = []
        self.migration_weights = []

    def _estimate_compute_power(self):
        """估计计算能力"""
        if self.attributes['device_type'] == 'high':
            return np.random.uniform(0.8, 1.0)
        elif self.attributes['device_type'] == 'medium':
            return np.random.uniform(0.5, 0.8)
        else:
            return np.random.uniform(0.2, 0.5)

    def _reinitialize_optimizer(self):
        """在全局聚合后重新初始化优化器"""
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        logging.info(f"客户端 {self.client_id} 在全局聚合后重新初始化了优化器")

    def set_global_model(self, global_model_state):
        """设置全局模型参数"""
        # 确保所有参数都在正确的设备上
        for key in global_model_state:
            global_model_state[key] = global_model_state[key].to(self.device)
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        self.model.load_state_dict(global_model_state)
        self._reinitialize_optimizer()
        logging.info(f"客户端 {self.client_id} 已接收并设置全局模型参数")
        
    def get_data_distribution(self):
        """获取客户端训练数据的分布"""
        # 从训练数据加载器中获取所有标签
        all_labels = []
        
        # 保存原始的训练模式
        was_training = self.model.training
        self.model.eval()
        
        # 遍历数据加载器获取标签
        with torch.no_grad():
            for _, labels in self.train_data:
                all_labels.append(labels.cpu().numpy())
        
        # 恢复原始训练模式
        if was_training:
            self.model.train()
        
        # 合并所有批次的标签
        if all_labels:
            all_labels = np.concatenate(all_labels)
            
            # 计算分布
            num_classes = 10  # 假设有10个类别，可以根据实际情况调整
            distribution = np.zeros(num_classes)
            
            # 计算每个类别的样本数量
            unique, counts = np.unique(all_labels, return_counts=True)
            for i, count in zip(unique, counts):
                if i < num_classes:
                    distribution[i] = count
            
            # 归一化分布
            if distribution.sum() > 0:
                distribution = distribution / distribution.sum()
            
            return distribution
        else:
            # 如果没有数据，返回均匀分布
            return np.ones(10) / 10

    # 在模型加载部分添加检查
    def load_model(self, model_state_dict):
        """加载模型参数"""
        try:
            # 确保模型和权重在同一设备上
            for key in model_state_dict:
                model_state_dict[key] = model_state_dict[key].to(self.device)
            
            # 检查模型的第一个卷积层权重形状
            if 'conv1.weight' in model_state_dict:
                conv1_weight = model_state_dict['conv1.weight']
                logging.info(f"加载模型 - conv1.weight 形状: {conv1_weight.shape}")
                
                if conv1_weight.shape[1] != self.model.conv1.weight.shape[1]:
                    logging.warning(f"输入通道数不匹配: 模型期望 {self.model.conv1.weight.shape[1]}，但权重有 {conv1_weight.shape[1]} 通道")
                    
                    # 创建新的权重张量（确保在正确的设备上）
                    new_weight = torch.zeros_like(self.model.conv1.weight).to(self.device)
                    
                    if conv1_weight.shape[1] > self.model.conv1.weight.shape[1]:
                        new_weight = conv1_weight[:, :self.model.conv1.weight.shape[1], :, :]
                    else:
                        for i in range(self.model.conv1.weight.shape[1]):
                            idx = i % conv1_weight.shape[1]
                            new_weight[:, i, :, :] = conv1_weight[:, idx, :, :]
                    
                    model_state_dict['conv1.weight'] = new_weight
            
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            # 加载模型参数
            self.model.load_state_dict(model_state_dict, strict=False)
            logging.info(f"客户端 {self.client_id} 成功加载模型参数")
            return True
            
        except Exception as e:
            logging.error(f"客户端 {self.client_id} 加载模型参数失败: {e}")
            return False




      
